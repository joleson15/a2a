# Agent 2 Agent Protocol (A2A) - a thread:

### What is A2A?
 As the sophistication of AI systems grows, so has the ability to automate tasks. However, scaling multi-agent systems has been a major bottleneck for many engineers. Frameworks like langgraph seek to solve this problem, but the framework is not very scalable and the api's don't always fit the needs of developers, can become outdated, and may introduce breaking changes...hence the need for custom agents, and modularity. A2A is an effort by Google to standardize inter-agent communication and allow developers to integrate the capabilities of various frameworks. 

The probabalistic origins of machine learning and AI models cause a significant amount of non-determinism in the outputs, on the contrary to traditional algorithms that provide a deterministic output (i.e. given the same input, it always produces the same output). Because of this, we can't always predict the output of a machine learning algorithm. So maybe it is obvious that in order to allow these systems to share information, the outputs must be wrapped in some metadata to provide some scaffolding for a message format, even though the messages themselves are always unique. 


See also MCP, a protocol introduced by Anthropic to provide a standard of communication between AI models and tools, data, prompts, and other resources. While MCP allows the AI "agent" external functionality like querying a database, searching the internet, or executing a workflow, A2A allows multiple specialized AI agents to communicate with each other and complete complex tasks by utilizing their variety of skills. 

Example: Google Search Agent, Database Query agent, summarizer agent, analysis agent.

_A2A is a protocol to allow AI agents to communicate autonomously amongst themselves in a standardized way. It was introduced by Google in April 2025 and donated to the Linux Foundation for more widespread adoption._

### The Agent

So what is an Agent? While the A2A protocol seems like it has been standardized, the definition of "agent" certainly has not. Langchain and Microsoft define an [agent](https://langchain-ai.github.io/langgraph/agents/overview/) as "an LLM + tools + a prompt". Google has more broadly stated that an AI agent is a software system that uses AI (not just LLMs) to proactively pursue goals and complete tasks. This allows us to expand the scope of agents to include more specialized machine learning algorithms, although the implementation of Google's agent development kit ([ADK](https://google.github.io/adk-docs/)) indicates that they really fall into the first camp. For now, I will focus on the "LLM + tools + prompt" definition, but stay tuned for more exploration into nuanced algorithms in the future. 

Let's take a look at our first "agent"...

**Prerequisites**:
- Pyton 3.11+
- [Gemini API Key](https://aistudio.google.com/apikey)


```python
import os
from dotenv import load_dotenv

load_dotenv()
```

<!-- If you would like to follow along, and don't want to copy and paste -->

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

my_first_agent = Agent(
    model="gemini-2.5-flash",
    name="My_first_agent",
    description="A simple agent that can call a google search",
    instruction="You are a helpful google search agent. Conduct a search when you determine it is necessary to do so.",
    tools=[google_search]
)
```

As you can see, there is not much effort to declare your first agent. Google's ADK abstracts away a lot of the headache, it's details are out of scope for this work, but will be covered in a future blog post. Each of the supplied parameters are self-descriptive:
- Model: the large language model used as the intelligence engine for your agent
- Name: a descriptive name, can only contain letters, numbers and underscores
- Description: a helpful description can go a long way, especially once multiple agents are involved in a complex process
- Instruction: an optional system prompt that gets passed in as context to the LLM that contains any additional information or instructions
- Tools: a list of tools an agent is capable of using. In this case we have only outfitted our agent with the ability to conduct a google search

Now we have a primitive agent. So how do we handle messages?

### Agent Executor
The Agent Executor interface is contained in the a2a library, and provides a wrapper for the Agent to handle requests. The interface contains two main methods: `execute()`, which handles the main execution logic of the Agent's runtime and `cancel()`, which can be called during the execution of a long-running task. 

```python
class AgentExecutor(ABC):
    """Agent Executor interface.

    Implementations of this interface contain the core logic of the agent,
    executing tasks based on requests and publishing updates to an event queue.
    """

    @abstractmethod
    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute the agent's logic for a given request context.

        The agent should read necessary information from the `context` and
        publish `Task` or `Message` events, or `TaskStatusUpdateEvent` /
        `TaskArtifactUpdateEvent` to the `event_queue`. This method should
        return once the agent's execution for this request is complete or
        yields control (e.g., enters an input-required state).

        Args:
            context: The request context containing the message, task ID, etc.
            event_queue: The queue to publish events to.
        """

    @abstractmethod
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Request the agent to cancel an ongoing task.

        The agent should attempt to stop the task identified by the task_id
        in the context and publish a `TaskStatusUpdateEvent` with state
        `TaskState.canceled` to the `event_queue`.

        Args:
            context: The request context containing the task ID to cancel.
            event_queue: The queue to publish the cancellation status update to.
        """

```
For now, we will focus only on event execution and leave cancellation to a future blog. Google provides an implementation of `execute()` for a generic ADK agent in their [A2A Samples](https://github.com/a2aproject) repo that I will borrow for this example.


The `execute()` method concurrently processes requests by keeping track of them with an `EventQueue`. Requests are presented in the form of a `RequestContext`, which contains the content of the request, along with other metadata. That is enough info for now, we will get into the weeds in future work.

A critical component to the successful operation of an Agent is the `session`. A session is a stateful container that allows agents to interact asynchronously. It manages the ongoing interaction between the agents and/or users, context such as memory and state, and a request/response loop to handle communication and execution of actions.




```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_task, new_agent_text_message
from a2a.types import TaskState, TextPart, Part

from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

class MyAgentExecutor(AgentExecutor):

    def __init__(self, agent: Agent, status_message: str = "Executing task...", artifact_name: str = "response"):
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name
        self.runner = Runner(
            app_name=agent.name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Update status with custom message
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(self.status_message, task.contextId, task.id)
            )

            # Process with ADK agent
            session = await self.runner.session_service.create_session(
                app_name=self.agent.name,
                user_id="a2a_user",
                state={},
                session_id=task.contextId,
            )

            content = types.Content(
                role='user',
                parts=[types.Part.from_text(text=query)]
            )

            response_text = ""
            async for event in self.runner.run_async(
                user_id="a2a_user",
                session_id=session.id,
                new_message=content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text + '\n'
                        elif hasattr(part, 'function_call'):
                            # Log or handle function calls if needed
                            pass  # Function calls are handled internally by ADK

            # Add response as artifact with custom name
            await updater.add_artifact(
                [Part(root=TextPart(text=response_text))],
                name=self.artifact_name
            )

            await updater.complete()

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e!s}", task.contextId, task.id),
                final=True
            )


    def cancel(self, context, event_queue):
        pass
```

Now that we can run an agent, how do we (and more importantly other agents, since this is a blog about A2A after all...) know what its capabilities are?

### Agent Card
The agent card is a public-facing JSON schema that exposes information and metadata to clients (users and other agents). It is like a user profile, but for AI agents. To define an agent card properly, you need:
- `name`
- `description`: description of the agent, its skills, and other useful information
- `version`
- `url`: the web endpoint where we can find our agent
- `capabilities`: supported A2A features like streaming or push notifications
- `skills`: A pillar of A2A, the skills discovered in the `AgentCard` come in a list of `AgentSkill` objects.

For public agents, the A2A project reccomends that the agent card be discoverable at a well-known uri. The standard discovery path is: `https://{server-url}/.well-known/agent.json` This is the discovery method we will use for now, but for more information, see the [A2A discovery](https://a2aproject.github.io/A2A/latest/topics/agent-discovery/#the-role-of-the-agent-card) page.

Lets give our agent an `AgentCard`:


```python
from a2a.types import AgentCard, AgentCapabilities

agent_card = AgentCard(
    name=my_first_agent.name,
    description=my_first_agent.description,
    url="http://localhost:9999/",
    version='1.0',
    capabilities=AgentCapabilities(
        streaming=True
    ),
    defaultInputModes=["text", "text/plain"],
    defaultOutputModes=["text", "text/plain"],
    skills=[]
)
```

### Agent Skills
Agent skills describe specific capabilities the agent has, like searching the web, querying a database, executing an algorithmm...etc. Clients can find out what skills an agent has from the `AgentCard`. It's kind of like the agentic version of a resume. Skills have some attributes to define:
- `id`: a unique id
- `name`
- `description`: more detailed information about the skill's functionality
- `tags`: keywords
- `examples`: example usage of the skill
- `inputModes` and `outputModes`: supported modes for input and output, like text or json

Let's go back and define the Google search `AgentSkill` for our agent:


```python
from a2a.types import AgentSkill

web_search_skill = AgentSkill(
    id='google_search',
    name='Google Search',
    description='Searches the web using the google_search tool',
    tags=['web search', 'google', 'search', 'look up']
)

agent_card.skills.append(web_search_skill)
```

Now our agent has advertized the ability to search the web on its agent card. We will take a deep dive into `AgentSkill`s, `AgentCapabilities`, and `AgentCard`s another time, but now lets zoom out a little bit. We have given lots of detail about our agent. Where it lives, what it can do, examples for how to use it... So how do we run it and start testing this stuff out?

### Starting the Server

A2A follows a client-server architecture, where the client--a user-facing application or agent--initiates a request to other agents acting as servers that handle those requests, similarly to a traditional web browser or API sending requests to a remote web server. They send structured metadata and information over HTTP using [JSON-RPC 2.0](https://www.jsonrpc.org/specification) as the format.

With all of the agent functionality defined thus far, let's deploy an `agent` as a `server`. The A2A library provides an out-of-the-box app using the [Starlette](https://www.starlette.io/) framework to implement the server endpoints and route JSON-RPC requests. This is perfect for our use-case.


```python
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication


def create_server():

    request_handler = DefaultRequestHandler(
        agent_executor=MyAgentExecutor(my_first_agent),
        task_store=InMemoryTaskStore()
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
```

The server application takes a `DefaultRequestHandler` as an argument, which is the engine for all A2A protocol methods, the `AgentExecutor`, the `TaskStore`, the `QueueManager`, and all other JSON-RPC related logic. Lets spin up the server. 

Since we are using jupyter, we need some fancy event handling so as to not have the server process cause the cell to run forever. We need to run it in a background process using a thread.


```python
import uvicorn
import asyncio
import nest_asyncio
import threading
import time

nest_asyncio.apply()

def run_agent():

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    server = create_server()
    config = uvicorn.Config(
        app=server.build(),
        host="127.0.0.1",
        port=9999,
        log_level="error"
    )

    uvicorn_server = uvicorn.Server(config)
    loop.run_until_complete(uvicorn_server.serve())


thread = threading.Thread(target=run_agent, daemon=True)
thread.start()

time.sleep(2)

print("Server is running at http://127.0.0.1:9999")
```

    Server is running at http://127.0.0.1:9999


**Quick note: for now, if we need to stop the server, we have to restart the jupyter kernel.**

Great news! We now have our very first agent up and running! So now what? We want to be able to interact with our AI agent and benefit from all of the cool stuff it can do. As I mentioned, A2A communication occurs over HTTP, so lets send a simple http request to our agent. We can start by fetching the `AgentCard` to see what our agent can do.

### Fetching the `AgentCard`


```python
import httpx
import json

async with httpx.AsyncClient() as http_client:
    response = await http_client.get("http://localhost:9999/.well-known/agent.json")
    agent_card_json = response.json()

print(json.dumps(agent_card_json, indent=2)) #json.dumps is completely optional but makes the output more readable
```

    {
      "capabilities": {
        "streaming": true
      },
      "defaultInputModes": [
        "text",
        "text/plain"
      ],
      "defaultOutputModes": [
        "text",
        "text/plain"
      ],
      "description": "A simple agent that can call a google search",
      "name": "My_first_agent",
      "protocolVersion": "0.2.5",
      "skills": [
        {
          "description": "Searches the web using the google_search tool",
          "id": "google_search",
          "name": "Google Search",
          "tags": [
            "web search",
            "google",
            "search",
            "look up"
          ]
        }
      ],
      "url": "http://localhost:9999/",
      "version": "1.0"
    }


Amazing! Just as we defined it earlier, the `AgentCard` is showing all of the information about our agent. To interact with our A2A server properly, we need an A2A client. 

```python
from a2a.client import A2AClient

my_first_agent_card = AgentCard(**agent_card_json)

a2a_client = A2AClient(
    httpx_client=http_client,
    agent_card=my_first_agent_card,
    url="http://localhost:9999/") 
```

`agent_card` and `url` parameters are not both required, but one of them is neeeded to fetch the url of the agent's RPC endpoint. With our client, we can send RPC messages to the A2A server (our agent) by sending an HTTP POST request to the RPC endpoint (the URL). So how do we structure this message and what is it going to look like?

### JSON RPC

Without diving too deep into the weeds of JSON-RPC 2.0, we can analyze the layout of a simple A2A message, encapsulated in a `JSONRPCRequest` object:
- `jsonrpc`: must be `2.0`
- `method`: `"message/send"`, `"tasks/get"` ...
- `params`: typically an `object` containing the parameters used for invocation
- `id`: unique identifier, not always required if the request does not expect a response.

More info [here](https://a2aproject.github.io/A2A/v0.2.5/specification/#611-json-rpc-structures). We will use the `message/send` method, and pass in our first message:

```python
rpc_request = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'What are some trending topics in AI right now?'}
        ],
        'messageId': uuid.uuid4().hex,
    }
}
```

The python sdk for A2A provides lots of useful helper methods that help properly structure our input. Here, we'll create a `SendMessageRequest` and pass in the `rpc_request` as the `params` to our `JSONRPCRequest` object using the `MessageSendParams` method:

```python
request = SendMessageRequest(
        id=str(uuid.uuid4()),
        params=MessageSendParams(**rpc_request)
    )
```

##### Optional: run the below cell to view the whole RPC Request


```python
import uuid
from a2a.types import SendMessageRequest, MessageSendParams

rpc_request = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'What are some trending topics in AI right now?'}
                ],
                'messageId': uuid.uuid4().hex,
            }
    }

request = SendMessageRequest(
    id=str(uuid.uuid4()),
    params=MessageSendParams(**rpc_request)
)

print(request)
```

    id='70c6ac85-0e2e-44d9-8183-a71ffaa846b9' jsonrpc='2.0' method='message/send' params=MessageSendParams(configuration=None, message=Message(contextId=None, extensions=None, kind='message', messageId='de7bfa7380ed4c909ad2ad0b22d888f3', metadata=None, parts=[Part(root=TextPart(kind='text', metadata=None, text='What are some trending topics in AI right now?'))], referenceTaskIds=None, role=<Role.user: 'user'>, taskId=None), metadata=None)


Tying everything together, we can send a properly formatted JSON RPC request to our server:


```python
import uuid
from a2a.types import SendMessageRequest, MessageSendParams
from a2a.client import A2AClient

timeout_config = httpx.Timeout(
            timeout=120.0,
            connect=10.0,
            read=120.0,
            write=10.0,
            pool=5.0
        )

async with httpx.AsyncClient(timeout=timeout_config) as http_client:

    response = await http_client.get("http://localhost:9999/.well-known/agent.json")
    agent_card_json = response.json()

    my_first_agent_card = AgentCard(**agent_card_json)

    a2a_client = A2AClient(
        httpx_client=http_client,
        agent_card=my_first_agent_card,
    )

    message = "What are some trending topics in AI right now?"

    rpc_request = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message}
                ],
                'messageId': uuid.uuid4().hex,
            }
    }

    request = SendMessageRequest(
        id=str(uuid.uuid4()),
        params=MessageSendParams(**rpc_request)
    )

    response = await a2a_client.send_message(request)
```

Just for fun, lets print out the response. We should expect to get back a response from our server in the form of a [JSONRPCResponse](https://a2aproject.github.io/A2A/latest/specification/#6112-jsonrpcresponse-object) object.


```python
print(response)
```

    root=SendMessageSuccessResponse(id='9d9dd71f-f5cf-4005-942a-5ecbdd474dbc', jsonrpc='2.0', result=Task(artifacts=[Artifact(artifactId='2c1511f7-200c-470b-8059-f9ca3819c348', description=None, extensions=None, metadata=None, name='response', parts=[Part(root=TextPart(kind='text', metadata=None, text='Several key themes and advancements are dominating the field of Artificial Intelligence (AI) in 2025, reflecting its rapid evolution and increasing integration across various sectors.\n\nSome of the most prominent trending topics in AI right now include:\n\n*   **Generative AI Evolution and Integration** Generative AI continues to be a major force, moving beyond basic content creation to more advanced and integrated applications. This includes the development of more sophisticated generative AI models, often referred to as Generative AI 2.0, that are becoming autonomous creators. There\'s a strong focus on integrating generative AI into everyday applications, making it an "obligatory copilot" for skilled workers across various industries. Multimodal AI, which can process and generate content from various inputs like text, images, and video, is also seeing significant advancements.\n*   **Agentic AI** A significant trend is the rise of "Agentic AI," where AI systems can act autonomously to achieve specific goals, make decisions, and take actions independently without constant human oversight. These AI agents are envisioned as the "apps of the AI era," handling tasks on users\' behalf and transforming business processes. Examples include AIs that can surf the web, order groceries, predict sales, and manage inventory.\n*   **AI in the Workplace and Productivity** AI\'s adoption in the workplace is accelerating, with a focus on enhancing productivity, automating repetitive and mundane tasks, and transforming work dynamics. AI is increasingly used for tasks like sifting through emails, taking notes, generating content, and streamlining job processes. While concerns about job displacement exist, AI is also seen as augmenting human roles and creating new job opportunities.\n*   **Specialized AI Applications** AI is being deeply embedded into specific industries, driving innovation and efficiency:\n    *   **Healthcare:** AI is accelerating scientific research and improving healthcare outcomes, with applications in diagnosing patients and identifying health problems.\n    *   **Finance:** AI is revolutionizing the finance landscape through autonomous finance, including customer service chatbots, automated forecasting, and AI-powered fraud detection.\n    *   **Supply Chain:** AI is enhancing efficiency in supply chain management through optimized inventory and logistics, risk prediction, and sustainability drives.\n    *   **Education:** AI is being explored for personalized learning experiences, although its implementation can be controversial.\n*   **Ethical AI and Regulation** As AI becomes more powerful and pervasive, there is increased scrutiny on AI ethics, responsible AI, and the development of broader AI regulations. This includes concerns about privacy, data breaches, and ensuring fundamental rights are respected, as seen with initiatives like the European Union\'s AI Act.\n*   **Technical Advancements** Underlying these applications are continuous technical improvements. This includes more reasonable reasoning models, advancements in smaller language models (SLMs) that offer efficiency benefits, and the emerging field of "Embodied AI," which aims to bring multimodal AI abilities into the physical world through robotics. There\'s also a focus on improving the cost-efficiency of AI inference and exploring new model architectures beyond traditional transformers.\n'))])], contextId='508efe3a-41ca-4e0b-9478-ea82dbce2ea6', history=[Message(contextId='508efe3a-41ca-4e0b-9478-ea82dbce2ea6', extensions=None, kind='message', messageId='7f67307cfede45438e5ed867c38bd9bb', metadata=None, parts=[Part(root=TextPart(kind='text', metadata=None, text='What are some trending topics in AI right now?'))], referenceTaskIds=None, role=<Role.user: 'user'>, taskId='fb816920-46e3-4bdf-8889-c7fdab377e7a'), Message(contextId='508efe3a-41ca-4e0b-9478-ea82dbce2ea6', extensions=None, kind='message', messageId='f2f05466-2943-4671-bd83-fe41eb2193d2', metadata=None, parts=[Part(root=TextPart(kind='text', metadata=None, text='Executing task...'))], referenceTaskIds=None, role=<Role.agent: 'agent'>, taskId='fb816920-46e3-4bdf-8889-c7fdab377e7a')], id='fb816920-46e3-4bdf-8889-c7fdab377e7a', kind='task', metadata=None, status=TaskStatus(message=None, state=<TaskState.completed: 'completed'>, timestamp='2025-07-17T00:37:42.277421+00:00')))


This is a bit tricky to parse through, since RPC calls come with a lot of attached metadata. Let's print just the json:


```python
response_dict = response.model_dump(mode='json', exclude_none=True)
print(json.dumps(response_dict, indent=2))
```

    {
      "id": "9d9dd71f-f5cf-4005-942a-5ecbdd474dbc",
      "jsonrpc": "2.0",
      "result": {
        "artifacts": [
          {
            "artifactId": "2c1511f7-200c-470b-8059-f9ca3819c348",
            "name": "response",
            "parts": [
              {
                "kind": "text",
                "text": "Several key themes and advancements are dominating the field of Artificial Intelligence (AI) in 2025, reflecting its rapid evolution and increasing integration across various sectors.\n\nSome of the most prominent trending topics in AI right now include:\n\n*   **Generative AI Evolution and Integration** Generative AI continues to be a major force, moving beyond basic content creation to more advanced and integrated applications. This includes the development of more sophisticated generative AI models, often referred to as Generative AI 2.0, that are becoming autonomous creators. There's a strong focus on integrating generative AI into everyday applications, making it an \"obligatory copilot\" for skilled workers across various industries. Multimodal AI, which can process and generate content from various inputs like text, images, and video, is also seeing significant advancements.\n*   **Agentic AI** A significant trend is the rise of \"Agentic AI,\" where AI systems can act autonomously to achieve specific goals, make decisions, and take actions independently without constant human oversight. These AI agents are envisioned as the \"apps of the AI era,\" handling tasks on users' behalf and transforming business processes. Examples include AIs that can surf the web, order groceries, predict sales, and manage inventory.\n*   **AI in the Workplace and Productivity** AI's adoption in the workplace is accelerating, with a focus on enhancing productivity, automating repetitive and mundane tasks, and transforming work dynamics. AI is increasingly used for tasks like sifting through emails, taking notes, generating content, and streamlining job processes. While concerns about job displacement exist, AI is also seen as augmenting human roles and creating new job opportunities.\n*   **Specialized AI Applications** AI is being deeply embedded into specific industries, driving innovation and efficiency:\n    *   **Healthcare:** AI is accelerating scientific research and improving healthcare outcomes, with applications in diagnosing patients and identifying health problems.\n    *   **Finance:** AI is revolutionizing the finance landscape through autonomous finance, including customer service chatbots, automated forecasting, and AI-powered fraud detection.\n    *   **Supply Chain:** AI is enhancing efficiency in supply chain management through optimized inventory and logistics, risk prediction, and sustainability drives.\n    *   **Education:** AI is being explored for personalized learning experiences, although its implementation can be controversial.\n*   **Ethical AI and Regulation** As AI becomes more powerful and pervasive, there is increased scrutiny on AI ethics, responsible AI, and the development of broader AI regulations. This includes concerns about privacy, data breaches, and ensuring fundamental rights are respected, as seen with initiatives like the European Union's AI Act.\n*   **Technical Advancements** Underlying these applications are continuous technical improvements. This includes more reasonable reasoning models, advancements in smaller language models (SLMs) that offer efficiency benefits, and the emerging field of \"Embodied AI,\" which aims to bring multimodal AI abilities into the physical world through robotics. There's also a focus on improving the cost-efficiency of AI inference and exploring new model architectures beyond traditional transformers.\n"
              }
            ]
          }
        ],
        "contextId": "508efe3a-41ca-4e0b-9478-ea82dbce2ea6",
        "history": [
          {
            "contextId": "508efe3a-41ca-4e0b-9478-ea82dbce2ea6",
            "kind": "message",
            "messageId": "7f67307cfede45438e5ed867c38bd9bb",
            "parts": [
              {
                "kind": "text",
                "text": "What are some trending topics in AI right now?"
              }
            ],
            "role": "user",
            "taskId": "fb816920-46e3-4bdf-8889-c7fdab377e7a"
          },
          {
            "contextId": "508efe3a-41ca-4e0b-9478-ea82dbce2ea6",
            "kind": "message",
            "messageId": "f2f05466-2943-4671-bd83-fe41eb2193d2",
            "parts": [
              {
                "kind": "text",
                "text": "Executing task..."
              }
            ],
            "role": "agent",
            "taskId": "fb816920-46e3-4bdf-8889-c7fdab377e7a"
          }
        ],
        "id": "fb816920-46e3-4bdf-8889-c7fdab377e7a",
        "kind": "task",
        "status": {
          "state": "completed",
          "timestamp": "2025-07-17T00:37:42.277421+00:00"
        }
      }
    }


If we want to extract just the text, which is what we will pass in as context to future interactions with our agent, we can do that as well:


```python
if 'result' in response_dict and 'artifacts' in response_dict['result']:
    artifacts = response_dict['result']['artifacts']
    for artifact in artifacts:
        if 'parts' in artifact:
                for part in artifact['parts']:
                    if 'text' in part:
                        print(part['text'])
```

    Several key themes and advancements are dominating the field of Artificial Intelligence (AI) in 2025, reflecting its rapid evolution and increasing integration across various sectors.
    
    Some of the most prominent trending topics in AI right now include:
    
    *   **Generative AI Evolution and Integration** Generative AI continues to be a major force, moving beyond basic content creation to more advanced and integrated applications. This includes the development of more sophisticated generative AI models, often referred to as Generative AI 2.0, that are becoming autonomous creators. There's a strong focus on integrating generative AI into everyday applications, making it an "obligatory copilot" for skilled workers across various industries. Multimodal AI, which can process and generate content from various inputs like text, images, and video, is also seeing significant advancements.
    *   **Agentic AI** A significant trend is the rise of "Agentic AI," where AI systems can act autonomously to achieve specific goals, make decisions, and take actions independently without constant human oversight. These AI agents are envisioned as the "apps of the AI era," handling tasks on users' behalf and transforming business processes. Examples include AIs that can surf the web, order groceries, predict sales, and manage inventory.
    *   **AI in the Workplace and Productivity** AI's adoption in the workplace is accelerating, with a focus on enhancing productivity, automating repetitive and mundane tasks, and transforming work dynamics. AI is increasingly used for tasks like sifting through emails, taking notes, generating content, and streamlining job processes. While concerns about job displacement exist, AI is also seen as augmenting human roles and creating new job opportunities.
    *   **Specialized AI Applications** AI is being deeply embedded into specific industries, driving innovation and efficiency:
        *   **Healthcare:** AI is accelerating scientific research and improving healthcare outcomes, with applications in diagnosing patients and identifying health problems.
        *   **Finance:** AI is revolutionizing the finance landscape through autonomous finance, including customer service chatbots, automated forecasting, and AI-powered fraud detection.
        *   **Supply Chain:** AI is enhancing efficiency in supply chain management through optimized inventory and logistics, risk prediction, and sustainability drives.
        *   **Education:** AI is being explored for personalized learning experiences, although its implementation can be controversial.
    *   **Ethical AI and Regulation** As AI becomes more powerful and pervasive, there is increased scrutiny on AI ethics, responsible AI, and the development of broader AI regulations. This includes concerns about privacy, data breaches, and ensuring fundamental rights are respected, as seen with initiatives like the European Union's AI Act.
    *   **Technical Advancements** Underlying these applications are continuous technical improvements. This includes more reasonable reasoning models, advancements in smaller language models (SLMs) that offer efficiency benefits, and the emerging field of "Embodied AI," which aims to bring multimodal AI abilities into the physical world through robotics. There's also a focus on improving the cost-efficiency of AI inference and exploring new model architectures beyond traditional transformers.
    


...and that's it for now, we have covered a lot of ground, yet only brushed the surface of A2A and the power it truly holds.

#### Summary: What have we learned?

* Created a basic agent using Google's ADK
* Covered the basics of A2A protocol
* Launched a server to host our remote agent
* RPC 2.0 Calls to interact with our agent

#### Whats Next?
In the coming weeks, we will be releasing a series of blog posts covering the nuances of A2A in depth:
- RPC Deep Dive
- The "Agent"
- `AgentCard`, `AgentSkills`, `AgentExecutor`
- Tools
- Event Handling (tasks, messages...)
- A2A with Model Context Protocol (MCP)
- Debugging A2A + Tips and Tricks
- Custom Agents
- Authentication and Authorization


We will also be building complex multi-agent workflows and learning how to orchestrate them with the power of [Littlehorse.io](https://littlehorse.io/)! Stay tuned!


