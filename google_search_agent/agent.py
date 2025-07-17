from google.adk.agents import Agent
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from google.adk.tools import google_search
from a2a.server.request_handlers import DefaultRequestHandler
from agent_executor import GoogleSearchAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
import uvicorn

if __name__ == "__main__":

    host = '127.0.0.1'
    port = 9999

    search_agent = Agent(
        model="gemini-2.5-flash",
        name="google_search_agent",
        description="Performs a Google Search, aggregates relevant results and links",
        instruction="""
            ...
        """,
        tools=[google_search]
    )

    google_search_skill = AgentSkill(
        id='google_search',
        name="Google Search",
        description="conduct a google search",
        tags=["search", "google"]
    )

    search_agent_card = AgentCard(
        name='google_search_agent',
        description=search_agent.description,
        url=f"http://{host}:{port}",
        version='0.1.0',
        capabilities=AgentCapabilities(
            streaming=True
        ),
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        skills=[google_search_skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=GoogleSearchAgentExecutor(search_agent),
        task_store=InMemoryTaskStore()
    )

    app = A2AStarletteApplication(
        agent_card=search_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)