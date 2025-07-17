import uuid
from a2a.types import SendMessageRequest, MessageSendParams, AgentCard
from a2a.client import A2AClient
import httpx
import asyncio
import json


timeout_config = httpx.Timeout(
            timeout=120.0,
            connect=10.0,
            read=120.0,
            write=10.0,
            pool=5.0
        )

async def request():

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

        response_dict = response.model_dump(mode='json', exclude_none=True)
        if 'result' in response_dict and 'artifacts' in response_dict['result']:
            artifacts = response_dict['result']['artifacts']
            for artifact in artifacts:
                if 'parts' in artifact:
                        for part in artifact['parts']:
                            if 'text' in part:
                                print(part['text'])


if __name__ == "__main__":
     asyncio.run(request())
