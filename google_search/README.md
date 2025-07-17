### Getting Started:

Set up your environment:
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Create a `.env` file and add this: GOOGLE_API_KEY=<your-api-key>. If you don't have a gemini api key, you can get one [here](https://aistudio.google.com/apikey)

Run your agent: `python agent.py`
In a new terminal, run the test client: `python test_client.py`