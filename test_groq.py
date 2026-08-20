import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'} (starts with {api_key[:4] if api_key else 'None'})")

try:
    from groq import Groq
    client = Groq(api_key=api_key)
    models = client.models.list()
    print("Available Models:")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print("API Test Failed:", str(e))
