import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "inference_results.csv"
)
print("API KEY LOADED:", OPENAI_API_KEY[:5] if OPENAI_API_KEY else "NOT FOUND")