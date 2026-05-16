from openai import OpenAI
from .config import OPENAI_API_KEY, MODEL_NAME
from .formatter import format_for_llm

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Check .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)


def explain_anomaly(row):

    prompt = format_for_llm(row)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a cybersecurity analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content