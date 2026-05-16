from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME
from tools import get_ml_context, get_ip_info
import pandas as pd

client = OpenAI(api_key=OPENAI_API_KEY)

def detect_intent(query):
    query = query.lower()

    # BOTH (IMPORTANT - FIRST CHECK)
    if "both" in query or ("normal" in query and "anomal" in query):
        return "both"

    # COUNT
    elif "how many" in query or "count" in query:
        if "normal" in query:
            return "count_normal"
        else:
            return "count_anomaly"

    # SPECIFIC
    elif "second" in query:
        return "second"

    # LIST
    elif "list" in query:
        return "list"

    else:
        return "explain"

def run_agent(user_query):

    # 🔥 Step 1: detect intent
    intent = detect_intent(user_query)

    # 🔥 Step 2: load data
    df = pd.read_csv("/home/user/Downloads/MTP/mtp/ml_anomaly/outputs/inference_results.csv")
    anomalies = df[df["anomaly"] == True]

    # 🔥 Step 3: handle intent

    if intent == "count_anomaly":
        return f"Total anomalies detected: {len(anomalies)}"

    elif intent == "count_normal":
        normal = df[df["anomaly"] == False]
        return f"Total normal (non-anomalous) records: {len(normal)}"

    elif intent == "list":
        return anomalies.head(5).to_dict(orient="records")

    elif intent == "specific":
        row = anomalies.iloc[1]
    
    elif intent == "second":
        if len(anomalies) > 1:
            row = anomalies.iloc[1]
        else:
            return "Not enough anomalies found."
    elif intent == "both":
        total_anomaly = len(anomalies)
        total_normal = len(df[df["anomaly"] == False])

        return {
            "total_anomalies": total_anomaly,
            "total_normal": total_normal,
            "total_records": len(df)
        }
    
    else:  # explain
        row = anomalies.iloc[0]

    # 🔥 Step 4: tool usage
    ml_data = row.to_dict()
    ip_data = get_ip_info()

    # 🔥 Step 5: LLM reasoning
    prompt = f"""
User Query: {user_query}

ML Data: {ml_data}
IP Data: {ip_data}

Explain clearly based on user query.
Return JSON.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
