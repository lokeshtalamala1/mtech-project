from fastapi import FastAPI
import pandas as pd

from .config import DATA_PATH
from .llm_explainer import explain_anomaly

app = FastAPI(title="ML + MCP + LLM Server")


# HEALTH CHECK

@app.get("/")
def health():
    return {"status": "MCP server running"}


# GET RAW ANOMALIES

@app.get("/anomalies")
def get_anomalies(limit: int = 10):

    df = pd.read_csv(DATA_PATH)

    anomalies = df[df["anomaly"] == True].head(limit)

    return anomalies.to_dict(orient="records")

# GET EXPLANATIONS
@app.get("/explain")
def explain(limit: int = 5):

    df = pd.read_csv(DATA_PATH)
    anomalies = df[df["anomaly"] == True].head(limit)

    results = []

    for _, row in anomalies.iterrows():
        try:
            explanation = explain_anomaly(row)

            results.append({
                "severity": row["severity"],
                "fusion_score": row["fusion_score"],
                "explanation": explanation
            })

        except Exception as e:
            results.append({
                "severity": row["severity"],
                "fusion_score": row["fusion_score"],
                "error": str(e)
            })

    return results