# AI Integration with Wazuh SIEMs

An intelligent cybersecurity project that combines **Wazuh SIEM**, **machine learning anomaly detection**, **UEBA-style behavioral analysis**, **FastAPI-based MCP access**, and **LLM-powered explanations** to reduce alert noise and help SOC analysts understand suspicious activity faster.

The project evolved from an initial AI-enabled Wazuh integration layer into a Phase 2 anomaly detection and explainability system built on real Wazuh network-flow logs.

---

## Highlights

- Uses **Wazuh logs** and **OpenSearch** as the security data source.
- Filters and processes **network flow logs** for anomaly detection.
- Extracts traffic and behavioral features such as bytes, packets, asymmetry, ports, and protocol encodings.
- Applies **Isolation Forest** and **LSTM Autoencoder** for complementary anomaly detection.
- Uses **score fusion**, **persistence filtering**, and **percentile-based severity classification**.
- Exposes results through a **FastAPI MCP-style layer** and a **CLI agent**.
- Generates **human-readable explanations** using OpenAI LLMs.

---

## Project Motivation

Modern SOCs generate huge volumes of logs from endpoints, servers, networks, and cloud systems. Traditional SIEM workflows depend heavily on predefined rules and signatures, which makes them weak against unknown attacks, behavior-driven threats, and alert fatigue.

This project addresses that gap by adding:

- ML-based anomaly detection
- Behavioral analysis
- Explainable AI
- LLM-powered reasoning
- MCP-style interaction

over Wazuh security data.

---

## System Architecture

```text
Wazuh Network Flow Logs
        ↓
Data Ingestion & Filtering
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Isolation Forest + LSTM Autoencoder
        ↓
Score Calibration & Fusion
        ↓
Persistence Filtering
        ↓
Severity Classification
        ↓
FastAPI / CLI Output
        ↓
LLM Explanation
```

---

## Data Pipeline

The data comes from Wazuh SIEM logs stored in OpenSearch.

### Dataset Statistics
- Raw logs collected: ~45,000
- Filtered usable network-flow logs: ~3,400

### Filtering Strategy
Only logs containing:

```text
data.flow
```

were retained.

Removed:

- Sysmon logs
- Non-network logs
- Irrelevant event records

This ensured the dataset focused only on meaningful network behavior.

---

## Feature Engineering

Raw flow records are converted into structured numerical features.

### Features Used

| Feature | Description |
|---|---|
| `total_bytes` | Total traffic volume |
| `total_packets` | Total packet count |
| `avg_packet_size` | Bytes per packet |
| `byte_asymmetry` | Incoming vs outgoing byte imbalance |
| `packet_asymmetry` | Incoming vs outgoing packet imbalance |
| `log_bytes` | Log-scaled bytes |
| `log_packets` | Log-scaled packets |
| `port_diff` | Source/destination port relationship |

### Additional Encodings
- IP address encoding
- Protocol encoding

These features help models understand:

- traffic behavior
- communication patterns
- abnormal network activity

---

## Pseudo Labeling

Because real attack labels were unavailable, heuristic pseudo labels were created.

### Binary Labels

| Label | Meaning |
|---|---|
| `1` | Anomaly |
| `0` | Normal |

### Conditions Used
A record is labeled anomalous if it shows:

- high traffic volume
- high packet count
- strong asymmetry
- high Wazuh rule severity

---

## Models Used

### Isolation Forest
Used for:

- statistical anomaly detection
- outlier detection
- abnormal traffic identification

**Observation:** very effective for sudden traffic spikes and rare statistical deviations.

### LSTM Autoencoder
Used for:

- behavioral anomaly detection
- sequential pattern learning
- temporal deviation analysis

**Observation:** effective for slow attacks, evolving behavior, and communication pattern anomalies.

### Other Models Explored

| Model | Observation |
|---|---|
| One-Class SVM | Moderate performance |
| LOF | Unstable / underperformed |
| GraphSAGE | Underperformed due to limited graph structure |

---

## Model Performance

| Model | Performance |
|---|---|
| Isolation Forest | Recall ≈ 1.0 |
| LSTM Autoencoder | F1 ≈ 0.73 |
| One-Class SVM | Good ROC-AUC |

### Key Insight
Isolation Forest and LSTM complement each other:

- IF captures statistical anomalies
- LSTM captures behavioral anomalies

---

## Fusion Strategy

Both model scores are combined using weighted fusion.

### Formula

```math
Fusion Score = 0.65 × max(IF, LSTM) + 0.35 × min(IF, LSTM)
```

### Why Fusion?
This preserves:

- the stronger anomaly signal
- while still considering both models

---

## Persistence Filtering

A sliding window mechanism is used to reduce noise.

### Configuration
- Window size = 5
- Minimum hits = 2

### Purpose
Ensures that:

- temporary spikes are ignored
- only persistent anomalies are flagged

This significantly reduces false positives.

---

## Severity Classification

Severity is determined using percentile thresholds over fusion scores.

### Thresholds

| Percentile | Value |
|---|---|
| P85 | ≈ 0.3649 |
| P95 | ≈ 0.5628 |

### Severity Levels

| Fusion Score Range | Severity |
|---|---|
| `< 0.3649` | WEAK |
| `0.3649 – 0.5628` | MODERATE |
| `> 0.5628` | STRONG |

### Key Insight
Severity is:

- relative
- adaptive
- data-driven

---

## FastAPI MCP Layer

The project exposes a FastAPI-based MCP-style interface.

### API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Health check |
| `/anomalies` | Returns anomaly records |
| `/explain` | Generates LLM explanations |

---

## Swagger UI

FastAPI automatically generates Swagger UI.

### Access
```text
http://127.0.0.1:8000/docs
```

### Features
- API testing
- Response visualization
- Schema inspection
- Debugging support

---

## CLI MCP Agent

A terminal-based intelligent agent was developed.

### Example Queries
- how many anomalies?
- how many normal records?
- list anomalies
- explain second anomaly

---

## Intent-Based Agent

The agent detects user intent before deciding actions.

### Implemented Intents

| Intent | Action |
|---|---|
| `count_anomaly` | Count anomalies |
| `count_normal` | Count normal records |
| `both` | Return both counts |
| `list` | List anomaly records |
| `specific` | Retrieve specific anomaly |
| `explain` | Generate explanation |

### Optimization
Simple queries are handled directly. Complex reasoning is sent to the LLM.

This reduces:

- latency
- unnecessary API usage

---

## LLM Explanation Layer

The OpenAI LLM converts anomaly outputs into analyst-friendly explanations.

### Example Output

```json
{
  "attack_type": "Potential Data Exfiltration",
  "risk_level": "High",
  "reason": "Persistent abnormal outbound traffic detected.",
  "recommendation": "Investigate outbound connections immediately."
}
```

---

## Experimental Results

| Metric | Value |
|---|---|
| Raw logs processed | ~45,000 |
| Filtered flow logs | ~3,400 |
| Isolation Forest Recall | ≈ 1.0 |
| LSTM Autoencoder F1 | ≈ 0.73 |

---

## Key Contributions

- Hybrid SIEM anomaly detection pipeline for Wazuh logs.
- Combination of statistical and behavioral anomaly detection.
- Score fusion and persistence filtering.
- Severity-based SOC prioritization.
- FastAPI-based MCP access layer.
- LLM-powered explainable cybersecurity assistant.

---

## Example Use Cases

### Banking / BFSI
Detect:

- data exfiltration
- suspicious outbound traffic
- abnormal transaction behavior

### Insider Threat Detection
Detect:

- abnormal employee behavior
- lateral movement
- compromised credentials

### Telecom / ISP
Detect:

- DDoS behavior
- abnormal traffic bursts
- unusual packet patterns

---

## Project Structure

```text
mtp/
├── ml_anomaly/
│   ├── data/
│   ├── models/
│   ├── outputs/
│   ├── src/
│   ├── mcp/
│   └── mcp_agent/
├── models/
├── requirements.txt
└── venv/
```

---

## How to Run

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Inference Pipeline
```bash
python ml_anomaly/src/inference_pipeline.py
```

### 4. Start FastAPI Server
```bash
PYTHONPATH=. uvicorn ml_anomaly.mcp.server:app --reload
```

### 5. Open Swagger UI
```text
http://127.0.0.1:8000/docs
```

### 6. Run CLI MCP Agent
```bash
python ml_anomaly/mcp_agent/main.py
```

---

## API Examples

### Get Anomalies
```bash
curl -X GET "http://127.0.0.1:8000/anomalies?limit=10"
```

### Get Explanations
```bash
curl -X GET "http://127.0.0.1:8000/explain?limit=5"
```

---

## Environment Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

---

## Current Limitations

- Depends on quality of Wazuh logs.
- LLM explanations depend on available context.
- No real IP intelligence integration yet.
- Stateless agent (no memory/context awareness).
- LLM calls introduce latency.

---

## Future Work

- LLM-based intent detection
- Real threat intelligence APIs
- Context-aware memory agents
- SOAR integration
- Full FastMCP integration
- Autonomous tool-calling agents

---

## Technologies Used

- Wazuh
- OpenSearch
- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- FastAPI
- OpenAI API
- Uvicorn

---

## Acknowledgements

Developed as part of MTP work under the guidance of:

- Prof. N. S. Narayanaswamy
- Dr. Jaimandeep Singh

Department of Computer Science and Engineering  
Indian Institute of Technology Madras
