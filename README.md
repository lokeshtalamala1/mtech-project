AI Integration with Wazuh SIEMs

An intelligent cybersecurity project that combines Wazuh SIEM, machine learning anomaly detection, UEBA-style behavioral analysis, FastAPI-based MCP access, and LLM-powered explanations to reduce alert noise and help SOC analysts understand suspicious activity faster. The project evolved from an initial AI-enabled Wazuh integration layer with multiple query pipelines into a Phase 2 anomaly detection and explainability system built on real Wazuh network-flow logs.

Highlights
Uses Wazuh logs and OpenSearch as the security data source.
Filters and processes network flow logs for anomaly detection.
Extracts traffic and behavioral features such as bytes, packets, asymmetry, ports, and protocol encodings.
Applies Isolation Forest and LSTM Autoencoder for complementary anomaly detection.
Uses score fusion, persistence filtering, and percentile-based severity classification.
Exposes results through a FastAPI MCP-style layer and a CLI agent.
Generates human-readable explanations using OpenAI LLMs.
Project Motivation

Modern SOCs generate huge volumes of logs from endpoints, servers, networks, and cloud systems. Traditional SIEM workflows depend heavily on predefined rules and signatures, which makes them weak against unknown attacks, behavior-driven threats, and alert fatigue. This project addresses that gap by adding ML-based anomaly detection and LLM-based explanation over Wazuh security data.

What This Project Does

This system takes Wazuh network flow logs, cleans and transforms them into ML-ready features, detects anomalies using two complementary models, combines their outputs into a single anomaly score, filters unstable alerts, assigns severity levels, and then exposes the results through an API and a terminal agent. The final output is not only a detection score, but also an explanation that helps analysts understand what happened and why it matters.

Project Evolution
Phase 1

The first phase focused on AI-assisted integration with Wazuh SIEM using an MCP-style architecture, natural language querying, multiple query pipelines, log de-duplication, and AI-generated summaries.

Phase 2

The second phase extended the system into a full anomaly detection and explanation pipeline with:

Wazuh network-flow data processing,
feature engineering,
pseudo labeling,
Isolation Forest,
LSTM Autoencoder,
score fusion,
persistence filtering,
severity classification,
FastAPI MCP endpoints,
OpenAI explanations,
CLI-based intent routing.
System Architecture
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

This layered architecture is designed to keep detection, prioritization, and explanation modular so that each part can evolve independently.

Data Pipeline

The data comes from Wazuh SIEM logs stored in OpenSearch. In the latest setup, around 45,000 raw logs were collected, and after filtering, about 3,400 network-flow logs were retained for modeling. Only logs with data.flow were used, while Sysmon and other non-network events were removed to reduce noise.

Feature Engineering

Raw flow records are converted into structured numerical features that capture both traffic and behavior. The feature set includes:

total_bytes
total_packets
avg_packet_size
byte_asymmetry
packet_asymmetry
log_bytes
log_packets
port_diff

The system also encodes IP addresses and protocol values numerically so they can be used by machine learning models.

Pseudo Labeling

Because the dataset does not contain clean ground-truth labels, the project uses heuristic pseudo labeling to create a binary target:

1 = anomaly
0 = normal

A record is labeled anomalous if it shows high traffic volume, high packet count, strong asymmetry, or high Wazuh rule severity. This is used for evaluation and comparison, while the detection itself remains largely unsupervised.

Models Used
Isolation Forest

Used to detect statistical outliers in feature space. It is effective for rare, extreme, or unusual records.

LSTM Autoencoder

Used to learn sequential behavior and detect temporal deviations through reconstruction error. It is useful for behavioral anomalies and slow-evolving attacks.

Other Models Explored

The project also evaluated One-Class SVM, LOF, and GraphSAGE, but these were not selected as final models because they performed less effectively on this dataset.

Training Strategy

The models are trained offline on historical data to learn what normal behavior looks like. Isolation Forest learns statistical rarity, and the LSTM Autoencoder learns sequential behavior. This allows the system to detect deviations from the learned baseline without requiring large labeled attack datasets.

Inference Pipeline

During inference, each log is scored by both models. The scores are calibrated and combined using weighted fusion:

fusion_score = 0.65 * max(if_score, lstm_score) + 0.35 * min(if_score, lstm_score)

Then the persistence layer checks whether anomalies repeat within a short window before generating the final alert. This reduces false positives and makes the output more stable for SOC use.

Severity Classification

The system uses percentile-based thresholds over the fused anomaly scores. This makes severity adaptive to the dataset rather than relying on fixed hardcoded cutoffs. The final labels are:

WEAK
MODERATE
STRONG

For the latest calibration, the approximate thresholds are:

P85 ≈ 0.3649
P95 ≈ 0.5628

So:

below 0.3649 → WEAK
0.3649 to 0.5628 → MODERATE
above 0.5628 → STRONG

This helps prioritize anomalies in a SOC-friendly way.

Fusion and Persistence

Fusion combines the statistical and behavioral signals from IF and LSTM. Persistence filtering ensures that isolated spikes are not treated as alerts unless they repeat across multiple observations. This is one of the main mechanisms used to reduce noise and false positives.

FastAPI MCP Layer

The system exposes a FastAPI-based MCP-style interface with endpoints such as:

/ — health check
/anomalies — returns anomaly rows
/explain — generates LLM-based explanations

This layer serves as the bridge between model output and user-facing interaction. Swagger UI is used to inspect and test the endpoints.

CLI Agent

A terminal-based agent allows users to ask natural language questions such as:

how many anomalies?
how many normal records?
list anomalies
explain the second anomaly

The agent uses intent-based routing so that simple queries are answered directly, while explanation queries are routed through the LLM.

LLM Explanation Layer

The LLM turns structured anomaly information into a concise security explanation. The output typically includes:

attack type
reason
risk level
recommendation

This improves interpretability and makes the system more useful for analysts who need actionable insights rather than raw scores.

Experimental Results

The current system was evaluated on real Wazuh logs and produced the following representative results:

Raw logs processed: ~45,000
Filtered network-flow logs: ~3,400
Isolation Forest recall: 0.98
LSTM Autoencoder F1: ≈ 0.73

The experiments show that the hybrid system captures both statistical outliers and behavioral anomalies while reducing alert noise after fusion and persistence filtering.

Key Contributions
Hybrid SIEM anomaly detection pipeline for Wazuh logs.
Combination of statistical and temporal anomaly detection.
Score fusion and persistence filtering for noise reduction.
Severity-based prioritization for SOC workflows.
FastAPI-based access layer for structured and natural language use.
Future Work

The next possible extensions include:

SOAR integration for automated response
advanced ML-based UEBA models
semantic log de-duplication using similarity techniques
AI-based alert prioritization and remediation
integration with other MCP-enabled security tools

These are natural improvements that extend the current detection-and-explanation pipeline into a more autonomous SOC assistant.

Use Cases

This system is useful for:

banking and BFSI SOCs,
telecom and ISP traffic monitoring,
enterprise insider-threat detection,
government and smart infrastructure monitoring,
low-cost security monitoring in Indian organizations using open-source SIEM tooling.
Project Structure
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
How to Run
1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
2. Install dependencies
pip install -r requirements.txt
3. Run inference pipeline
python ml_anomaly/src/inference_pipeline.py
4. Start FastAPI MCP server
PYTHONPATH=. uvicorn ml_anomaly.mcp.server:app --reload
5. Open Swagger UI
http://127.0.0.1:8000/docs
6. Run CLI agent
python ml_anomaly/mcp_agent/main.py
API Endpoints
GET /

Health check endpoint.

GET /anomalies?limit=10

Returns anomaly rows from the inference output CSV.

GET /explain?limit=5

Generates LLM-based explanations for the selected anomalies.

Configuration

Create a .env file and store your OpenAI API key there:

OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
Limitations
AI explanation quality depends on the completeness of the underlying logs.
UEBA-style modeling becomes stronger with longer history.
LLM responses can add latency.
De-duplication and severity logic may need tuning for different environments.
Acknowledgements

This project was developed under the guidance of Prof. N. S. Narayanaswamy and Dr. Jaimandeep Singh, with support from the Department of Computer Science and Engineering, IIT Madras. The overall work builds on AI-assisted Wazuh integration ideas developed across the MTP phases.
