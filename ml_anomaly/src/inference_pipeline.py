import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "data/features.csv")
model_dir = os.path.join(BASE_DIR, "models")
output_path = os.path.join(BASE_DIR, "outputs/inference_results.csv")

# -------------------------------
# LOAD MODELS
# -------------------------------
if_model = joblib.load(os.path.join(model_dir, "isolation_forest.pkl"))
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

lstm_model = load_model(
    os.path.join(model_dir, "lstm_autoencoder.h5"),
    compile=False
)

# -------------------------------
# FEATURES
# -------------------------------
features = [
    "src_port", "dest_port", "rule_level",
    "total_bytes", "total_packets",
    "avg_packet_size", "byte_asymmetry", "packet_asymmetry",
    "proto_num", "src_ip_num", "dest_ip_num",
    "port_diff", "log_bytes", "log_packets"
]

SEQ_LEN = 10

# -------------------------------
# IF SCORE
# -------------------------------
def get_if_score(x):
    score = if_model.decision_function(x)[0]
    return -score +0.3 # higher = anomaly

# -------------------------------
# LSTM SCORE (SEQUENCE BASED)
# -------------------------------
def get_lstm_score(seq):
    seq_scaled = scaler.transform(seq)
    seq_scaled = seq_scaled.reshape(1, SEQ_LEN, -1)

    recon = lstm_model.predict(seq_scaled, verbose=0)
    error = np.mean((seq_scaled - recon) ** 2)

    return error

# -------------------------------
# PERSISTENCE
# -------------------------------
def apply_persistence(fusion_score, history, threshold=0.6, window=5, min_hits=2):
    history.append(fusion_score)

    if len(history) > window:
        history.pop(0)

    count = sum(1 for x in history if x > threshold)

    return count >= min_hits

# -------------------------------
# MAIN INFERENCE (2-PASS)
# -------------------------------
def run_inference(df):

    X = df[features].values

    results = []
    history = []

    lstm_scores = []
    if_scores = []
    fusion_scores = []

    # =========================
    # PASS 1 → compute scores
    # =========================
    for i in range(len(X)):

        if i % 500 == 0:
            print(f"Processed {i}/{len(X)} rows")

        # IF
        x_row = pd.DataFrame([X[i]], columns=features)
        if_score = get_if_score(x_row)

        # LSTM
        if i < SEQ_LEN:
            lstm_score = 0
        else:
            seq = X[i-SEQ_LEN:i]
            lstm_score = get_lstm_score(seq)

        fusion_score = 0.65 * max(lstm_score, if_score) + 0.35 * min(lstm_score, if_score)

        lstm_scores.append(lstm_score)
        if_scores.append(if_score)
        fusion_scores.append(fusion_score)

    # =========================
    # CALIBRATE THRESHOLDS
    # =========================
    fusion_mid  = np.percentile(fusion_scores, 85)
    fusion_high = np.percentile(fusion_scores, 95)

    print("\nFusion Thresholds:")
    print("Mid (85):", fusion_mid)
    print("High (95):", fusion_high)

    # =========================
    # PASS 2 → assign labels
    # =========================
    for i in range(len(X)):

        lstm_score = lstm_scores[i]
        if_score = if_scores[i]
        fusion_score = fusion_scores[i]

        # Severity
        if fusion_score >= fusion_high:
            severity = "STRONG"
        elif fusion_score >= fusion_mid:
            severity = "MODERATE"
        else:
            severity = "WEAK"

        # Persistence
        final_anomaly = apply_persistence(fusion_score, history)

        results.append({
            "if_score": if_score,
            "lstm_score": lstm_score,
            "fusion_score": fusion_score,
            "severity": severity,
            "anomaly": final_anomaly
        })

    return pd.DataFrame(results), lstm_scores, if_scores


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":

    df = pd.read_csv(data_path)

    output, lstm_scores, if_scores = run_inference(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)

    print("\nInference completed. Results saved.")

    print("\nScore Ranges:")
    print("LSTM range:", np.min(lstm_scores), np.max(lstm_scores))
    print("IF range:", np.min(if_scores), np.max(if_scores))