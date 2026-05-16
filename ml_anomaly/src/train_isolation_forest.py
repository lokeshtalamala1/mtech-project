import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# Load Data
df = pd.read_csv("/home/user/Downloads/MTP/mtp/ml_anomaly/data/labeled_features.csv")

# Feature Selection
features = [
    "src_port",
    "dest_port",
    "rule_level",
    "total_bytes",
    "total_packets",
    "avg_packet_size",
    "byte_asymmetry",
    "packet_asymmetry",
    
    "proto_num",
    "src_ip_num",
    "dest_ip_num",
    "port_diff",
    "log_bytes",
    "log_packets"
]

X = df[features]

# Train Model
model = IsolationForest(
    contamination=0.02,
    random_state=42
)

model.fit(X)

df["anomaly"] = model.predict(X)
# -------------------------------
# METRICS (only if labels exist)
# -------------------------------
if "label" in df.columns:
    print("\n===== Isolation Forest Evaluation =====")

    y_true = df["label"]
    y_pred = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)
    anomaly_score = -model.decision_function(X)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, anomaly_score)
    pr_auc = average_precision_score(y_true, anomaly_score)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


# Save Results
df.to_csv("/home/user/Downloads/MTP/mtp/ml_anomaly/outputs/isolation_forest_results.csv", index=False)

print("\n Anomaly detection complete\n")

# Debug
print("🔹 Anomaly Distribution:")
print(df["anomaly"].value_counts())

print("\n🔹 Unique Traffic Patterns (sample):")
print(df[["total_bytes", "total_packets"]].value_counts().head(10))

# Visualization
plt.figure(figsize=(8, 6))
plt.style.use("default")  # white background

# Custom colors
palette = {
    1: "green",   # normal
    -1: "red"     # anomaly
}

# Sort so anomalies appear on top
df_sorted = df.sort_values(by="anomaly")

sns.scatterplot(
    x=df_sorted["log_bytes"],
    y=df_sorted["log_packets"],
    hue=df_sorted["anomaly"],
    palette=palette,
    alpha=0.7,
    s=30  # point size
)

# Labels
plt.title("Anomaly Detection (Real Data)", color="black")
plt.xlabel("log_bytes", color="black")
plt.ylabel("log_packets", color="black")

plt.xticks(color="black")
plt.yticks(color="black")

plt.legend(title="Anomaly (1=Normal, -1=Anomaly)")

# Save plot
plt.savefig("/home/user/Downloads/MTP/mtp/ml_anomaly/outputs/isolation_forest_plot.png")

print("\n📊 Plot saved to outputs/anomaly_plot.png\n")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models/isolation_forest.pkl")

os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(model, model_path)

print(f"Model saved to {model_path}")

print("Model saved to models/isolation_forest.pkl")