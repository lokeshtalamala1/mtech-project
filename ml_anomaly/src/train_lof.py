import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

print("Step 1: script started")

df = pd.read_csv("data/labeled_features.csv")
print("Step 2: data loaded, shape =", df.shape)

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
y = df["label"]
print("Step 3: features selected")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Step 4: scaling done")

# LOF is usually fit on the full data in unsupervised mode
model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.02,
    novelty=False
)

print("Step 5: fitting + predicting")
pred = model.fit_predict(X_scaled)

# LOF output:
#  1  = inlier
# -1  = outlier
y_pred = [1 if p == -1 else 0 for p in pred]

# negative_outlier_factor_:
# more negative = more anomalous
anomaly_score = -model.negative_outlier_factor_

precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, anomaly_score)
pr_auc = average_precision_score(y, anomaly_score)
cm = confusion_matrix(y, y_pred)

print("\n===== LOF Results =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))

df["lof_pred"] = y_pred
df["lof_score"] = anomaly_score
df.to_csv("outputs/lof_results.csv", index=False)

print("\nSaved results to outputs/lof_results.csv")