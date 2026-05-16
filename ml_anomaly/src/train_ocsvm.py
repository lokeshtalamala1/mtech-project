import pandas as pd
from sklearn.svm import OneClassSVM
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

X_train = X_scaled[y == 0]
print("Step 5: normal-only training shape =", X_train.shape)

model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.02
)

model.fit(X_train)
print("Step 6: model fitted")

pred = model.predict(X_scaled)
print("Step 7: prediction done")

# OneClassSVM output: 1=inlier, -1=outlier
y_pred = [1 if p == -1 else 0 for p in pred]

# higher score should mean more anomalous
anomaly_score = -model.decision_function(X_scaled)

precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, anomaly_score)
pr_auc = average_precision_score(y, anomaly_score)
cm = confusion_matrix(y, y_pred)

print("\n===== One-Class SVM Results =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))

df["ocsvm_pred"] = y_pred
df["ocsvm_score"] = anomaly_score
df.to_csv("outputs/ocsvm_results.csv", index=False)

print("\nSaved results to outputs/ocsvm_results.csv")