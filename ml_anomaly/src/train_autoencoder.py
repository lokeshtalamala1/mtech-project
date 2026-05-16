import pandas as pd
import numpy as np
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("Step 1: script started")

# =========================
# 1. Load labeled data
# =========================
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

X = df[features].values
y = df["label"].values   # 0 = normal, 1 = anomaly
print("Step 3: features selected")

# =========================
# 2. Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Step 4: scaling done")

# =========================
# 3. Train only on normal data
# =========================
X_train = X_scaled[y == 0]
print("Step 5: normal-only training shape =", X_train.shape)

input_dim = X_train.shape[1]
encoding_dim = 8

# =========================
# 4. Build autoencoder
# =========================
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(encoding_dim, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")

print("Step 6: model built")

# =========================
# 5. Train model
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stop],
    verbose=1
)

print("Step 7: model trained")

# =========================
# 6. Reconstruction error
# =========================
X_pred = autoencoder.predict(X_scaled, verbose=0)
recon_error = np.mean(np.square(X_scaled - X_pred), axis=1)

# threshold from normal training distribution
X_train_pred = autoencoder.predict(X_train, verbose=0)
train_error = np.mean(np.square(X_train - X_train_pred), axis=1)
threshold = np.percentile(train_error, 98)

print(f"Step 8: threshold selected = {threshold:.6f}")

# 0 = normal, 1 = anomaly
y_pred = (recon_error > threshold).astype(int)

# =========================
# 7. Metrics
# =========================
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, recon_error)
pr_auc = average_precision_score(y, recon_error)
cm = confusion_matrix(y, y_pred)

print("\n===== Autoencoder Results =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))

# =========================
# 8. Save results
# =========================
df["ae_pred"] = y_pred
df["ae_score"] = recon_error
df.to_csv("outputs/autoencoder_results.csv", index=False)

print("\nSaved results to outputs/autoencoder_results.csv")