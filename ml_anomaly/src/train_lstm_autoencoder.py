import pandas as pd
import numpy as np
import os
import joblib

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
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("Step 1: script started")

# =========================
# PATH SETUP (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "data/labeled_features.csv")
output_path = os.path.join(BASE_DIR, "outputs/lstm_autoencoder_results.csv")
model_path = os.path.join(BASE_DIR, "models/lstm_autoencoder.h5")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")

# =========================
# 1. Load data
# =========================
df = pd.read_csv(data_path)
print("Step 2: data loaded, shape =", df.shape)

features = [
    "src_port", "dest_port", "rule_level",
    "total_bytes", "total_packets",
    "avg_packet_size", "byte_asymmetry", "packet_asymmetry",
    "proto_num", "src_ip_num", "dest_ip_num",
    "port_diff", "log_bytes", "log_packets"
]

X = df[features].values
y = df["label"].values
print("Step 3: features selected")

# =========================
# 2. Scale features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Step 4: scaling done")

# =========================
# 3. Create sequences
# =========================
SEQ_LEN = 10

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(1 if np.any(y[i:i+seq_len] == 1) else 0)
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)
print("Step 5: sequences created, shape =", X_seq.shape)

# =========================
# 4. Train on normal sequences
# =========================
X_train = X_seq[y_seq == 0]
print("Step 6: normal-only sequence training shape =", X_train.shape)

timesteps = X_train.shape[1]
n_features = X_train.shape[2]

# =========================
# 5. Build model
# =========================
inputs = Input(shape=(timesteps, n_features))

encoded = LSTM(32, activation="relu", return_sequences=False)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(32, activation="relu", return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(n_features))(decoded)

model = Model(inputs, decoded)
model.compile(optimizer="adam", loss="mse")

print("Step 7: model built")

# =========================
# 6. Train
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, X_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stop],
    verbose=1
)

print("Step 8: model trained")

# =========================
# 7. Reconstruction error
# =========================
X_seq_pred = model.predict(X_seq, verbose=0)
seq_error = np.mean(np.square(X_seq - X_seq_pred), axis=(1, 2))

X_train_pred = model.predict(X_train, verbose=0)
train_error = np.mean(np.square(X_train - X_train_pred), axis=(1, 2))

threshold = np.percentile(train_error, 98)
print(f"Step 9: threshold selected = {threshold:.6f}")

y_pred = (seq_error > threshold).astype(int)

# =========================
# 8. Metrics
# =========================
precision = precision_score(y_seq, y_pred, zero_division=0)
recall = recall_score(y_seq, y_pred, zero_division=0)
f1 = f1_score(y_seq, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_seq, seq_error)
pr_auc = average_precision_score(y_seq, seq_error)

print("\n===== LSTM Autoencoder Results =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_seq, y_pred))

print("\nClassification Report:")
print(classification_report(y_seq, y_pred, digits=4))

# =========================
# 9. Save results
# =========================
os.makedirs(os.path.dirname(output_path), exist_ok=True)

results_df = pd.DataFrame({
    "seq_label": y_seq,
    "lstm_ae_pred": y_pred,
    "lstm_ae_score": seq_error
})

results_df.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")

# =========================
# 10. Save model + scaler
# =========================
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

model.save(model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to {model_path}")
print(f"Scaler saved to {scaler_path}")