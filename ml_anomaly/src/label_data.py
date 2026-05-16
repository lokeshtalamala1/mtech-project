import pandas as pd
import numpy as np

# Load engineered features
df = pd.read_csv("/home/user/Downloads/MTP/mtp/ml_anomaly/data/features.csv")

# -----------------------------
# 1. Compute thresholds
# -----------------------------
bytes_thr = df["total_bytes"].quantile(0.99)
packets_thr = df["total_packets"].quantile(0.99)
byte_asym_thr = df["byte_asymmetry"].abs().quantile(0.99)
packet_asym_thr = df["packet_asymmetry"].abs().quantile(0.99)

print("Thresholds used for labeling:")
print(f"total_bytes >= {bytes_thr}")
print(f"total_packets >= {packets_thr}")
print(f"|byte_asymmetry| >= {byte_asym_thr}")
print(f"|packet_asymmetry| >= {packet_asym_thr}")

# -----------------------------
# 2. Strong anomaly rules
# -----------------------------
anomaly_mask = (
    (df["rule_level"] >= 10) |
    (df["total_bytes"] >= bytes_thr) |
    (df["total_packets"] >= packets_thr) |
    (df["byte_asymmetry"].abs() >= byte_asym_thr) |
    (df["packet_asymmetry"].abs() >= packet_asym_thr)
)

# -----------------------------
# 3. Strong normal rules
# -----------------------------
normal_mask = (
    (df["rule_level"] <= 5) &
    (df["total_bytes"] < bytes_thr) &
    (df["total_packets"] < packets_thr) &
    (df["byte_asymmetry"].abs() < byte_asym_thr) &
    (df["packet_asymmetry"].abs() < packet_asym_thr)
)

# -----------------------------
# 4. Assign labels
# -----------------------------
df["label"] = np.nan
df.loc[anomaly_mask, "label"] = 1
df.loc[normal_mask, "label"] = 0

# Keep only clearly labeled rows
df_labeled = df.dropna(subset=["label"]).copy()
df_labeled["label"] = df_labeled["label"].astype(int)

# -----------------------------
# 5. Save output
# -----------------------------
df_labeled.to_csv("/home/user/Downloads/MTP/mtp/ml_anomaly/data/labeled_features.csv", index=False)

print("\nLabel distribution:")
print(df_labeled["label"].value_counts())

print(f"\nSaved labeled dataset to data/labeled_features.csv")
print(f"Total labeled rows: {len(df_labeled)}")

print("\nSample anomalies:")
print(df_labeled[df_labeled["label"] == 1][[
    "rule_level", "total_bytes", "total_packets",
    "byte_asymmetry", "packet_asymmetry"
]].head(10))

print("\nSample normals:")
print(df_labeled[df_labeled["label"] == 0][[
    "rule_level", "total_bytes", "total_packets",
    "byte_asymmetry", "packet_asymmetry"
]].head(10))