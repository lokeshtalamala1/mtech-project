import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data/features.csv")

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

print("Vectorized shape:", X_scaled.shape)