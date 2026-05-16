import json
import pandas as pd
import os

input_path = "data/logs.jsonl"
output_path = "data/processed_logs.csv"

rows = []

with open(input_path) as f:
    for line in f:
        try:
            source = json.loads(line)

            if "flow" not in source.get("data", {}):
                continue

            row = {
                "agent_ip": source["agent"]["ip"],
                "src_ip": source["data"].get("src_ip"),
                "dest_ip": source["data"].get("dest_ip"),
                "src_port": source["data"].get("src_port"),
                "dest_port": source["data"].get("dest_port"),
                "proto": source["data"].get("proto"),
                "bytes_to_server": source["data"]["flow"].get("bytes_toserver"),
                "bytes_to_client": source["data"]["flow"].get("bytes_toclient"),
                "pkts_to_server": source["data"]["flow"].get("pkts_toserver"),
                "pkts_to_client": source["data"]["flow"].get("pkts_toclient"),
                "rule_level": source["rule"]["level"],
            }

            rows.append(row)

        except Exception:
            continue

import pandas as pd
import numpy as np

df = pd.read_csv("data/processed_logs.csv")

# Convert numeric columns first
df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce")
df["dest_port"] = pd.to_numeric(df["dest_port"], errors="coerce")

df["bytes_to_server"] = pd.to_numeric(df["bytes_to_server"], errors="coerce")
df["bytes_to_client"] = pd.to_numeric(df["bytes_to_client"], errors="coerce")
df["pkts_to_server"] = pd.to_numeric(df["pkts_to_server"], errors="coerce")
df["pkts_to_client"] = pd.to_numeric(df["pkts_to_client"], errors="coerce")

# Handle missing
df = df.fillna(0)

# Basic Features
df["total_bytes"] = df["bytes_to_server"] + df["bytes_to_client"]
df["total_packets"] = df["pkts_to_server"] + df["pkts_to_client"]

df["avg_packet_size"] = df["total_bytes"] / (df["total_packets"] + 1)

df["byte_asymmetry"] = (
    df["bytes_to_server"] - df["bytes_to_client"]
) / (df["total_bytes"] + 1)

df["packet_asymmetry"] = (
    df["pkts_to_server"] - df["pkts_to_client"]
) / (df["total_packets"] + 1)

# Advanced Features
df["proto_num"] = df["proto"].map({"TCP": 1, "UDP": 2}).fillna(0)

df["src_ip_num"] = df["src_ip"].astype("category").cat.codes
df["dest_ip_num"] = df["dest_ip"].astype("category").cat.codes

df["port_diff"] = df["dest_port"] - df["src_port"]

# Log Features
df["log_bytes"] = np.log1p(df["total_bytes"])
df["log_packets"] = np.log1p(df["total_packets"])

# Debug
print("Columns:", df.columns)
print(df.head())

# Save
df.to_csv("data/features.csv", index=False)