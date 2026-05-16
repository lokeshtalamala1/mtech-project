import json
import pandas as pd
import os

input_path = "data/logs.jsonl"
output_path = "data/processed_logs.csv"

rows = []

flow_count = 0
non_flow_count = 0

with open(input_path) as f:
    for line in f:
        try:
            source = json.loads(line)

            # Basic safety checks
            if "agent" not in source or "data" not in source:
                continue

            # Filter only flow logs
            if "flow" not in source["data"]:
                non_flow_count += 1
                continue

            # Skip incomplete logs
            if source["data"].get("src_ip") is None:
                continue

            if source["data"]["flow"].get("bytes_toserver") is None:
                continue

            flow_count += 1

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

df = pd.DataFrame(rows)

print("Flow logs:", flow_count)
print("Non-flow logs:", non_flow_count)
print(f"Processed {len(df)} valid network logs")

# Ensure output folder exists
os.makedirs("data", exist_ok=True)

df.to_csv(output_path, index=False)