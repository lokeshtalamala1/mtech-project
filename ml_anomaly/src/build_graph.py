import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

print("Step 1: script started")

# =========================
# 1. Load data
# =========================
df = pd.read_csv("data/labeled_features.csv")
print("Step 2: data loaded, shape =", df.shape)

# keep rows with both IPs present
df = df.dropna(subset=["src_ip", "dest_ip"]).copy()
print("Step 3: after dropping missing IP rows =", df.shape)

# =========================
# 2. Build node list
# =========================
all_nodes = pd.Index(df["src_ip"]).append(pd.Index(df["dest_ip"])).unique()
node_to_id = {node: i for i, node in enumerate(all_nodes)}
print("Step 4: number of nodes =", len(all_nodes))

# =========================
# 3. Build edge list
# =========================
src_ids = df["src_ip"].map(node_to_id).values
dst_ids = df["dest_ip"].map(node_to_id).values

edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
print("Step 5: edge_index shape =", edge_index.shape)

# =========================
# 4. Build node-level aggregated features
# =========================
# aggregate outgoing behavior for each src_ip
agg = df.groupby("src_ip").agg({
    "total_bytes": ["mean", "sum", "max"],
    "total_packets": ["mean", "sum", "max"],
    "rule_level": ["mean", "max"],
    "avg_packet_size": ["mean"],
    "byte_asymmetry": ["mean"],
    "packet_asymmetry": ["mean"],
    "src_port": ["nunique"],
    "dest_port": ["nunique"],
    "proto_num": ["mean"]
})

agg.columns = ["_".join(col) for col in agg.columns]
agg = agg.reset_index().rename(columns={"src_ip": "node"})

# ensure all nodes exist in feature table
node_df = pd.DataFrame({"node": all_nodes})
node_features_df = node_df.merge(agg, how="left", on="node").fillna(0)

print("Step 6: node feature table shape =", node_features_df.shape)

# =========================
# 5. Scale node features
# =========================
feature_cols = [c for c in node_features_df.columns if c != "node"]
scaler = StandardScaler()
x = scaler.fit_transform(node_features_df[feature_cols])

x = torch.tensor(x, dtype=torch.float)
print("Step 7: node feature tensor shape =", x.shape)

# =========================
# 6. Create node labels
# =========================
# label node anomalous if it appears in any anomalous row
anom_src = set(df.loc[df["label"] == 1, "src_ip"].dropna().tolist())
anom_dst = set(df.loc[df["label"] == 1, "dest_ip"].dropna().tolist())
anom_nodes = anom_src.union(anom_dst)

y = torch.tensor(
    [1 if node in anom_nodes else 0 for node in all_nodes],
    dtype=torch.long
)

print("Step 8: node labels created")
print("Normal nodes   =", int((y == 0).sum()))
print("Anomaly nodes  =", int((y == 1).sum()))

# =========================
# 7. Save graph artifacts
# =========================
torch.save(edge_index, "data/graph_edge_index.pt")
torch.save(x, "data/graph_node_features.pt")
torch.save(y, "data/graph_labels.pt")

node_features_df.to_csv("data/graph_nodes.csv", index=False)
pd.DataFrame({
    "src_id": src_ids,
    "dst_id": dst_ids
}).to_csv("data/graph_edges.csv", index=False)

print("\nSaved:")
print("- data/graph_edge_index.pt")
print("- data/graph_node_features.pt")
print("- data/graph_labels.pt")
print("- data/graph_nodes.csv")
print("- data/graph_edges.csv")