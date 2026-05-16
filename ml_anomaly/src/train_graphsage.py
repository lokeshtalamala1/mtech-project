import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import numpy as np

print("Step 1: script started")

# =========================
# 1. Load graph artifacts
# =========================
edge_index = torch.load("data/graph_edge_index.pt")
x = torch.load("data/graph_node_features.pt")
y = torch.load("data/graph_labels.pt")

print("Step 2: graph loaded")
print("x shape =", x.shape)
print("edge_index shape =", edge_index.shape)
print("y shape =", y.shape)

data = Data(x=x, edge_index=edge_index, y=y)

# =========================
# 2. Train/test masks
# =========================
num_nodes = y.shape[0]
indices = np.arange(num_nodes)

normal_idx = np.where(y.numpy() == 0)[0]
anom_idx = np.where(y.numpy() == 1)[0]

np.random.seed(42)
np.random.shuffle(normal_idx)
np.random.shuffle(anom_idx)

# use 70% train, 30% test
n_train_norm = int(0.7 * len(normal_idx))
n_train_anom = max(1, int(0.7 * len(anom_idx)))

train_idx = np.concatenate([
    normal_idx[:n_train_norm],
    anom_idx[:n_train_anom]
])

test_idx = np.array([i for i in indices if i not in train_idx])

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

print("Step 3: masks created")
print("Train nodes =", int(train_mask.sum()))
print("Test nodes  =", int(test_mask.sum()))
print("Train anomalies =", int(y[train_mask].sum()))
print("Test anomalies  =", int(y[test_mask].sum()))

# =========================
# 3. GraphSAGE model
# =========================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(
    in_channels=x.shape[1],
    hidden_channels=16,
    out_channels=2
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# class weighting because anomalies are rare
train_labels = y[train_mask]
num_normal = (train_labels == 0).sum().item()
num_anom = (train_labels == 1).sum().item()

weight_normal = 1.0
weight_anom = num_normal / max(1, num_anom)

class_weights = torch.tensor([weight_normal, weight_anom], dtype=torch.float)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

print("Step 4: model initialized")
print("Class weights =", class_weights)

# =========================
# 4. Train
# =========================
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

print("Step 5: model trained")

# =========================
# 5. Evaluate
# =========================
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

y_true = y[test_mask].cpu().numpy()
y_pred = preds[test_mask.cpu().numpy()]
y_score = probs[test_mask.cpu().numpy()]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# guard against tiny-class issues
if len(np.unique(y_true)) > 1:
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
else:
    roc_auc = float("nan")
    pr_auc = float("nan")

cm = confusion_matrix(y_true, y_pred)

print("\n===== GraphSAGE Results =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# =========================
# 6. Save results
# =========================
import pandas as pd

result_df = pd.DataFrame({
    "node_id": np.arange(num_nodes),
    "true_label": y.numpy(),
    "pred_label": preds,
    "anomaly_score": probs
})
result_df.to_csv("outputs/graphsage_results.csv", index=False)

print("\nSaved results to outputs/graphsage_results.csv")