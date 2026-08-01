"""
Script 3: Chronological split + train the GAT model.
Run: python scripts/03_train_gat.py
Input: processed_data/graphs_large.pt
Output: models/conflict_gat_v2.pth, processed_data/{train,val,test}_graphs.pt
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

os.makedirs("models", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

graphs = torch.load("processed_data/graphs_large.pt", weights_only=False)
print("Loaded graphs:", len(graphs))

# --- Chronological split ---
graphs_sorted = sorted(graphs, key=lambda g: (g.source_day, g.time))
n = len(graphs_sorted)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_graphs = graphs_sorted[:train_end]
val_graphs = graphs_sorted[train_end:val_end]
test_graphs = graphs_sorted[val_end:]
print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

torch.save(train_graphs, "processed_data/train_graphs.pt")
torch.save(val_graphs, "processed_data/val_graphs.pt")
torch.save(test_graphs, "processed_data/test_graphs.pt")

train_pos = sum(int(g.y.sum()) for g in train_graphs)
train_total = sum(g.y.numel() for g in train_graphs)
val_pos = sum(int(g.y.sum()) for g in val_graphs)
test_pos = sum(int(g.y.sum()) for g in test_graphs)
print(f"Train positives: {train_pos} | Val positives: {val_pos} | Test positives: {test_pos}")

# --- DataLoaders ---
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)

pos_weight = torch.tensor([train_total / max(train_pos, 1)], dtype=torch.float)
print("pos_weight:", pos_weight.item())


class ConflictGAT(nn.Module):
    def __init__(self, node_dim=6, edge_dim=5, hidden_dim=64):
        super().__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=2, edge_dim=edge_dim, concat=True)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, edge_dim=edge_dim, concat=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = F.relu(self.gat1(data.x, data.edge_index, data.edge_attr))
        x = self.gat2(x, data.edge_index, data.edge_attr)
        src, dst = x[data.edge_index[0]], x[data.edge_index[1]]
        edge_input = torch.cat([src, dst, data.edge_attr], dim=1)
        return self.edge_mlp(edge_input).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = ConflictGAT(edge_dim=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float("nan")
    return {
        "loss": np.mean(losses),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc,
    }


# --- Train ---
NUM_EPOCHS = 30
best_f1, best_state = 0.0, None

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val F1 {val_metrics['f1']:.4f} | Recall {val_metrics['recall']:.4f} | Precision {val_metrics['precision']:.4f}")
    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

print("\nBest validation F1:", best_f1)

if best_state is not None:
    torch.save(best_state, "models/conflict_gat_v2.pth")
    print("Saved models/conflict_gat_v2.pth")

print("DONE: 03_train_gat.py")
