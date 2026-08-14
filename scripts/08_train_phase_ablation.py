import random
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

graphs = torch.load("processed_data/graphs_phase.pt", weights_only=False)
print("Loaded phase-extended graphs:", len(graphs))

graphs_sorted = sorted(graphs, key=lambda g: (g.source_day, g.time))
n = len(graphs_sorted)
train_end = int(0.70 * n)
val_end = int(0.85 * n)
train_graphs = graphs_sorted[:train_end]
val_graphs = graphs_sorted[train_end:val_end]
test_graphs = graphs_sorted[val_end:]
print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

train_pos = sum(int(g.y.sum()) for g in train_graphs)
train_total = sum(g.y.numel() for g in train_graphs)
val_pos = sum(int(g.y.sum()) for g in val_graphs)
test_pos = sum(int(g.y.sum()) for g in test_graphs)
print(f"Train positives: {train_pos} | Val positives: {val_pos} | Test positives: {test_pos}")

train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)

pos_weight = torch.tensor([train_total / max(train_pos, 1)], dtype=torch.float)
print("pos_weight:", pos_weight.item())

class ConflictGAT(nn.Module):
    def __init__(self, node_dim=9, edge_dim=6, hidden_dim=64):
        super().__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=2, concat=True, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, data):
        x = F.elu(self.gat1(data.x, data.edge_index, data.edge_attr))
        x = self.gat2(x, data.edge_index, data.edge_attr)
        src, dst = x[data.edge_index[0]], x[data.edge_index[1]]
        edge_input = torch.cat([src, dst, data.edge_attr], dim=1)
        return self.edge_mlp(edge_input).squeeze(-1)

device = torch.device("cpu")
model = ConflictGAT(node_dim=9, edge_dim=6).to(device)
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

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

best_f1, best_state = 0.0, None
for epoch in range(1, 31):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    y_true, y_pred, _ = evaluate(model, val_loader, device)
    val_f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val F1 {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

print("\nBest validation F1 (Phase-extended GAT):", best_f1)

if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(best_state, "models/conflict_gat_phase.pth")

y_true, y_pred, y_prob = evaluate(model, test_loader, device)
results = {
    "model": "GAT",
    "feature_set": "trajectory+phase",
    "n_node_features": 9,
    "n_edge_features": 6,
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_true, y_prob),
    "best_val_f1": best_f1,
}
print("\nPhase-extended GAT test results:", results)

with open("processed_data/gat_phase_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved models/conflict_gat_phase.pth and processed_data/gat_phase_results.json")
print("DONE: 08_train_phase_ablation.py")
