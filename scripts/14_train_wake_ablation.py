import json
from pyexpat import model
from unittest import loader
import random
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("WAKE-ENHANCED GAT TRAINING")
print("=" * 60)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

graphs = torch.load(
    "processed_data/graphs_wake.pt",
    weights_only=False,
)

print("Loaded wake graphs:", len(graphs))

train_graphs = graphs[:503]
val_graphs = graphs[503:612]
test_graphs = graphs[612:]

print(
    f"Train: {len(train_graphs)} | "
    f"Val: {len(val_graphs)} | "
    f"Test: {len(test_graphs)}"
)

train_loader = DataLoader(
    train_graphs,
    batch_size=16,
    shuffle=True,
)

val_loader = DataLoader(
    val_graphs,
    batch_size=16,
)

test_loader = DataLoader(
    test_graphs,
    batch_size=16,
)

positive = 0
negative = 0

for g in train_graphs:
    positive += int(g.y.sum())
    negative += len(g.y) - int(g.y.sum())

print("Train positives:", positive)

pos_weight = torch.tensor(
    [negative / positive],
    device=device,
)

print("pos_weight:", pos_weight.item())


class ConflictGAT(torch.nn.Module):

    def __init__(self, node_dim):

        super().__init__()

        self.conv1 = GATConv(
            node_dim,
            64,
            heads=4,
            concat=True,
        )

        self.conv2 = GATConv(
            64 * 4,
            64,
            heads=1,
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, data):

        x = self.conv1(
            data.x,
            data.edge_index,
        )

        x = F.elu(x)

        x = self.conv2(
            x,
            data.edge_index,
        )

        src = x[data.edge_index[0]]
        dst = x[data.edge_index[1]]

        edge = self.edge_mlp(
            data.edge_attr,
        )

        out = torch.cat(
            [src, dst, edge],
            dim=1,
        )
        return self.classifier(out).squeeze()
def evaluate(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            logits = model(data)

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
    )


model = ConflictGAT(
    node_dim=13,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight,
)

best_f1 = 0.0
best_state = None

for epoch in range(30):

    model.train()

    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        logits = model(data)

        loss = criterion(
            logits,
            data.y.float(),
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    y_true, y_pred, y_prob = evaluate(
        model,
        val_loader,
    )

    val_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if val_f1 > best_f1:

        best_f1 = val_f1
        best_state = model.state_dict()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train {total_loss:.4f} | "
        f"Val F1 {val_f1:.4f}"
    )

print("\nBest Validation F1:", best_f1)

if best_state is not None:

    model.load_state_dict(best_state)

torch.save(
    best_state,
    "models/conflict_gat_wake.pth",
)

y_true, y_pred, y_prob = evaluate(
    model,
    test_loader,
)

results = {

    "model": "GAT",

    "feature_set": "trajectory+phase+wake",

    "n_node_features": 13,

    "n_edge_features": 9,

    "precision": precision_score(
        y_true,
        y_pred,
        zero_division=0,
    ),

    "recall": recall_score(
        y_true,
        y_pred,
        zero_division=0,
    ),

    "f1": f1_score(
        y_true,
        y_pred,
        zero_division=0,
    ),

    "roc_auc": roc_auc_score(
        y_true,
        y_prob,
    ),

    "best_val_f1": best_f1,
}

print("\nWake GAT Test Results")
print(results)

with open(
    "processed_data/gat_wake_results.json",
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )

print("\nSaved:")
print("models/conflict_gat_wake.pth")
print("processed_data/gat_wake_results.json")

print("\nDONE: 14_train_wake_ablation.py")