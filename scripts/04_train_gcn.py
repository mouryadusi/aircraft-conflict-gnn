"""
Script 4: Train GCN on the large chronological graph dataset.

Input:
    processed_data/graphs_large.pt

Output:
    models/conflict_gcn_large.pth

The script:
    1. Loads the large graph dataset.
    2. Splits graphs chronologically.
    3. Automatically detects node and edge feature dimensions.
    4. Trains an edge-level GCN classifier.
    5. Selects the best model using validation F1.
    6. Evaluates once on the held-out test set.
"""

import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# Configuration
# ============================================================

GRAPH_PATH = "processed_data/graphs_large.pt"
MODEL_PATH = "models/conflict_gcn_large.pth"

NUM_EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
SEED = 42

os.makedirs("models", exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# ============================================================
# Load graphs
# ============================================================

graphs = torch.load(
    GRAPH_PATH,
    weights_only=False,
)

print("Loaded graphs:", len(graphs))


if len(graphs) == 0:
    raise RuntimeError("No graphs were loaded.")


# ============================================================
# Inspect feature dimensions
# ============================================================

sample = graphs[0]

NODE_DIM = sample.x.shape[1]
EDGE_DIM = sample.edge_attr.shape[1]

print("Node feature dimension:", NODE_DIM)
print("Edge feature dimension:", EDGE_DIM)
print("Sample nodes:", sample.x.shape[0])
print("Sample edges:", sample.edge_index.shape[1])
print("Sample labels:", sample.y.shape)


# ============================================================
# Chronological split
# ============================================================

graphs_sorted = sorted(
    graphs,
    key=lambda g: g.time
)

n = len(graphs_sorted)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_graphs = graphs_sorted[:train_end]
val_graphs = graphs_sorted[train_end:val_end]
test_graphs = graphs_sorted[val_end:]

print()
print(
    f"Train: {len(train_graphs)} | "
    f"Val: {len(val_graphs)} | "
    f"Test: {len(test_graphs)}"
)


# ============================================================
# Verify chronological split
# ============================================================

print()
print("Train time:", train_graphs[0].time, "->", train_graphs[-1].time)
print("Val time  :", val_graphs[0].time, "->", val_graphs[-1].time)
print("Test time :", test_graphs[0].time, "->", test_graphs[-1].time)


# ============================================================
# Save splits
# ============================================================

torch.save(
    train_graphs,
    "processed_data/train_graphs.pt"
)

torch.save(
    val_graphs,
    "processed_data/val_graphs.pt"
)

torch.save(
    test_graphs,
    "processed_data/test_graphs.pt"
)


# ============================================================
# Class balance
# ============================================================

train_pos = sum(
    int(g.y.sum())
    for g in train_graphs
)

train_total = sum(
    g.y.numel()
    for g in train_graphs
)

val_pos = sum(
    int(g.y.sum())
    for g in val_graphs
)

val_total = sum(
    g.y.numel()
    for g in val_graphs
)

test_pos = sum(
    int(g.y.sum())
    for g in test_graphs
)

test_total = sum(
    g.y.numel()
    for g in test_graphs
)

print()
print("Training edges :", train_total)
print("Training positives:", train_pos)
print("Validation edges:", val_total)
print("Validation positives:", val_pos)
print("Test edges      :", test_total)
print("Test positives   :", test_pos)


# ============================================================
# Positive class weight
# ============================================================

pos_weight = torch.tensor(
    [
        (train_total - train_pos) /
        max(train_pos, 1)
    ],
    dtype=torch.float,
)

print()
print("pos_weight:", pos_weight.item())


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# GCN model
# ============================================================

class ConflictGCN(nn.Module):

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=64,
    ):
        super().__init__()

        self.gcn1 = GCNConv(
            node_dim,
            hidden_dim,
        )

        self.gcn2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + edge_dim,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, data):

        x = self.gcn1(
            data.x,
            data.edge_index,
        )

        x = F.relu(x)

        x = self.gcn2(
            x,
            data.edge_index,
        )

        x = F.relu(x)

        src = x[data.edge_index[0]]
        dst = x[data.edge_index[1]]

        edge_input = torch.cat(
            [
                src,
                dst,
                data.edge_attr,
            ],
            dim=1,
        )

        return self.edge_mlp(
            edge_input
        ).squeeze(-1)


# ============================================================
# Model
# ============================================================

model = ConflictGCN(
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    hidden_dim=HIDDEN_DIM,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(device)
)


# ============================================================
# Training
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)

        loss = criterion(
            logits,
            batch.y.float(),
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ============================================================
# Evaluation
# ============================================================

def evaluate(loader):

    model.eval()

    losses = []

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            logits = model(batch)

            loss = criterion(
                logits,
                batch.y.float(),
            )

            probs = torch.sigmoid(logits)

            preds = (
                probs >= 0.5
            ).long()

            losses.append(
                loss.item()
            )

            y_true.extend(
                batch.y.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

            y_prob.extend(
                probs.cpu().numpy()
            )

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )

    except ValueError:

        roc_auc = float("nan")

    return {
        "loss": float(np.mean(losses)),
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
        "roc_auc": roc_auc,
    }


# ============================================================
# Train loop
# ============================================================

best_f1 = -1.0
best_state = None

print()
print("=" * 70)
print("TRAINING GCN")
print("=" * 70)

for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    train_loss = train_epoch()

    val = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train {train_loss:.4f} | "
        f"Val F1 {val['f1']:.4f} | "
        f"Recall {val['recall']:.4f} | "
        f"Precision {val['precision']:.4f} | "
        f"ROC-AUC {val['roc_auc']:.4f}"
    )

    if val["f1"] > best_f1:

        best_f1 = val["f1"]

        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }


# ============================================================
# Restore best validation model
# ============================================================

print()
print("Best validation F1:", best_f1)

if best_state is None:
    raise RuntimeError(
        "No best model state was captured."
    )

model.load_state_dict(
    best_state
)


# ============================================================
# Save
# ============================================================

torch.save(
    best_state,
    MODEL_PATH,
)

print(
    f"Saved {MODEL_PATH}"
)


# ============================================================
# Final test evaluation
# ============================================================

test = evaluate(
    test_loader
)

print()
print("=" * 70)
print("GCN TEST RESULTS")
print("=" * 70)

print(
    f"loss      : {test['loss']}"
)

print(
    f"precision : {test['precision']}"
)

print(
    f"recall    : {test['recall']}"
)

print(
    f"f1        : {test['f1']}"
)

print(
    f"roc_auc   : {test['roc_auc']}"
)

print()
print("DONE: 04_train_gcn.py")

