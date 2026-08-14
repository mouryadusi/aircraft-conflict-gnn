
"""
Script 3
Train GAT on rebuilt graph dataset

Run:
    python scripts/03_train_gat.py

Input:
    processed_data/graphs_large.pt

Output:
    models/conflict_gat_v2.pth
    processed_data/train_graphs.pt
    processed_data/val_graphs.pt
    processed_data/test_graphs.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

GRAPH_PATH = "processed_data/graphs_large.pt"

graphs = torch.load(
    GRAPH_PATH,
    weights_only=False,
)

print(f"Loaded graphs: {len(graphs):,}")

if len(graphs) == 0:
    raise RuntimeError("No graphs were loaded.")

# ============================================================
# FEATURE DIMENSIONS
# ============================================================

sample = graphs[0]

NODE_DIM = sample.x.shape[1]
EDGE_DIM = sample.edge_attr.shape[1]

print("Node feature dimension:", NODE_DIM)
print("Edge feature dimension:", EDGE_DIM)

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

graphs_sorted = sorted(
    graphs,
    key=lambda g: int(g.time),
)

n = len(graphs_sorted)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_graphs = graphs_sorted[:train_end]
val_graphs = graphs_sorted[train_end:val_end]
test_graphs = graphs_sorted[val_end:]

print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(f"Train : {len(train_graphs):,}")
print(f"Val   : {len(val_graphs):,}")
print(f"Test  : {len(test_graphs):,}")

print()
print(
    "Train time:",
    train_graphs[0].time,
    "->",
    train_graphs[-1].time,
)

print(
    "Val time  :",
    val_graphs[0].time,
    "->",
    val_graphs[-1].time,
)

print(
    "Test time :",
    test_graphs[0].time,
    "->",
    test_graphs[-1].time,
)

# ============================================================
# SAVE SPLITS
# ============================================================

torch.save(
    train_graphs,
    "processed_data/train_graphs.pt",
)

torch.save(
    val_graphs,
    "processed_data/val_graphs.pt",
)

torch.save(
    test_graphs,
    "processed_data/test_graphs.pt",
)

# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_statistics(dataset):

    total_edges = 0
    positives = 0

    for graph in dataset:

        total_edges += graph.y.numel()
        positives += int(graph.y.sum())

    return total_edges, positives


train_total, train_pos = dataset_statistics(
    train_graphs
)

val_total, val_pos = dataset_statistics(
    val_graphs
)

test_total, test_pos = dataset_statistics(
    test_graphs
)

print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("Training edges    :", train_total)
print("Training positives:", train_pos)

print("Validation edges  :", val_total)
print("Validation positives:", val_pos)

print("Test edges        :", test_total)
print("Test positives    :", test_pos)

# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

negative_count = train_total - train_pos

pos_weight_value = (
    negative_count / max(train_pos, 1)
)

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float,
)

print()
print("Negative training edges:", negative_count)
print("Positive training edges:", train_pos)
print("pos_weight:", pos_weight.item())

# ============================================================
# DATALOADERS
# ============================================================

BATCH_SIZE = 32

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
# MODEL
# ============================================================

class ConflictGAT(nn.Module):

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=64,
    ):

        super().__init__()

        self.gat1 = GATConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            heads=2,
            edge_dim=edge_dim,
            concat=True,
        )

        self.gat2 = GATConv(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
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

        x = self.gat1(
            data.x,
            data.edge_index,
            data.edge_attr,
        )

        x = F.relu(x)

        x = self.gat2(
            x,
            data.edge_index,
            data.edge_attr,
        )

        src = x[data.edge_index[0]]

        dst = x[data.edge_index[1]]

        edge_features = torch.cat(
            [
                src,
                dst,
                data.edge_attr,
            ],
            dim=1,
        )

        logits = self.edge_mlp(
            edge_features
        )

        return logits.squeeze(-1)

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print()
print("Device:", device)

# ============================================================
# MODEL / OPTIMIZER / LOSS
# ============================================================

model = ConflictGAT(
    node_dim=NODE_DIM,
    edge_dim=EDGE_DIM,
    hidden_dim=64,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(device)
)

# ============================================================
# TRAIN FUNCTION
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

    return total_loss / max(len(train_loader), 1)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    losses = []

    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(batch)

        loss = criterion(
            logits,
            batch.y.float(),
        )

        losses.append(
            loss.item()
        )

        probs = torch.sigmoid(logits)

        preds = (
            probs > 0.5
        ).long()

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

        "loss": float(
            np.mean(losses)
        ),

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
# TRAIN LOOP
# ============================================================

EPOCHS = 30

best_f1 = -1.0

best_state = None

print()
print("=" * 70)
print("TRAINING GAT")
print("=" * 70)

for epoch in range(
    1,
    EPOCHS + 1,
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
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

# ============================================================
# RESTORE BEST MODEL
# ============================================================

print()
print("=" * 70)
print("BEST VALIDATION MODEL")
print("=" * 70)

print(
    "Best validation F1:",
    best_f1,
)

if best_state is None:

    raise RuntimeError(
        "No best model state was captured."
    )

model.load_state_dict(
    best_state
)

# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = (
    "models/conflict_gat_v2.pth"
)

torch.save(
    best_state,
    MODEL_PATH,
)

print(
    "Saved:",
    MODEL_PATH,
)

# ============================================================
# FINAL TEST
# ============================================================

test = evaluate(
    test_loader
)

print()
print("=" * 70)
print("GAT TEST RESULTS")
print("=" * 70)

print(
    f"loss      : {test['loss']:.6f}"
)

print(
    f"precision : {test['precision']:.6f}"
)

print(
    f"recall    : {test['recall']:.6f}"
)

print(
    f"f1        : {test['f1']:.6f}"
)

print(
    f"roc_auc   : {test['roc_auc']:.6f}"
)

print()
print("DONE")

