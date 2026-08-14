"""
Script 17: Train GAT without CPA-derived features.

Purpose:
    Leakage-control / ablation experiment.

Uses ONLY observable/current-state edge features:
    1. distance_now
    2. vertical_distance
    3. relative_speed
    4. relative_heading

EXCLUDES:
    5. tcpa
    6. dcpa

Input:
    processed_data/graphs_large.pt

Output:
    models/conflict_gat_no_cpa.pth
    processed_data/gat_no_cpa_results.json

Run:
    python scripts/17_train_gat_no_cpa.py
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = "processed_data/graphs_large.pt"
MODEL_PATH = "models/conflict_gat_no_cpa.pth"
RESULT_PATH = "processed_data/gat_no_cpa_results.json"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

NODE_DIM = 6

# Original edge features:
# 0 distance_now
# 1 vertical_distance
# 2 relative_speed
# 3 relative_heading
# 4 tcpa
# 5 dcpa
#
# No-CPA experiment keeps only 0:4.
NO_CPA_EDGE_DIM = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", device)


# ============================================================
# LOAD GRAPHS
# ============================================================

graphs = torch.load(
    GRAPH_PATH,
    weights_only=False
)

print("Loaded graphs:", len(graphs))


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

graphs = sorted(
    graphs,
    key=lambda g: g.time
)


# ============================================================
# INSPECT ORIGINAL GRAPH
# ============================================================

sample = graphs[0]

print()
print("=" * 70)
print("ORIGINAL GRAPH")
print("=" * 70)

print("Node features :", sample.x.shape)
print("Edge features :", sample.edge_attr.shape)
print("Labels        :", sample.y.shape)
print("Graph keys    :", sample.keys())


# ============================================================
# CREATE NO-CPA GRAPHS
# ============================================================

no_cpa_graphs = []

for g in graphs:

    # Keep only observable/current-state features.
    edge_attr = g.edge_attr[:, :NO_CPA_EDGE_DIM].clone()

    new_g = Data(
        x=g.x.clone(),
        edge_index=g.edge_index.clone(),
        edge_attr=edge_attr,
        y=g.y.clone(),
    )

    # Preserve timestamp for chronological splitting.
    new_g.time = int(g.time)

    no_cpa_graphs.append(new_g)


graphs = no_cpa_graphs


# ============================================================
# VERIFY NO-CPA GRAPH
# ============================================================

sample = graphs[0]

print()
print("=" * 70)
print("NO-CPA GRAPH")
print("=" * 70)

print("Node features :", sample.x.shape)
print("Edge features :", sample.edge_attr.shape)
print("Labels        :", sample.y.shape)
print("Graph keys    :", sample.keys())

print()
print("No-CPA edge feature dimension:",
      sample.edge_attr.shape[1])

assert sample.edge_attr.shape[1] == 4, (
    "ERROR: No-CPA graph does not have exactly 4 edge features."
)


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(graphs)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_graphs = graphs[:train_end]
val_graphs = graphs[train_end:val_end]
test_graphs = graphs[val_end:]


print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print("Train:", len(train_graphs))
print("Val  :", len(val_graphs))
print("Test :", len(test_graphs))

print()

print(
    "Train time:",
    train_graphs[0].time,
    "->",
    train_graphs[-1].time
)

print(
    "Val time  :",
    val_graphs[0].time,
    "->",
    val_graphs[-1].time
)

print(
    "Test time :",
    test_graphs[0].time,
    "->",
    test_graphs[-1].time
)


# ============================================================
# LABEL STATISTICS
# ============================================================

def dataset_stats(graph_list):

    total_edges = 0
    total_positive = 0

    for g in graph_list:

        total_edges += g.y.numel()
        total_positive += int(g.y.sum())

    return total_edges, total_positive


train_edges, train_pos = dataset_stats(train_graphs)
val_edges, val_pos = dataset_stats(val_graphs)
test_edges, test_pos = dataset_stats(test_graphs)


print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("Training edges   :", train_edges)
print("Training positive:", train_pos)

print("Validation edges :", val_edges)
print("Validation positive:", val_pos)

print("Test edges       :", test_edges)
print("Test positive    :", test_pos)


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

train_neg = train_edges - train_pos

pos_weight_value = train_neg / max(train_pos, 1)

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float
)

print()
print("pos_weight:", pos_weight.item())


# ============================================================
# DATA LOADERS
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
# GAT MODEL
# ============================================================

class ConflictGAT(nn.Module):

    def __init__(
        self,
        node_dim=6,
        edge_dim=4,
        hidden_dim=64,
    ):

        super().__init__()

        self.gat1 = GATConv(
            node_dim,
            hidden_dim,
            heads=2,
            edge_dim=edge_dim,
            concat=True,
        )

        self.gat2 = GATConv(
            hidden_dim * 2,
            hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
        )

        self.edge_mlp = nn.Sequential(

            nn.Linear(
                hidden_dim * 2 + edge_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
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
# MODEL / OPTIMIZER / LOSS
# ============================================================

model = ConflictGAT(
    node_dim=NODE_DIM,
    edge_dim=NO_CPA_EDGE_DIM,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(device)
)


# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    for batch in loader:

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

    return total_loss / len(loader)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            logits = model(batch)

            loss = criterion(
                logits,
                batch.y.float(),
            )

            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                batch.y.cpu().numpy()
            )

    all_probs = np.asarray(
        all_probs
    )

    all_labels = np.asarray(
        all_labels
    )

    predictions = (
        all_probs >= 0.5
    ).astype(int)

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        all_labels,
        all_probs,
    )

    return {
        "loss": total_loss / len(loader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING NO-CPA GAT")
print("=" * 70)

best_val_f1 = -1.0
best_state = None

for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
    )

    val_results = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train {train_loss:.4f} | "
        f"Val F1 {val_results['f1']:.4f} | "
        f"Recall {val_results['recall']:.4f} | "
        f"Precision {val_results['precision']:.4f} | "
        f"ROC-AUC {val_results['roc_auc']:.4f}"
    )

    if val_results["f1"] > best_val_f1:

        best_val_f1 = val_results["f1"]

        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)

model.to(device)


print()
print(
    "Best validation F1:",
    best_val_f1
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH,
)

print(
    "Saved:",
    MODEL_PATH
)


# ============================================================
# FINAL TEST
# ============================================================

test_results = evaluate(
    model,
    test_loader,
    criterion,
    device,
)


print()
print("=" * 70)
print("NO-CPA GAT TEST RESULTS")
print("=" * 70)

print(
    f"loss      : {test_results['loss']}"
)

print(
    f"precision : {test_results['precision']}"
)

print(
    f"recall    : {test_results['recall']}"
)

print(
    f"f1        : {test_results['f1']}"
)

print(
    f"roc_auc   : {test_results['roc_auc']}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "processed_data",
    exist_ok=True
)

results_to_save = {
    "model": "GAT",
    "features": "no_cpa",
    "edge_features": [
        "distance_now",
        "vertical_distance",
        "relative_speed",
        "relative_heading",
    ],
    "excluded_features": [
        "tcpa",
        "dcpa",
    ],
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "train_graphs": len(train_graphs),
    "validation_graphs": len(val_graphs),
    "test_graphs": len(test_graphs),
    "train_edges": train_edges,
    "train_positive": train_pos,
    "validation_edges": val_edges,
    "validation_positive": val_pos,
    "test_edges": test_edges,
    "test_positive": test_pos,
    "best_validation_f1": best_val_f1,
    "test_results": test_results,
}


with open(
    RESULT_PATH,
    "w"
) as f:

    json.dump(
        results_to_save,
        f,
        indent=4,
    )


print()
print("Saved:", RESULT_PATH)

print()
print("=" * 70)
print("DONE: 17_train_gat_no_cpa.py")
print("=" * 70)
