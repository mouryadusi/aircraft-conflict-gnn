
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

SEED = 42
BATCH_SIZE = 64
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS = 30

GRAPH_PATH = "processed_data/graphs_large.pt"
MODEL_PATH = "models/conflict_gat_cpa_only.pth"
RESULT_PATH = "processed_data/gat_cpa_only_results.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# VALIDATE ORIGINAL GRAPH STRUCTURE
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
# CREATE CPA-ONLY GRAPH COPIES
# ============================================================

cpa_graphs = []

for g in graphs:

    # --------------------------------------------------------
    # Original edge feature layout:
    #
    # 0 = distance_now
    # 1 = vertical_distance
    # 2 = relative_speed
    # 3 = relative_heading
    # 4 = tcpa
    # 5 = dcpa
    #
    # Keep ONLY:
    # 4 = tcpa
    # 5 = dcpa
    # --------------------------------------------------------

    new_edge_attr = g.edge_attr[:, [4, 5]].clone()

    new_g = Data(
        x=g.x.clone(),
        edge_index=g.edge_index.clone(),
        edge_attr=new_edge_attr,
        y=g.y.clone(),
    )

    new_g.time = int(g.time)

    cpa_graphs.append(new_g)


print()
print("=" * 70)
print("CPA-ONLY GRAPH")
print("=" * 70)

print("Node features :", cpa_graphs[0].x.shape)
print("Edge features :", cpa_graphs[0].edge_attr.shape)
print("Labels        :", cpa_graphs[0].y.shape)


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

graphs_sorted = sorted(
    cpa_graphs,
    key=lambda g: g.time
)


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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
# EDGE COUNTS
# ============================================================

def dataset_statistics(dataset):

    total_edges = 0
    positives = 0

    for g in dataset:
        total_edges += g.y.numel()
        positives += int(g.y.sum())

    return total_edges, positives


train_total, train_pos = dataset_statistics(train_graphs)
val_total, val_pos = dataset_statistics(val_graphs)
test_total, test_pos = dataset_statistics(test_graphs)


print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

print("Training edges   :", train_total)
print("Training positive:", train_pos)

print("Validation edges :", val_total)
print("Validation positive:", val_pos)

print("Test edges       :", test_total)
print("Test positive    :", test_pos)


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

negative_count = train_total - train_pos

pos_weight_value = (
    negative_count / max(train_pos, 1)
)

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
# CPA-ONLY GAT
# ============================================================

class ConflictGAT(nn.Module):

    def __init__(
        self,
        node_dim=6,
        edge_dim=2,
        hidden_dim=64
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

        x = F.relu(
            self.gat1(
                data.x,
                data.edge_index,
                data.edge_attr
            )
        )

        x = self.gat2(
            x,
            data.edge_index,
            data.edge_attr
        )

        src = x[data.edge_index[0]]
        dst = x[data.edge_index[1]]

        edge_input = torch.cat(
            [
                src,
                dst,
                data.edge_attr
            ],
            dim=1
        )

        return self.edge_mlp(
            edge_input
        ).squeeze(-1)


# ============================================================
# MODEL
# ============================================================

model = ConflictGAT(
    node_dim=6,
    edge_dim=2,
    hidden_dim=HIDDEN_DIM,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
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
    device
):

    model.train()

    total_loss = 0.0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)

        loss = criterion(
            logits,
            batch.y.float()
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
    device
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            logits = model(batch)

            loss = criterion(
                logits,
                batch.y.float()
            )

            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_labels.extend(
                batch.y.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    y_true = np.asarray(
        all_labels
    )

    y_prob = np.asarray(
        all_probs
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
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
print("TRAINING CPA-ONLY GAT")
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
        device
    )

    val_results = evaluate(
        model,
        val_loader,
        criterion,
        device
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
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST VALIDATION MODEL
# ============================================================

model.load_state_dict(
    best_state
)

print()
print(
    "Best validation F1:",
    best_val_f1
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
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
    device
)


print()
print("=" * 70)
print("CPA-ONLY GAT TEST RESULTS")
print("=" * 70)

for key, value in test_results.items():

    print(
        f"{key:10s}: {value}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "GAT",
    "feature_set": "CPA-only",
    "edge_features": [
        "tcpa",
        "dcpa"
    ],
    "node_feature_dimension": 6,
    "edge_feature_dimension": 2,
    "seed": SEED,
    "best_val_f1": best_val_f1,
    **test_results,
}


with open(
    RESULT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print(
    "Saved:",
    RESULT_PATH
)

print()
print("=" * 70)
print("DONE: CPA-ONLY GAT")
print("=" * 70)