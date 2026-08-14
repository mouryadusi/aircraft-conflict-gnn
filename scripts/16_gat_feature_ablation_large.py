"""
Large-dataset GAT feature ablation.

Input:
    processed_data/graphs_large.pt

Configurations:
    1. full
       distance_now + vertical_distance + relative_speed
       + relative_heading + tcpa + dcpa

    2. no_cpa
       distance_now + vertical_distance
       + relative_speed + relative_heading

    3. separation_only
       distance_now + vertical_distance

    4. cpa_only
       tcpa + dcpa

Output:
    processed_data/gat_feature_ablation_large.csv
    models/gat_ablation_<configuration>.pth
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from torch import nn
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

INPUT_FILE = "processed_data/graphs_large.pt"
RESULT_FILE = "processed_data/gat_feature_ablation_large.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

NUM_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("LARGE-DATASET GAT FEATURE ABLATION")
print("=" * 70)
print("Device:", device)
print("Seed:", SEED)


# ============================================================
# LOAD GRAPHS
# ============================================================

graphs = torch.load(
    INPUT_FILE,
    weights_only=False,
)

print()
print("Loaded graphs:", len(graphs))


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

graphs = sorted(
    graphs,
    key=lambda g: g.time,
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(graphs)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_graphs = graphs[:train_end]
val_graphs = graphs[train_end:val_end]
test_graphs = graphs[val_end:]


print()
print(
    f"Train: {len(train_graphs)} | "
    f"Val: {len(val_graphs)} | "
    f"Test: {len(test_graphs)}"
)

print()
print(
    f"Train time: {train_graphs[0].time} -> "
    f"{train_graphs[-1].time}"
)

print(
    f"Val time  : {val_graphs[0].time} -> "
    f"{val_graphs[-1].time}"
)

print(
    f"Test time : {test_graphs[0].time} -> "
    f"{test_graphs[-1].time}"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

EDGE_FEATURE_NAMES = [
    "distance_now",
    "vertical_distance",
    "relative_speed",
    "relative_heading",
    "tcpa",
    "dcpa",
]


CONFIGURATIONS = {

    "full": [
        "distance_now",
        "vertical_distance",
        "relative_speed",
        "relative_heading",
        "tcpa",
        "dcpa",
    ],

    "no_cpa": [
        "distance_now",
        "vertical_distance",
        "relative_speed",
        "relative_heading",
    ],

    "separation_only": [
        "distance_now",
        "vertical_distance",
    ],

    "cpa_only": [
        "tcpa",
        "dcpa",
    ],
}


# ============================================================
# GRAPH FEATURE REDUCTION
# ============================================================

def prepare_graphs(graph_list, selected_features):

    indices = [
        EDGE_FEATURE_NAMES.index(name)
        for name in selected_features
    ]

    prepared = []

    for g in graph_list:

        new_g = Data(
            x=g.x.clone(),
            edge_index=g.edge_index.clone(),
            edge_attr=g.edge_attr[:, indices].clone(),
            y=g.y.clone(),
        )

        new_g.time = g.time

        prepared.append(new_g)

    return prepared


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
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1,
            ),
        )


    def forward(self, data):

        x = F.relu(
            self.gat1(
                data.x,
                data.edge_index,
                data.edge_attr,
            )
        )

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
# TRAINING
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
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

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

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
            probs >= 0.5
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
# RUN ONE CONFIGURATION
# ============================================================

def run_configuration(
    name,
    selected_features,
):

    print()
    print("=" * 70)
    print("CONFIGURATION:", name)
    print("=" * 70)

    print(
        "Edge features:",
        ", ".join(selected_features),
    )

    # --------------------------------------------------------
    # Prepare graphs
    # --------------------------------------------------------

    train = prepare_graphs(
        train_graphs,
        selected_features,
    )

    val = prepare_graphs(
        val_graphs,
        selected_features,
    )

    test = prepare_graphs(
        test_graphs,
        selected_features,
    )

    node_dim = train[0].x.shape[1]
    edge_dim = train[0].edge_attr.shape[1]

    print()
    print("Node dimension:", node_dim)
    print("Edge dimension:", edge_dim)

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    train_edges = sum(
        g.y.numel()
        for g in train
    )

    train_positive = sum(
        int(g.y.sum())
        for g in train
    )

    pos_weight = torch.tensor(
        [
            train_edges /
            max(train_positive, 1)
        ],
        dtype=torch.float,
    )

    print(
        "Training edges:",
        train_edges,
    )

    print(
        "Training positives:",
        train_positive,
    )

    print(
        "pos_weight:",
        pos_weight.item(),
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ConflictGAT(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_f1 = -1.0
    best_state = None

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f} | "
            f"Recall {val_metrics['recall']:.4f} | "
            f"Precision {val_metrics['precision']:.4f} | "
            f"ROC-AUC {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }


    print()
    print(
        "Best validation F1:",
        best_f1,
    )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model_path = (
        f"{MODEL_DIR}/"
        f"gat_ablation_{name}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    print(
        "Saved:",
        model_path,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
    )

    print()
    print("TEST RESULTS")

    for key, value in test_metrics.items():

        print(
            f"{key:10s}: {value:.6f}"
        )

    return {

        "Model": "GAT",

        "Configuration": name,

        "Features": "+".join(
            selected_features
        ),

        "Precision": test_metrics[
            "precision"
        ],

        "Recall": test_metrics[
            "recall"
        ],

        "F1": test_metrics[
            "f1"
        ],

        "ROC_AUC": test_metrics[
            "roc_auc"
        ],

        "Best_Val_F1": best_f1,
    }


# ============================================================
# RUN ALL CONFIGURATIONS
# ============================================================

results = []

for name, features in CONFIGURATIONS.items():

    set_seed(SEED)

    result = run_configuration(
        name,
        features,
    )

    results.append(
        result
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print()
print("=" * 70)
print("FINAL GAT FEATURE ABLATION")
print("=" * 70)

print(
    results_df[
        [
            "Model",
            "Configuration",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "Best_Val_F1",
        ]
    ].to_string(index=False)
)

results_df.to_csv(
    RESULT_FILE,
    index=False,
)

print()
print(
    "Saved:",
    RESULT_FILE,
)

print()
print("DONE: GAT feature ablation")