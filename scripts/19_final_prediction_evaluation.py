
"""
Script 19
Final prediction-level evaluation for GCN and GAT.

Purpose:
1. Load the already-trained GCN model.
2. Rebuild/load the chronological test graphs.
3. Generate genuine GCN edge-level probabilities and predictions.
4. Generate GCN confusion matrix + classification report.
5. Validate the existing GAT prediction file.
6. Generate final comparison tables and plots.
7. Update the final evaluation package.

Run:
    python scripts/19_final_prediction_evaluation.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = "processed_data/graphs_large.pt"
GCN_MODEL_PATH = "models/conflict_gcn_large.pth"

GAT_PRED_PATH = "processed_data/final_gat_predictions.csv"

OUT_DIR = "processed_data/final_evaluation_package"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("FINAL PREDICTION-LEVEL EVALUATION")
print("=" * 70)

print("Device:", device)


# ============================================================
# LOAD GRAPHS
# ============================================================

print("\nLoading graphs...")

graphs = torch.load(
    GRAPH_PATH,
    weights_only=False,
)

print("Loaded graphs:", len(graphs))


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

graphs_sorted = sorted(
    graphs,
    key=lambda g: int(g.time)
)

n = len(graphs_sorted)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_graphs = graphs_sorted[:train_end]
val_graphs = graphs_sorted[train_end:val_end]
test_graphs = graphs_sorted[val_end:]

print("\nChronological split")
print("-" * 40)

print("Train:", len(train_graphs))
print("Val  :", len(val_graphs))
print("Test :", len(test_graphs))

print(
    "Test time:",
    int(test_graphs[0].time),
    "->",
    int(test_graphs[-1].time),
)


# ============================================================
# TEST LOADER
# ============================================================

test_loader = DataLoader(
    test_graphs,
    batch_size=32,
    shuffle=False,
)


# ============================================================
# GCN MODEL
# ============================================================

class ConflictGCN(nn.Module):

    def __init__(self):

        super().__init__()

        self.gcn1 = GCNConv(
            in_channels=6,
            out_channels=64,
        )

        self.gcn2 = GCNConv(
            in_channels=64,
            out_channels=64,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(64 * 2 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):

        x = F.relu(
            self.gcn1(
                data.x,
                data.edge_index,
            )
        )

        x = self.gcn2(
            x,
            data.edge_index,
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

        logits = self.edge_mlp(edge_features)

        return logits.squeeze(-1)


# ============================================================
# LOAD GCN
# ============================================================

print("\nLoading GCN model...")

gcn = ConflictGCN().to(device)

state = torch.load(
    GCN_MODEL_PATH,
    map_location=device,
    weights_only=False,
)

gcn.load_state_dict(state)

gcn.eval()

print("GCN model loaded successfully.")


# ============================================================
# GENERATE GCN PREDICTIONS
# ============================================================

print("\nGenerating GCN test predictions...")

gcn_labels = []
gcn_probabilities = []
gcn_predictions = []

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        logits = gcn(batch)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).long()

        gcn_labels.extend(
            batch.y.detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

        gcn_probabilities.extend(
            probabilities.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        gcn_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )


gcn_labels = np.asarray(gcn_labels, dtype=int)
gcn_probabilities = np.asarray(gcn_probabilities)
gcn_predictions = np.asarray(gcn_predictions, dtype=int)


# ============================================================
# BASIC CHECKS
# ============================================================

print("\nGCN prediction statistics")
print("-" * 40)

print("Samples:", len(gcn_labels))

print(
    "Positive labels:",
    int(gcn_labels.sum()),
)

print(
    "Positive predictions:",
    int(gcn_predictions.sum()),
)

print(
    "Probability range:",
    float(gcn_probabilities.min()),
    "->",
    float(gcn_probabilities.max()),
)


# ============================================================
# SAVE RAW GCN PREDICTIONS
# ============================================================

gcn_prediction_df = pd.DataFrame(
    {
        "label": gcn_labels,
        "probability": gcn_probabilities,
        "prediction": gcn_predictions,
    }
)

gcn_prediction_path = (
    "processed_data/final_gcn_predictions.csv"
)

gcn_prediction_df.to_csv(
    gcn_prediction_path,
    index=False,
)

print(
    "\nSaved:",
    gcn_prediction_path,
)


# ============================================================
# GCN CONFUSION MATRIX
# ============================================================

gcn_cm = confusion_matrix(
    gcn_labels,
    gcn_predictions,
)

print("\nGCN Confusion Matrix")
print(gcn_cm)


# ============================================================
# GCN CLASSIFICATION REPORT
# ============================================================

gcn_report = classification_report(
    gcn_labels,
    gcn_predictions,
    zero_division=0,
)

print("\nGCN Classification Report")
print(gcn_report)


# ============================================================
# GCN METRICS
# ============================================================

gcn_precision = precision_score(
    gcn_labels,
    gcn_predictions,
    zero_division=0,
)

gcn_recall = recall_score(
    gcn_labels,
    gcn_predictions,
    zero_division=0,
)

gcn_f1 = f1_score(
    gcn_labels,
    gcn_predictions,
    zero_division=0,
)

try:

    gcn_roc_auc = roc_auc_score(
        gcn_labels,
        gcn_probabilities,
    )

except ValueError:

    gcn_roc_auc = float("nan")


# ============================================================
# SAVE GCN METRICS
# ============================================================

gcn_metrics = {
    "samples": int(len(gcn_labels)),
    "positive_labels": int(gcn_labels.sum()),
    "positive_predictions": int(gcn_predictions.sum()),
    "precision": float(gcn_precision),
    "recall": float(gcn_recall),
    "f1": float(gcn_f1),
    "roc_auc": float(gcn_roc_auc),
    "confusion_matrix": gcn_cm.tolist(),
}


with open(
    f"{OUT_DIR}/gcn_final_metrics.json",
    "w",
) as f:

    json.dump(
        gcn_metrics,
        f,
        indent=4,
    )


# ============================================================
# GCN CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

im = ax.imshow(
    gcn_cm,
    interpolation="nearest",
)

ax.set_title(
    "GCN Confusion Matrix"
)

ax.set_xlabel(
    "Predicted label"
)

ax.set_ylabel(
    "True label"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Non-conflict", "Conflict"]
)

ax.set_yticklabels(
    ["Non-conflict", "Conflict"]
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            str(gcn_cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()

gcn_cm_path = (
    f"{OUT_DIR}/gcn_confusion_matrix_final.png"
)

plt.savefig(
    gcn_cm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    gcn_cm_path,
)


# ============================================================
# LOAD / VALIDATE GAT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING GAT RESULTS")
print("=" * 70)

gat_df = pd.read_csv(
    GAT_PRED_PATH
)

required_columns = {
    "label",
    "probability",
    "prediction",
}

if not required_columns.issubset(
    gat_df.columns
):

    raise ValueError(
        "GAT prediction file does not contain "
        "label/probability/prediction columns."
    )


gat_labels = (
    gat_df["label"]
    .astype(int)
    .to_numpy()
)

gat_probabilities = (
    gat_df["probability"]
    .astype(float)
    .to_numpy()
)

gat_predictions = (
    gat_df["prediction"]
    .astype(int)
    .to_numpy()
)

print(
    "GAT samples:",
    len(gat_labels),
)

print(
    "GAT positive labels:",
    int(gat_labels.sum()),
)

print(
    "GAT positive predictions:",
    int(gat_predictions.sum()),
)


# ============================================================
# GAT METRICS
# ============================================================

gat_precision = precision_score(
    gat_labels,
    gat_predictions,
    zero_division=0,
)

gat_recall = recall_score(
    gat_labels,
    gat_predictions,
    zero_division=0,
)

gat_f1 = f1_score(
    gat_labels,
    gat_predictions,
    zero_division=0,
)

try:

    gat_roc_auc = roc_auc_score(
        gat_labels,
        gat_probabilities,
    )

except ValueError:

    gat_roc_auc = float("nan")


gat_cm = confusion_matrix(
    gat_labels,
    gat_predictions,
)

print("\nGAT Confusion Matrix")
print(gat_cm)

print("\nGAT Classification Report")

print(
    classification_report(
        gat_labels,
        gat_predictions,
        zero_division=0,
    )
)


# ============================================================
# SAVE GAT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

ax.imshow(
    gat_cm,
    interpolation="nearest",
)

ax.set_title(
    "GAT Confusion Matrix"
)

ax.set_xlabel(
    "Predicted label"
)

ax.set_ylabel(
    "True label"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Non-conflict", "Conflict"]
)

ax.set_yticklabels(
    ["Non-conflict", "Conflict"]
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            str(gat_cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()

gat_cm_path = (
    f"{OUT_DIR}/gat_confusion_matrix_final.png"
)

plt.savefig(
    gat_cm_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved:",
    gat_cm_path,
)


# ============================================================
# FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "Model": "GCN",
            "Features": "Trajectory",
            "Precision": gcn_precision,
            "Recall": gcn_recall,
            "F1": gcn_f1,
            "ROC_AUC": gcn_roc_auc,
        },
        {
            "Model": "GAT",
            "Features": "Trajectory + CPA",
            "Precision": gat_precision,
            "Recall": gat_recall,
            "F1": gat_f1,
            "ROC_AUC": gat_roc_auc,
        },
    ]
)


print("\n" + "=" * 70)
print("FINAL PREDICTION-LEVEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


comparison_path = (
    f"{OUT_DIR}/final_prediction_level_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# SAVE JSON
# ============================================================

final_results = {
    "GCN": gcn_metrics,
    "GAT": {
        "samples": int(len(gat_labels)),
        "positive_labels": int(gat_labels.sum()),
        "positive_predictions": int(
            gat_predictions.sum()
        ),
        "precision": float(gat_precision),
        "recall": float(gat_recall),
        "f1": float(gat_f1),
        "roc_auc": float(gat_roc_auc),
        "confusion_matrix": gat_cm.tolist(),
    },
}


with open(
    f"{OUT_DIR}/final_prediction_level_results.json",
    "w",
) as f:

    json.dump(
        final_results,
        f,
        indent=4,
    )


# ============================================================
# FINAL REPORT
# ============================================================

report_path = (
    f"{OUT_DIR}/FINAL_PREDICTION_LEVEL_REPORT.md"
)

with open(
    report_path,
    "w",
) as f:

    f.write(
        "# Final Prediction-Level Evaluation\n\n"
    )

    f.write(
        "## GCN\n\n"
    )

    f.write(
        f"- Samples: {len(gcn_labels):,}\n"
    )

    f.write(
        f"- Positive labels: {int(gcn_labels.sum()):,}\n"
    )

    f.write(
        f"- Positive predictions: {int(gcn_predictions.sum()):,}\n"
    )

    f.write(
        f"- Precision: {gcn_precision:.6f}\n"
    )

    f.write(
        f"- Recall: {gcn_recall:.6f}\n"
    )

    f.write(
        f"- F1: {gcn_f1:.6f}\n"
    )

    f.write(
        f"- ROC-AUC: {gcn_roc_auc:.6f}\n\n"
    )

    f.write(
        "## GAT\n\n"
    )

    f.write(
        f"- Samples: {len(gat_labels):,}\n"
    )

    f.write(
        f"- Positive labels: {int(gat_labels.sum()):,}\n"
    )

    f.write(
        f"- Positive predictions: {int(gat_predictions.sum()):,}\n"
    )

    f.write(
        f"- Precision: {gat_precision:.6f}\n"
    )

    f.write(
        f"- Recall: {gat_recall:.6f}\n"
    )

    f.write(
        f"- F1: {gat_f1:.6f}\n"
    )

    f.write(
        f"- ROC-AUC: {gat_roc_auc:.6f}\n\n"
    )

    f.write(
        "## Confusion Matrices\n\n"
    )

    f.write(
        "### GCN\n\n"
    )

    f.write(
        "```\n"
        f"{gcn_cm}\n"
        "```\n\n"
    )

    f.write(
        "### GAT\n\n"
    )

    f.write(
        "```\n"
        f"{gat_cm}\n"
        "```\n"
    )


print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nSaved:",
    comparison_path,
)

print(
    "Saved:",
    f"{OUT_DIR}/final_prediction_level_results.json",
)

print(
    "Saved:",
    report_path,
)

print(
    "\nGCN F1:",
    f"{gcn_f1:.6f}",
)

print(
    "GAT F1:",
    f"{gat_f1:.6f}",
)

print("\nDONE")

