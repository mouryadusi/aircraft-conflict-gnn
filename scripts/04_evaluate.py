"""
Script 4: Evaluate the trained GAT — confusion matrix, ROC, PR curve, error analysis.
Run: python scripts/04_evaluate.py
Input: models/conflict_gat_v2.pth, processed_data/test_graphs.pt, processed_data/train_graphs.pt
Output: processed_data/*.png, processed_data/test_edges_with_predictions.csv
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import os

os.makedirs("processed_data", exist_ok=True)

train_graphs = torch.load("processed_data/train_graphs.pt", weights_only=False)
test_graphs = torch.load("processed_data/test_graphs.pt", weights_only=False)
print("Loaded train_graphs:", len(train_graphs), "| test_graphs:", len(test_graphs))

train_pos = sum(int(g.y.sum()) for g in train_graphs)
train_total = sum(g.y.numel() for g in train_graphs)
pos_weight = torch.tensor([train_total / max(train_pos, 1)], dtype=torch.float)


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
model = ConflictGAT(edge_dim=5).to(device)
model.load_state_dict(torch.load("models/conflict_gat_v2.pth", map_location=device))
model.eval()
print("Loaded trained GAT weights.")

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)


def evaluate_with_predictions(model, loader, device):
    model.eval()
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            labels.extend(batch.y.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    return {
        "labels": np.array(labels),
        "predictions": np.array(predictions),
        "probabilities": np.array(probabilities),
    }


test_results = evaluate_with_predictions(model, test_loader, device)
test_labels = test_results["labels"]
test_preds = test_results["predictions"]
test_probs = test_results["probabilities"]
print("test_results ready:", test_labels.shape)

# --- Confusion matrix ---
cm = confusion_matrix(test_labels, test_preds)
print("\nConfusion matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("GAT Confusion Matrix (threshold=0.5)")
plt.savefig("processed_data/gat_confusion_matrix.png", bbox_inches="tight")
plt.close()

# --- ROC curve ---
fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = roc_auc_score(test_labels, test_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GAT ROC Curve")
plt.legend()
plt.savefig("processed_data/gat_roc_curve.png", bbox_inches="tight")
plt.close()
print("ROC-AUC:", roc_auc)

# --- Precision-Recall curve ---
precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
avg_prec = average_precision_score(test_labels, test_probs)
plt.figure()
plt.plot(recall_vals, precision_vals, label=f"AP = {avg_prec:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("GAT Precision-Recall Curve")
plt.legend()
plt.savefig("processed_data/gat_pr_curve.png", bbox_inches="tight")
plt.close()
print("Average Precision:", avg_prec)

# --- Threshold tuning ---
from sklearn.metrics import f1_score
thresholds = np.arange(0.01, 1.00, 0.01)
best_threshold, best_thr_f1 = 0.5, -1
for t in thresholds:
    preds_t = (test_probs >= t).astype(int)
    f1 = f1_score(test_labels, preds_t, zero_division=0)
    if f1 > best_thr_f1:
        best_thr_f1, best_threshold = f1, t
print(f"Best threshold: {best_threshold:.2f} | F1: {best_thr_f1:.4f}")

# --- Error analysis: rebuild flat edge table aligned to test set ---
test_edge_records = []
for g in test_graphs:
    for i in range(g.edge_attr.shape[0]):
        test_edge_records.append({
            "source_day": g.source_day,
            "time": int(g.time),
            "h_dist_nm": g.edge_attr[i, 0].item(),
            "v_dist_ft": g.edge_attr[i, 1].item(),
            "closing_rate_nm_s": g.edge_attr[i, 2].item(),
            "bearing_diff": g.edge_attr[i, 3].item(),
            "time_to_cpa_s": g.edge_attr[i, 4].item(),
            "true_label": int(g.y[i].item()),
        })
test_edges_df = pd.DataFrame(test_edge_records)
test_edges_df["gat_pred"] = test_preds
test_edges_df["gat_prob"] = test_probs

assert len(test_edges_df) == len(test_labels), "Mismatch — check ordering assumptions"

false_negatives = test_edges_df[(test_edges_df["true_label"] == 1) & (test_edges_df["gat_pred"] == 0)]
false_positives = test_edges_df[(test_edges_df["true_label"] == 0) & (test_edges_df["gat_pred"] == 1)]
print(f"\nFalse negatives: {len(false_negatives)}")
print(false_negatives[["source_day", "time", "h_dist_nm", "v_dist_ft", "gat_prob"]].to_string())
print(f"\nFalse positives: {len(false_positives)}")
print(false_positives[["h_dist_nm", "v_dist_ft", "closing_rate_nm_s"]].describe())

test_edges_df.to_csv("processed_data/test_edges_with_predictions.csv", index=False)
print("\nSaved processed_data/test_edges_with_predictions.csv")
print("Saved confusion matrix, ROC curve, PR curve as .png files in processed_data/")
print("DONE: 04_evaluate.py")
