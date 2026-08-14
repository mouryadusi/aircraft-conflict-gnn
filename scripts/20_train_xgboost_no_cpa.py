"""
Script 20
Final XGBoost No-CPA Ablation

Purpose
-------
Evaluate XGBoost using ONLY observable pairwise state features.

CPA-derived features are removed:
    - time_to_cpa
    - DCPA / CPA distance

Remaining features:
    0. horizontal distance
    1. vertical distance
    2. closing rate
    3. bearing difference

Run:
    python scripts/20_train_xgboost_no_cpa.py

Outputs:
    processed_data/final_evaluation_package/
        xgboost_no_cpa_final_predictions.csv
        xgboost_no_cpa_final_metrics.json
        xgboost_no_cpa_confusion_matrix_final.png
        xgboost_no_cpa_feature_importance.png
        XGBOOST_NO_CPA_FINAL_REPORT.md
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

GRAPH_PATH = "processed_data/graphs_large.pt"

OUTPUT_DIR = "processed_data/final_evaluation_package"

os.makedirs(OUTPUT_DIR, exist_ok=True)


PREDICTION_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_no_cpa_final_predictions.csv",
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_no_cpa_final_metrics.json",
)

CONFUSION_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_no_cpa_confusion_matrix_final.png",
)

IMPORTANCE_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_no_cpa_feature_importance.png",
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "XGBOOST_NO_CPA_FINAL_REPORT.md",
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "horizontal_distance",
    "vertical_distance",
    "closing_rate",
    "bearing_difference",
]

# IMPORTANT:
#
# Graph edge attributes have six columns:
#
#   0 = horizontal distance
#   1 = vertical distance
#   2 = closing rate
#   3 = bearing difference
#   4 = time-to-CPA
#   5 = DCPA
#
# Therefore this ablation deliberately keeps ONLY:
#
#   [0, 1, 2, 3]
#
# and removes:
#
#   [4, 5]
#
OBSERVABLE_FEATURE_INDICES = [0, 1, 2, 3]


# ============================================================
# LOAD GRAPHS
# ============================================================

print("=" * 80)
print("FINAL XGBOOST NO-CPA ABLATION")
print("=" * 80)

print("\nLoading graphs...")

graphs = torch.load(
    GRAPH_PATH,
    weights_only=False,
)

print(f"Loaded graphs: {len(graphs):,}")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

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

print(f"Train: {len(train_graphs):,}")
print(f"Val  : {len(val_graphs):,}")
print(f"Test : {len(test_graphs):,}")

print(
    f"Train time: "
    f"{int(train_graphs[0].time)} -> "
    f"{int(train_graphs[-1].time)}"
)

print(
    f"Val time  : "
    f"{int(val_graphs[0].time)} -> "
    f"{int(val_graphs[-1].time)}"
)

print(
    f"Test time : "
    f"{int(test_graphs[0].time)} -> "
    f"{int(test_graphs[-1].time)}"
)


# ============================================================
# BUILD TABULAR DATA
# ============================================================

def graphs_to_arrays(graph_list):

    X_parts = []
    y_parts = []

    for g in graph_list:

        edge_attr = g.edge_attr.detach().cpu().numpy()
        labels = g.y.detach().cpu().numpy()

        # Keep ONLY observable state features.
        features = edge_attr[:, OBSERVABLE_FEATURE_INDICES]

        X_parts.append(features)
        y_parts.append(labels)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y


print("\nBuilding tabular datasets...")

X_train, y_train = graphs_to_arrays(train_graphs)
X_val, y_val = graphs_to_arrays(val_graphs)
X_test, y_test = graphs_to_arrays(test_graphs)

print("\nShapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

train_positive = int(y_train.sum())
train_negative = int(len(y_train) - train_positive)

val_positive = int(y_val.sum())
test_positive = int(y_test.sum())

print("Training positives:", train_positive)
print("Training negatives:", train_negative)
print("Validation positives:", val_positive)
print("Test positives:", test_positive)

scale_pos_weight = (
    train_negative / max(train_positive, 1)
)

print(
    "scale_pos_weight:",
    scale_pos_weight,
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST NO-CPA")
print("=" * 80)

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_val, y_val),
    ],
    verbose=False,
)

print("XGBoost training complete.")


# ============================================================
# VALIDATION THRESHOLD CALIBRATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION THRESHOLD CALIBRATION")
print("=" * 80)

val_prob = model.predict_proba(
    X_val
)[:, 1]

best_threshold = 0.5
best_val_f1 = -1.0

threshold_results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.01,
):

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": float(threshold),
            "f1": float(f1),
        }
    )

    if f1 > best_val_f1:

        best_val_f1 = f1
        best_threshold = float(threshold)


print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best validation F1: "
    f"{best_val_f1:.6f}"
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST NO-CPA TEST RESULTS")
print("=" * 80)

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_prob,
)

cm = confusion_matrix(
    y_test,
    test_pred,
)


print(f"Samples   : {len(y_test):,}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1        : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        zero_division=0,
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    {
        "label": y_test.astype(int),
        "probability": test_prob,
        "prediction": test_pred.astype(int),
    }
)

predictions_df.to_csv(
    PREDICTION_PATH,
    index=False,
)

print(
    "\nSaved:",
    PREDICTION_PATH,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "XGBoost",
    "feature_set": "observable_state_no_cpa",
    "features": FEATURE_NAMES,
    "removed_features": [
        "time_to_cpa",
        "dcpa",
    ],
    "total_graphs": len(graphs),
    "train_graphs": len(train_graphs),
    "validation_graphs": len(val_graphs),
    "test_graphs": len(test_graphs),
    "test_samples": int(len(y_test)),
    "train_positive": train_positive,
    "train_negative": train_negative,
    "validation_positive": val_positive,
    "test_positive": test_positive,
    "best_validation_threshold": best_threshold,
    "best_validation_f1": best_val_f1,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
}


with open(
    METRICS_PATH,
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


print(
    "Saved:",
    METRICS_PATH,
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title(
    "XGBoost No-CPA Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    ["Non-conflict", "Conflict"],
)

plt.yticks(
    [0, 1],
    ["Non-conflict", "Conflict"],
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=200,
)

plt.close()

print(
    "Saved:",
    CONFUSION_PATH,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "importance": importance,
    }
).sort_values(
    "importance",
    ascending=False,
)


plt.figure(
    figsize=(8, 5)
)

plt.barh(
    importance_df["feature"],
    importance_df["importance"],
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Feature importance"
)

plt.title(
    "XGBoost No-CPA Feature Importance"
)

plt.tight_layout()

plt.savefig(
    IMPORTANCE_PATH,
    dpi=200,
)

plt.close()

print(
    "Saved:",
    IMPORTANCE_PATH,
)


# ============================================================
# FINAL REPORT
# ============================================================

report = f"""# Final XGBoost No-CPA Ablation

## Purpose

This experiment evaluates XGBoost using only observable pairwise aircraft-state features.

CPA-derived features were deliberately removed to determine whether the strong XGBoost performance depends primarily on CPA/DCPA information.

## Features retained

1. Horizontal distance
2. Vertical distance
3. Closing rate
4. Bearing difference

## Features removed

1. Time to CPA
2. DCPA / CPA distance

## Dataset

- Total graphs: {len(graphs):,}
- Training graphs: {len(train_graphs):,}
- Validation graphs: {len(val_graphs):,}
- Test graphs: {len(test_graphs):,}

The split is chronological.

## Threshold calibration

- Best validation threshold: {best_threshold:.2f}
- Validation F1: {best_val_f1:.6f}

## Test results

| Metric | Result |
|---|---:|
| Precision | {precision:.6f} |
| Recall | {recall:.6f} |
| F1 | {f1:.6f} |
| ROC-AUC | {roc_auc:.6f} |

## Confusion matrix

{cm}
"""