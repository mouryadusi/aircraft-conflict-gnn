"""
Script 20
Final XGBoost edge-level baseline

Input:
    processed_data/graphs_large.pt

Output:
    processed_data/final_evaluation_package/
        xgboost_final_predictions.csv
        xgboost_final_metrics.json
        xgboost_confusion_matrix_final.png
        xgboost_feature_importance.png
        XGBOOST_FINAL_REPORT.md

Purpose:
    Train an XGBoost classifier using the six edge-level features
    stored in graphs_large.pt.

    The graph dataset is split chronologically:
        70% train
        15% validation
        15% test

    The validation set is used to select the probability threshold
    that maximizes F1.

    The test set is evaluated only once using that frozen threshold.

IMPORTANT:
    This is an edge-level tabular baseline, NOT a graph neural network.
"""

import os
import json
import random

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

INPUT_PATH = "processed_data/graphs_large.pt"

OUTPUT_DIR = (
    "processed_data/final_evaluation_package"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FINAL XGBOOST EVALUATION")
print("=" * 80)

print("\nLoading graphs...")

graphs = torch.load(
    INPUT_PATH,
    weights_only=False,
)

print(f"Loaded graphs: {len(graphs):,}")


# ============================================================
# CHRONOLOGICAL SORT
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

val_graphs = graphs_sorted[
    train_end:val_end
]

test_graphs = graphs_sorted[
    val_end:
]

print(f"Train: {len(train_graphs):,}")
print(f"Val  : {len(val_graphs):,}")
print(f"Test : {len(test_graphs):,}")

print(
    f"\nTrain time: "
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
# FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTING EDGE FEATURES")
print("=" * 80)


def extract_edges(graph_list):

    features = []
    labels = []

    for graph in graph_list:

        edge_attr = graph.edge_attr.detach().cpu().numpy()
        y = graph.y.detach().cpu().numpy()

        features.append(edge_attr)
        labels.append(y)

    X = np.concatenate(
        features,
        axis=0,
    )

    y = np.concatenate(
        labels,
        axis=0,
    )

    return X, y


X_train, y_train = extract_edges(
    train_graphs
)

X_val, y_val = extract_edges(
    val_graphs
)

X_test, y_test = extract_edges(
    test_graphs
)


print("\nFeature dimensions:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

train_positive = int(y_train.sum())
train_negative = int(len(y_train) - train_positive)

val_positive = int(y_val.sum())
val_negative = int(len(y_val) - val_positive)

test_positive = int(y_test.sum())
test_negative = int(len(y_test) - test_positive)

print(
    f"Train positives: {train_positive:,}"
)

print(
    f"Train negatives: {train_negative:,}"
)

print(
    f"Val positives:   {val_positive:,}"
)

print(
    f"Val negatives:   {val_negative:,}"
)

print(
    f"Test positives:  {test_positive:,}"
)

print(
    f"Test negatives:  {test_negative:,}"
)


# ============================================================
# CLASS WEIGHT
# ============================================================

scale_pos_weight = (
    train_negative /
    max(train_positive, 1)
)

print(
    "\nscale_pos_weight:",
    scale_pos_weight,
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

model = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    gamma=0,

    reg_alpha=0,

    reg_lambda=1,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=SEED,

    n_jobs=-1,

)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],

    verbose=True,
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION THRESHOLD CALIBRATION")
print("=" * 80)

val_prob = model.predict_proba(
    X_val
)[:, 1]


thresholds = np.arange(
    0.05,
    0.96,
    0.01,
)


best_threshold = 0.50
best_val_f1 = -1.0

threshold_results = []


for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )

    if f1 > best_val_f1:

        best_val_f1 = f1
        best_threshold = float(
            threshold
        )


print(
    f"\nBest validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best validation F1: "
    f"{best_val_f1:.6f}"
)


# ============================================================
# SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xgboost_threshold_analysis.csv",
    ),
    index=False,
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("FINAL XGBOOST TEST EVALUATION")
print("=" * 80)

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

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


print("\nXGBoost TEST RESULTS")

print(
    f"Samples   : {len(y_test):,}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1        : {f1:.6f}"
)

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    test_pred,
    zero_division=0,
)

print(report)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "label": y_test,
        "probability": test_prob,
        "prediction": test_pred,
    }
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_final_predictions.csv",
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)

print(
    "\nSaved:",
    prediction_path,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "XGBoost",

    "feature_set":
        "six edge-level graph features",

    "n_graphs":
        int(n),

    "train_graphs":
        int(len(train_graphs)),

    "validation_graphs":
        int(len(val_graphs)),

    "test_graphs":
        int(len(test_graphs)),

    "train_samples":
        int(len(y_train)),

    "validation_samples":
        int(len(y_val)),

    "test_samples":
        int(len(y_test)),

    "train_positive":
        train_positive,

    "train_negative":
        train_negative,

    "validation_positive":
        val_positive,

    "validation_negative":
        val_negative,

    "test_positive":
        test_positive,

    "test_negative":
        test_negative,

    "scale_pos_weight":
        float(scale_pos_weight),

    "best_validation_threshold":
        float(best_threshold),

    "best_validation_f1":
        float(best_val_f1),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "confusion_matrix":
        cm.tolist(),
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_final_metrics.json",
)


with open(
    metrics_path,
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


print(
    "Saved:",
    metrics_path,
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm,
)

plt.title(
    "XGBoost Confusion Matrix"
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

cm_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_confusion_matrix_final.png",
)

plt.savefig(
    cm_path,
    dpi=200,
)

plt.close()

print(
    "Saved:",
    cm_path,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = [

    "edge_feature_0",

    "edge_feature_1",

    "edge_feature_2",

    "edge_feature_3",

    "edge_feature_4",

    "edge_feature_5",
]


importance = model.feature_importances_


importance_df = pd.DataFrame(
    {
        "feature":
            feature_names,

        "importance":
            importance,
    }
).sort_values(
    "importance",
    ascending=False,
)


importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xgboost_feature_importance.csv",
    ),
    index=False,
)


plt.figure(
    figsize=(8, 5)
)

plt.barh(
    importance_df["feature"],
    importance_df["importance"],
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

importance_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_feature_importance.png",
)

plt.savefig(
    importance_path,
    dpi=200,
)

plt.close()

print(
    "Saved:",
    importance_path,
)


# ============================================================
# FINAL REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "XGBOOST_FINAL_REPORT.md",
)


with open(
    report_path,
    "w",
) as f:

    f.write(
        "# Final XGBoost Evaluation\n\n"
    )

    f.write(
        "## Model\n\n"
    )

    f.write(
        "XGBoost was trained as an edge-level "
        "tabular baseline using the six edge "
        "features stored in the rebuilt graph dataset.\n\n"
    )

    f.write(
        "## Dataset split\n\n"
    )

    f.write(
        f"- Total graphs: {n:,}\n"
    )

    f.write(
        f"- Training graphs: {len(train_graphs):,}\n"
    )

    f.write(
        f"- Validation graphs: {len(val_graphs):,}\n"
    )

    f.write(
        f"- Test graphs: {len(test_graphs):,}\n\n"
    )

    f.write(
        "The split was chronological to prevent "
        "future information from entering training.\n\n"
    )

    f.write(
        "## Validation threshold\n\n"
    )

    f.write(
        f"- Best threshold: {best_threshold:.2f}\n"
    )

    f.write(
        f"- Validation F1: {best_val_f1:.6f}\n\n"
    )

    f.write(
        "## Test results\n\n"
    )

    f.write(
        "| Metric | Result |\n"
        "|---|---:|\n"
    )

    f.write(
        f"| Precision | {precision:.6f} |\n"
    )

    f.write(
        f"| Recall | {recall:.6f} |\n"
    )

    f.write(
        f"| F1 | {f1:.6f} |\n"
    )

    f.write(
        f"| ROC-AUC | {roc_auc:.6f} |\n\n"
    )

    f.write(
        "## Confusion matrix\n\n"
    )

    f.write(
        "```text\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n```\n\n"
    )

    f.write(
        "## Important methodological note\n\n"
    )

    f.write(
        "This model is an edge-level tabular baseline. "
        "It does not perform message passing or explicitly "
        "model aircraft as a graph. Its purpose is to provide "
        "a conventional machine-learning reference against "
        "which the GCN and GAT models can be assessed.\n"
    )


print(
    "Saved:",
    report_path,
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST FINAL EVALUATION COMPLETE")
print("=" * 80)

print(
    f"\nXGBoost F1: {f1:.6f}"
)

print(
    f"XGBoost ROC-AUC: {roc_auc:.6f}"
)

print(
    "\nDONE"
)