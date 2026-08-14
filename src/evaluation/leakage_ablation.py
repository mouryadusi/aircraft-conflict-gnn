"""
Feature Leakage Ablation Experiments
-------------------------------------
Save as: src/evaluation/leakage_ablation.py  (or drop into 07_model_analysis.ipynb)
Repo:    aircraft-conflict-gnn

Tests whether conflict labels are leaking through tcpa/dcpa, which were
likely also used in the CPA-based label rule itself.

Run all three experiments and compare. If Experiment 2 (CPA-only) reproduces
near-perfect metrics, leakage is confirmed, and Experiment 3 (no-CPA) becomes
your real, reportable result.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, auc, classification_report
)

REPO_ROOT = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path(".")
PROCESSED_DIR = REPO_ROOT / "processed_data"
RESULTS_DIR = REPO_ROOT / "results" / "metrics"

# -----------------------------------------------------------------------
# STEP 0 — Inspect what's actually in processed_data/ before assuming
# column names. Run this block first, on its own, and confirm the
# printed columns match FEATURE_SETS below before running experiments.
# -----------------------------------------------------------------------
print("Files in processed_data/:")
for f in sorted(PROCESSED_DIR.glob("*")):
    print(" -", f.name)

candidate_csv = PROCESSED_DIR / "candidate_pairs_cpa.csv"
if not candidate_csv.exists():
    raise FileNotFoundError(
        f"{candidate_csv} not found. List processed_data/ above, find the "
        f"CSV with the flat feature columns (distance_now, relative_speed, "
        f"relative_heading, vertical_distance, tcpa, dcpa, conflict), and "
        f"set DATA_PATH to it manually before continuing."
    )

df = pd.read_csv(candidate_csv)
print("\nShape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -----------------------------------------------------------------------
# STEP 1 — If the printed columns above don't match REQUIRED_COLS, stop
# here, check region_final.csv instead (or merge the two), and fix
# DATA_PATH / column names before proceeding to the experiments below.
# -----------------------------------------------------------------------
REQUIRED_COLS = [
    "distance_now", "relative_speed", "relative_heading",
    "vertical_distance", "tcpa", "dcpa", "conflict",
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected columns: {missing}. Check region_final.csv or "
        f"your feature-engineering step for renamed/different columns."
    )

LABEL_COL = "conflict"

FEATURE_SETS = {
    "exp1_full_features": [
        "distance_now", "relative_speed", "relative_heading",
        "vertical_distance", "tcpa", "dcpa",
    ],
    "exp2_cpa_only": [
        "tcpa", "dcpa",
    ],
    "exp3_no_cpa": [
        "distance_now", "relative_speed", "relative_heading",
        "vertical_distance",
    ],
}

# -----------------------------------------------------------------------
# 2. Important: split BEFORE any leakage-prone preprocessing, and make sure
# the split doesn't let the same encounter/flight pair appear in both
# train and test (that's a second, subtler leakage source — group split
# by flight-pair or time window if you haven't already).
# -----------------------------------------------------------------------
def run_experiment(name, feature_cols, df, label_col=LABEL_COL, seed=42):
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)

    results = {
        "experiment": name,
        "features": feature_cols,
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return results, model


if __name__ == "__main__":
    all_results = []
    for name, cols in FEATURE_SETS.items():
        res, model = run_experiment(name, cols, df)
        all_results.append(res)
        print(f"\n=== {name} ===")
        print(f"Features: {cols}")
        print(f"F1: {res['f1']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | "
              f"PR-AUC: {res['pr_auc']:.4f}")
        print(f"Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f}")
        print(f"Confusion matrix:\n{np.array(res['confusion_matrix'])}")

    results_df = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "leakage_ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    # -------------------------------------------------------------------
    # Interpretation guide:
    # - exp1 (full) sets the upper bound.
    # - exp2 (cpa_only) near-perfect (F1/ROC-AUC ~1.0) => leakage CONFIRMED,
    #   because two features alone are reproducing the label rule.
    # - exp3 (no_cpa) is your scientifically valid result — report THIS
    #   as your headline number, with exp1/exp2 shown as the leakage
    #   diagnostic in a dedicated "Data Leakage Investigation" section.
    # -------------------------------------------------------------------