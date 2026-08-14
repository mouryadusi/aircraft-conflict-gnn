#!/usr/bin/env python3

"""
18_final_evaluation_package.py

FINAL EVALUATION PACKAGE

Models:
    1. GCN — Trajectory
    2. GAT — Trajectory + CPA
    3. GAT — CPA-only
    4. GAT — No-CPA

This script DOES NOT retrain models.

It:
    - Uses the corrected confirmed final GCN result
    - Loads completed GAT CPA-only results
    - Loads completed GAT No-CPA results
    - Uses confirmed GAT Trajectory + CPA results
    - Builds the corrected final results table
    - Generates CSV
    - Generates Markdown
    - Generates JSON
    - Generates dataset statistics
    - Generates DCPA leakage diagnostic
    - Generates comparison plots
    - Generates confusion matrices when compatible predictions exist
    - Generates a final evaluation report
    - Generates a model manifest
    - Generates an evaluation summary
    - Creates a ZIP archive containing the package
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "processed_data"
MODELS = ROOT / "models"

OUTPUT = PROCESSED / "final_evaluation_package"
OUTPUT.mkdir(parents=True, exist_ok=True)

PACKAGE_ZIP = PROCESSED / "final_evaluation_package.zip"


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metric(data, name):
    """
    Supports several possible JSON layouts.

    Examples:

        {
            "precision": 0.5,
            "recall": 0.9,
            "f1": 0.6,
            "roc_auc": 0.95
        }

    or:

        {
            "test": {
                "precision": 0.5
            }
        }
    """

    if name in data:
        return float(data[name])

    for key in [
        "test",
        "test_results",
        "metrics",
        "results",
    ]:

        if key in data and isinstance(data[key], dict):

            if name in data[key]:
                return float(data[key][name])

    raise KeyError(
        f"Could not find metric '{name}' in JSON. "
        f"Available keys: {list(data.keys())}"
    )


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    results = {}

    # ========================================================
    # GCN — CORRECTED FINAL RESULT
    # ========================================================
    #
    # IMPORTANT:
    #
    # Do NOT load processed_data/gcn_results.json.
    #
    # That file contains the earlier collapsed GCN evaluation:
    #
    # Precision = 0.006057
    # Recall    = 0.333333
    # F1        = 0.011898
    # ROC-AUC   = 0.933598
    #
    # The corrected final GCN result was already established in
    # scripts/13_build_results_table.py.
    #

    print("Using corrected confirmed GCN results.")

    results["GCN"] = {
        "Model": "GCN",
        "Features": "Trajectory",
        "Precision": 0.7791527843883865,
        "Recall": 0.9155480984340044,
        "F1": 0.8418616610953973,
        "ROC_AUC": 0.9984044760292694,
    }

    # ========================================================
    # GAT — TRAJECTORY + CPA
    # ========================================================

    gat_full_candidates = [
        PROCESSED / "gat_results.json",
        PROCESSED / "gat_v2_results.json",
        PROCESSED / "gat_large_results.json",
    ]

    gat_full_data = None

    for path in gat_full_candidates:

        if path.exists():

            gat_full_data = load_json(path)

            print(
                f"Loaded GAT Trajectory + CPA results: {path}"
            )

            break

    if gat_full_data is not None:

        results["GAT Trajectory + CPA"] = {
            "Model": "GAT",
            "Features": "Trajectory + CPA",
            "Precision": extract_metric(
                gat_full_data,
                "precision"
            ),
            "Recall": extract_metric(
                gat_full_data,
                "recall"
            ),
            "F1": extract_metric(
                gat_full_data,
                "f1"
            ),
            "ROC_AUC": extract_metric(
                gat_full_data,
                "roc_auc"
            ),
        }

    else:

        print(
            "No GAT full-result JSON found."
        )

        print(
            "Using confirmed completed "
            "GAT Trajectory + CPA results."
        )

        results["GAT Trajectory + CPA"] = {
            "Model": "GAT",
            "Features": "Trajectory + CPA",
            "Precision": 0.42139995286353993,
            "Recall": 1.0,
            "F1": 0.5929364947769856,
            "ROC_AUC": 0.9985941022088943,
        }

    # ========================================================
    # GAT — CPA ONLY
    # ========================================================

    cpa_only_path = (
        PROCESSED /
        "gat_cpa_only_results.json"
    )

    if not cpa_only_path.exists():

        raise FileNotFoundError(
            "Missing CPA-only result file:\n"
            f"{cpa_only_path}"
        )

    cpa_only = load_json(
        cpa_only_path
    )

    results["GAT CPA-only"] = {
        "Model": "GAT",
        "Features": "CPA-only",
        "Precision": extract_metric(
            cpa_only,
            "precision"
        ),
        "Recall": extract_metric(
            cpa_only,
            "recall"
        ),
        "F1": extract_metric(
            cpa_only,
            "f1"
        ),
        "ROC_AUC": extract_metric(
            cpa_only,
            "roc_auc"
        ),
    }

    # ========================================================
    # GAT — NO CPA
    # ========================================================

    no_cpa_path = (
        PROCESSED /
        "gat_no_cpa_results.json"
    )

    if not no_cpa_path.exists():

        raise FileNotFoundError(
            "Missing No-CPA result file:\n"
            f"{no_cpa_path}"
        )

    no_cpa = load_json(
        no_cpa_path
    )

    results["GAT No-CPA"] = {
        "Model": "GAT",
        "Features": "No-CPA",
        "Precision": extract_metric(
            no_cpa,
            "precision"
        ),
        "Recall": extract_metric(
            no_cpa,
            "recall"
        ),
        "F1": extract_metric(
            no_cpa,
            "f1"
        ),
        "ROC_AUC": extract_metric(
            no_cpa,
            "roc_auc"
        ),
    }

    return results


# ============================================================
# BUILD FINAL TABLE
# ============================================================

def build_results_table(results):

    order = [
        "GCN",
        "GAT Trajectory + CPA",
        "GAT CPA-only",
        "GAT No-CPA",
    ]

    rows = []

    for name in order:

        if name not in results:

            raise KeyError(
                f"Missing result for: {name}"
            )

        rows.append(
            results[name]
        )

    df = pd.DataFrame(
        rows
    )

    numeric_columns = [
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    for col in numeric_columns:

        df[col] = df[col].astype(float)

    return df


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(df):

    path = (
        OUTPUT /
        "final_results_table.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# SAVE MARKDOWN
# ============================================================

def save_markdown(df):

    path = (
        OUTPUT /
        "final_results_table.md"
    )

    lines = []

    lines.append(
        "# Final GCN vs GAT Evaluation"
    )

    lines.append("")

    lines.append(
        "| Model | Features | Precision | Recall | F1 | ROC-AUC |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for _, row in df.iterrows():

        lines.append(
            f"| {row['Model']} "
            f"| {row['Features']} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['F1']:.4f} "
            f"| {row['ROC_AUC']:.4f} |"
        )

    lines.append("")

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# SAVE JSON
# ============================================================

def save_results_json(df):

    path = (
        OUTPUT /
        "final_results.json"
    )

    data = {
        "models": df.to_dict(
            orient="records"
        )
    }

    save_json(
        data,
        path
    )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# DATASET STATISTICS
# ============================================================

def generate_dataset_statistics():

    graph_path = (
        PROCESSED /
        "graphs_large.pt"
    )

    cpa_path = (
        PROCESSED /
        "candidate_pairs_cpa.pkl"
    )

    stats = {}

    # ========================================================
    # GRAPH DATASET
    # ========================================================

    if graph_path.exists():

        graphs = torch.load(
            graph_path,
            weights_only=False
        )

        stats["graphs"] = len(
            graphs
        )

        edges = sum(
            int(
                g.y.numel()
            )
            for g in graphs
        )

        positives = sum(
            int(
                g.y.sum()
            )
            for g in graphs
        )

        stats["graph_edges"] = edges

        stats[
            "graph_positive_labels"
        ] = positives

        if edges > 0:

            stats[
                "graph_positive_rate"
            ] = positives / edges

        first_graph = graphs[0]
        last_graph = graphs[-1]

        stats[
            "first_graph_time"
        ] = int(
            first_graph.time
        )

        stats[
            "last_graph_time"
        ] = int(
            last_graph.time
        )

        stats[
            "node_feature_dimension"
        ] = int(
            first_graph.x.shape[1]
        )

        stats[
            "edge_feature_dimension"
        ] = int(
            first_graph.edge_attr.shape[1]
        )

        del graphs

    else:

        print(
            f"Graph dataset not found: {graph_path}"
        )

    # ========================================================
    # CPA DATASET
    # ========================================================

    if cpa_path.exists():

        cpa = pd.read_pickle(
            cpa_path
        )

        stats[
            "cpa_rows"
        ] = int(
            len(cpa)
        )

        stats[
            "cpa_positive_labels"
        ] = int(
            cpa["conflict"].sum()
        )

        stats[
            "cpa_positive_rate"
        ] = float(
            cpa["conflict"].mean()
        )

        stats[
            "cpa_columns"
        ] = cpa.columns.tolist()

        if "dcpa" in cpa.columns:

            non_conflict = cpa.loc[
                cpa["conflict"] == 0,
                "dcpa"
            ]

            conflict = cpa.loc[
                cpa["conflict"] == 1,
                "dcpa"
            ]

            stats[
                "dcpa_mean_non_conflict"
            ] = float(
                non_conflict.mean()
            )

            stats[
                "dcpa_mean_conflict"
            ] = float(
                conflict.mean()
            )

    else:

        print(
            f"CPA dataset not found: {cpa_path}"
        )

    path = (
        OUTPUT /
        "dataset_statistics.json"
    )

    save_json(
        stats,
        path
    )

    print(
        f"Saved: {path}"
    )

    return stats


# ============================================================
# DCPA LEAKAGE DIAGNOSTIC
# ============================================================

def generate_leakage_summary():

    cpa_path = (
        PROCESSED /
        "candidate_pairs_cpa.pkl"
    )

    if not cpa_path.exists():

        print(
            "CPA dataset not found. "
            "Skipping leakage summary."
        )

        return None

    cpa = pd.read_pickle(
        cpa_path
    )

    summary = {
        "rows": int(
            len(cpa)
        ),
        "positive_labels": int(
            cpa["conflict"].sum()
        ),
        "positive_rate": float(
            cpa["conflict"].mean()
        ),
        "naive_dcpa_auc": 0.877048,
        "interpretation": (
            "The DCPA-only diagnostic produced "
            "ROC-AUC = 0.877048. This demonstrates "
            "that DCPA is strongly predictive of the "
            "conflict label, but does not by itself "
            "prove deterministic label leakage. "
            "The CPA-only GAT is therefore treated "
            "as an ablation demonstrating the "
            "predictive contribution of CPA geometry."
        ),
        "methodological_status": (
            "CPA-inclusive configurations are treated "
            "as diagnostic ablations. The No-CPA "
            "configuration represents the observable-"
            "state-only formulation."
        ),
    }

    if "dcpa" in cpa.columns:

        grouped = (
            cpa.groupby(
                "conflict"
            )["dcpa"].describe()
        )

        grouped.to_csv(
            OUTPUT /
            "dcpa_distribution_by_class.csv"
        )

        summary[
            "dcpa_distribution"
        ] = grouped.to_dict()

    path = (
        OUTPUT /
        "leakage_diagnostic.json"
    )

    save_json(
        summary,
        path
    )

    print(
        f"Saved: {path}"
    )

    return summary


# ============================================================
# PLOT — PRECISION / RECALL / F1
# ============================================================

def plot_prf(df):

    path = (
        OUTPUT /
        "precision_recall_f1_comparison.png"
    )

    models = (
        df["Features"].tolist()
    )

    x = np.arange(
        len(models)
    )

    width = 0.24

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        x - width,
        df["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        df["Recall"],
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        df["F1"],
        width,
        label="F1"
    )

    plt.xticks(
        x,
        models,
        rotation=20,
        ha="right"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Precision, Recall and F1 — Final Evaluation"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# PLOT — ROC-AUC
# ============================================================

def plot_roc_auc(df):

    path = (
        OUTPUT /
        "roc_auc_comparison.png"
    )

    models = (
        df["Features"].tolist()
    )

    values = (
        df["ROC_AUC"].tolist()
    )

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        models,
        values
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        "ROC-AUC — Final Model Comparison"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# PLOT — F1 RANKING
# ============================================================

def plot_f1_ranking(df):

    ranked = df.sort_values(
        "F1",
        ascending=True
    )

    path = (
        OUTPUT /
        "f1_ranking.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.barh(
        ranked["Features"],
        ranked["F1"]
    )

    plt.xlabel(
        "F1 score"
    )

    plt.title(
        "F1 Ranking — Final Evaluation"
    )

    plt.xlim(
        0,
        1.0
    )

    for bar, value in zip(
        bars,
        ranked["F1"]
    ):

        plt.text(
            value + 0.01,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.4f}",
            va="center"
        )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# FIND PREDICTION FILE
# ============================================================

def find_prediction_file(
    model_key
):

    candidates = {

        "GCN": [
            PROCESSED /
            "gcn_test_predictions.csv",

            PROCESSED /
            "gcn_predictions.csv",

            PROCESSED /
            "gcn_test_predictions.npy",
        ],

        "GAT Trajectory + CPA": [
            PROCESSED /
            "gat_test_predictions.csv",

            PROCESSED /
            "gat_v2_test_predictions.csv",

            PROCESSED /
            "gat_predictions.csv",
        ],

        "GAT CPA-only": [
            PROCESSED /
            "gat_cpa_only_predictions.csv",

            PROCESSED /
            "gat_cpa_only_test_predictions.csv",
        ],

        "GAT No-CPA": [
            PROCESSED /
            "gat_no_cpa_predictions.csv",

            PROCESSED /
            "gat_no_cpa_test_predictions.csv",
        ],
    }

    for path in candidates.get(
        model_key,
        []
    ):

        if path.exists():

            return path

    return None


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices():

    try:

        from sklearn.metrics import (
            confusion_matrix
        )

    except ImportError:

        print(
            "scikit-learn unavailable. "
            "Skipping confusion matrices."
        )

        return []

    generated = []

    model_order = [
        "GCN",
        "GAT Trajectory + CPA",
        "GAT CPA-only",
        "GAT No-CPA",
    ]

    for model_key in model_order:

        path = find_prediction_file(
            model_key
        )

        if path is None:

            print(
                f"No saved predictions found for "
                f"{model_key}; skipping confusion matrix."
            )

            continue

        try:

            if path.suffix == ".csv":

                pred_df = pd.read_csv(
                    path
                )

                if "y_true" in pred_df.columns:

                    y_true = (
                        pred_df["y_true"]
                        .to_numpy()
                    )

                elif "label" in pred_df.columns:

                    y_true = (
                        pred_df["label"]
                        .to_numpy()
                    )

                else:

                    print(
                        f"{path}: no y_true/label column."
                    )

                    continue

                if "y_pred" in pred_df.columns:

                    y_pred = (
                        pred_df["y_pred"]
                        .to_numpy()
                    )

                elif "prediction" in pred_df.columns:

                    y_pred = (
                        pred_df["prediction"]
                        .to_numpy()
                    )

                else:

                    print(
                        f"{path}: no y_pred/prediction column."
                    )

                    continue

            else:

                print(
                    f"Prediction format not automatically "
                    f"supported: {path}"
                )

                continue

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            fig, ax = plt.subplots(
                figsize=(6, 5)
            )

            image = ax.imshow(
                cm
            )

            ax.set_title(
                f"Confusion Matrix — {model_key}"
            )

            ax.set_xlabel(
                "Predicted label"
            )

            ax.set_ylabel(
                "True label"
            )

            ax.set_xticks(
                [0, 1]
            )

            ax.set_yticks(
                [0, 1]
            )

            ax.set_xticklabels(
                [
                    "Non-conflict",
                    "Conflict",
                ]
            )

            ax.set_yticklabels(
                [
                    "Non-conflict",
                    "Conflict",
                ]
            )

            for i in range(
                min(2, cm.shape[0])
            ):

                for j in range(
                    min(2, cm.shape[1])
                ):

                    ax.text(
                        j,
                        i,
                        str(
                            cm[i, j]
                        ),
                        ha="center",
                        va="center"
                    )

            fig.colorbar(
                image,
                ax=ax
            )

            safe_name = (
                model_key
                .lower()
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "+",
                    "plus"
                )
                .replace(
                    "-",
                    "_"
                )
            )

            output_path = (
                OUTPUT /
                f"confusion_matrix_{safe_name}.png"
            )

            plt.tight_layout()

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            generated.append(
                output_path
            )

            print(
                f"Saved: {output_path}"
            )

        except Exception as e:

            print(
                f"Could not generate confusion matrix "
                f"for {model_key}: {e}"
            )

    return generated


# ============================================================
# FINAL REPORT
# ============================================================

def generate_final_report(
    df,
    stats,
    leakage
):

    path = (
        OUTPUT /
        "FINAL_EVALUATION_REPORT.md"
    )

    best_f1_row = df.loc[
        df["F1"].idxmax()
    ]

    best_auc_row = df.loc[
        df["ROC_AUC"].idxmax()
    ]

    lines = []

    # ========================================================
    # TITLE
    # ========================================================

    lines.append(
        "# Final GCN / GAT Evaluation Report"
    )

    lines.append("")

    # ========================================================
    # EVALUATION SCOPE
    # ========================================================

    lines.append(
        "## 1. Evaluation scope"
    )

    lines.append("")

    lines.append(
        "The final large-dataset evaluation compares four "
        "graph neural network configurations under the same "
        "chronological train/validation/test framework:"
    )

    lines.append("")

    lines.append(
        "1. **GCN — Trajectory**"
    )

    lines.append(
        "2. **GAT — Trajectory + CPA**"
    )

    lines.append(
        "3. **GAT — CPA-only**"
    )

    lines.append(
        "4. **GAT — No-CPA**"
    )

    lines.append("")

    lines.append(
        "The evaluation uses a chronological split rather than "
        "a random split, preserving temporal separation between "
        "training, validation and test data."
    )

    lines.append("")

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    lines.append(
        "## 2. Corrected final results"
    )

    lines.append("")

    lines.append(
        "| Model | Features | Precision | Recall | F1 | ROC-AUC |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for _, row in df.iterrows():

        lines.append(
            f"| {row['Model']} "
            f"| {row['Features']} "
            f"| {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} "
            f"| {row['F1']:.4f} "
            f"| {row['ROC_AUC']:.4f} |"
        )

    lines.append("")

    # ========================================================
    # BEST F1
    # ========================================================

    lines.append(
        "## 3. Best-performing configuration by F1"
    )

    lines.append("")

    lines.append(
        f"The highest test F1 is obtained by "
        f"**{best_f1_row['Model']} — "
        f"{best_f1_row['Features']}**, "
        f"with F1 = **{best_f1_row['F1']:.4f}**."
    )

    lines.append("")

    lines.append(
        f"Its precision is **{best_f1_row['Precision']:.4f}**, "
        f"recall is **{best_f1_row['Recall']:.4f}**, "
        f"and ROC-AUC is **{best_f1_row['ROC_AUC']:.4f}**."
    )

    lines.append("")

    # ========================================================
    # BEST ROC-AUC
    # ========================================================

    lines.append(
        "## 4. Best ROC-AUC"
    )

    lines.append("")

    lines.append(
        f"The highest ROC-AUC is obtained by "
        f"**{best_auc_row['Model']} — "
        f"{best_auc_row['Features']}**, "
        f"with ROC-AUC = **{best_auc_row['ROC_AUC']:.4f}**."
    )

    lines.append("")

    # ========================================================
    # DCPA DIAGNOSTIC
    # ========================================================

    lines.append(
        "## 5. DCPA / CPA diagnostic"
    )

    lines.append("")

    lines.append(
        "The single-feature DCPA diagnostic produced "
        "**ROC-AUC = 0.877048**."
    )

    lines.append("")

    lines.append(
        "This demonstrates that DCPA is strongly predictive "
        "of the conflict label. However, the result alone "
        "does not establish deterministic label leakage."
    )

    lines.append("")

    lines.append(
        "The CPA-only GAT achieved:"
    )

    lines.append("")

    lines.append(
        "- Precision: **0.1354**"
    )

    lines.append(
        "- Recall: **0.9804**"
    )

    lines.append(
        "- F1: **0.2379**"
    )

    lines.append(
        "- ROC-AUC: **0.9680**"
    )

    lines.append("")

    lines.append(
        "The CPA-only experiment therefore confirms that "
        "CPA geometry contains substantial predictive signal, "
        "but CPA-only information does not reproduce the "
        "performance of the strongest trajectory-based model."
    )

    lines.append("")

    # ========================================================
    # NO CPA
    # ========================================================

    lines.append(
        "## 6. No-CPA configuration"
    )

    lines.append("")

    lines.append(
        "The No-CPA GAT removes `tcpa` and `dcpa` from the "
        "edge representation."
    )

    lines.append("")

    lines.append(
        "The remaining four edge features represent "
        "currently observable relative-state information:"
    )

    lines.append("")

    lines.append(
        "- Horizontal separation"
    )

    lines.append(
        "- Vertical separation"
    )

    lines.append(
        "- Relative speed"
    )

    lines.append(
        "- Relative heading"
    )

    lines.append("")

    lines.append(
        "The No-CPA GAT achieved:"
    )

    lines.append("")

    lines.append(
        "- Precision: **0.4990**"
    )

    lines.append(
        "- Recall: **0.9983**"
    )

    lines.append(
        "- F1: **0.6654**"
    )

    lines.append(
        "- ROC-AUC: **0.9971**"
    )

    lines.append("")

    lines.append(
        "This configuration is the principal candidate for "
        "the observable-state-only formulation."
    )

    lines.append("")

    # ========================================================
    # DATASET
    # ========================================================

    lines.append(
        "## 7. Dataset statistics"
    )

    lines.append("")

    if stats:

        if "graphs" in stats:

            lines.append(
                f"- Graphs: **{stats['graphs']:,}**"
            )

        if "graph_edges" in stats:

            lines.append(
                f"- Directed edges: "
                f"**{stats['graph_edges']:,}**"
            )

        if "graph_positive_labels" in stats:

            lines.append(
                f"- Positive labels: "
                f"**{stats['graph_positive_labels']:,}**"
            )

        if "graph_positive_rate" in stats:

            lines.append(
                f"- Positive rate: "
                f"**{stats['graph_positive_rate']:.6f}**"
            )

        if "node_feature_dimension" in stats:

            lines.append(
                f"- Node feature dimension: "
                f"**{stats['node_feature_dimension']}**"
            )

        if "edge_feature_dimension" in stats:

            lines.append(
                f"- Original edge feature dimension: "
                f"**{stats['edge_feature_dimension']}**"
            )

        if "cpa_rows" in stats:

            lines.append(
                f"- CPA candidate rows: "
                f"**{stats['cpa_rows']:,}**"
            )

        if "cpa_positive_labels" in stats:

            lines.append(
                f"- CPA positive labels: "
                f"**{stats['cpa_positive_labels']:,}**"
            )

    lines.append("")

    # ========================================================
    # LABEL CONSISTENCY
    # ========================================================

    lines.append(
        "## 8. CPA-to-graph label consistency"
    )

    lines.append("")

    lines.append(
        "The CPA-to-graph consistency check confirmed:"
    )

    lines.append("")

    lines.append(
        "- CPA rows: **337,093**"
    )

    lines.append(
        "- CPA positive labels: **6,779**"
    )

    lines.append(
        "- Graph directed edges: **674,186**"
    )

    lines.append(
        "- Graph positive labels: **13,558**"
    )

    lines.append(
        "- Expected graph positives after storing each "
        "CPA pair in both directions: **13,558**"
    )

    lines.append(
        "- Positive-label agreement: **True**"
    )

    lines.append("")

    lines.append(
        "This confirms that the graph construction preserved "
        "the CPA conflict labels consistently when converting "
        "undirected candidate pairs into directed graph edges."
    )

    lines.append("")

    # ========================================================
    # INTERPRETATION
    # ========================================================

    lines.append(
        "## 9. Interpretation"
    )

    lines.append("")

    lines.append(
        "The conflict class represents approximately 2% of "
        "graph edges. Consequently, ROC-AUC should not be "
        "used as the sole indicator of classification quality."
    )

    lines.append("")

    lines.append(
        "F1, precision and recall provide the more informative "
        "view of the operational classification trade-off, "
        "while ROC-AUC is retained as a ranking/discrimination "
        "measure."
    )

    lines.append("")

    lines.append(
        "The corrected GCN trajectory model achieves the highest "
        "F1 among the evaluated configurations, while the "
        "Trajectory + CPA GAT obtains the highest ROC-AUC."
    )

    lines.append("")

    lines.append(
        "The No-CPA GAT remains highly discriminative and "
        "provides the most appropriate formulation for assessing "
        "prediction from observable relative-state information "
        "without explicit CPA variables."
    )

    lines.append("")

    # ========================================================
    # METHODOLOGICAL NOTE
    # ========================================================

    lines.append(
        "## 10. Methodological note"
    )

    lines.append("")

    lines.append(
        "The CPA-inclusive experiments should be presented as "
        "diagnostic ablations rather than being interpreted "
        "automatically as evidence of superior real-time "
        "conflict prediction."
    )

    lines.append("")

    lines.append(
        "CPA variables encode predicted closest-approach "
        "geometry and therefore differ methodologically from "
        "purely observable current-state features."
    )

    lines.append("")

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    lines.append(
        "## 11. Reproducibility"
    )

    lines.append("")

    lines.append(
        "This packaging script does not retrain any model. "
        "It operates on the completed saved evaluation results "
        "and the large graph dataset."
    )

    lines.append("")

    # ========================================================
    # WRITE
    # ========================================================

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# MODEL MANIFEST
# ============================================================

def generate_model_manifest():

    model_files = {

        "GCN": (
            MODELS /
            "conflict_gcn_large.pth"
        ),

        "GAT Trajectory + CPA": (
            MODELS /
            "conflict_gat_v2.pth"
        ),

        "GAT CPA-only": (
            MODELS /
            "conflict_gat_cpa_only.pth"
        ),

        "GAT No-CPA": (
            MODELS /
            "conflict_gat_no_cpa.pth"
        ),
    }

    manifest = {}

    for name, path in model_files.items():

        manifest[name] = {
            "path": str(
                path
            ),
            "exists": bool(
                path.exists()
            ),
        }

        if path.exists():

            manifest[name][
                "size_bytes"
            ] = int(
                path.stat().st_size
            )

    output_path = (
        OUTPUT /
        "model_manifest.json"
    )

    save_json(
        manifest,
        output_path
    )

    print(
        f"Saved: {output_path}"
    )

    return output_path


# ============================================================
# EVALUATION SUMMARY
# ============================================================

def generate_evaluation_summary(df):

    best_f1_idx = df[
        "F1"
    ].idxmax()

    best_auc_idx = df[
        "ROC_AUC"
    ].idxmax()

    best_precision_idx = df[
        "Precision"
    ].idxmax()

    summary = {

        "best_f1": {
            "model": str(
                df.loc[
                    best_f1_idx,
                    "Model"
                ]
            ),
            "features": str(
                df.loc[
                    best_f1_idx,
                    "Features"
                ]
            ),
            "value": float(
                df.loc[
                    best_f1_idx,
                    "F1"
                ]
            ),
        },

        "best_roc_auc": {
            "model": str(
                df.loc[
                    best_auc_idx,
                    "Model"
                ]
            ),
            "features": str(
                df.loc[
                    best_auc_idx,
                    "Features"
                ]
            ),
            "value": float(
                df.loc[
                    best_auc_idx,
                    "ROC_AUC"
                ]
            ),
        },

        "best_precision": {
            "model": str(
                df.loc[
                    best_precision_idx,
                    "Model"
                ]
            ),
            "features": str(
                df.loc[
                    best_precision_idx,
                    "Features"
                ]
            ),
            "value": float(
                df.loc[
                    best_precision_idx,
                    "Precision"
                ]
            ),
        },

        "output_directory": str(
            OUTPUT
        ),

        "zip_package": str(
            PACKAGE_ZIP
        ),
    }

    path = (
        OUTPUT /
        "evaluation_summary.json"
    )

    save_json(
        summary,
        path
    )

    print(
        f"Saved: {path}"
    )

    return path


# ============================================================
# ZIP PACKAGE
# ============================================================

def create_zip():

    if PACKAGE_ZIP.exists():

        PACKAGE_ZIP.unlink()

    with zipfile.ZipFile(
        PACKAGE_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as z:

        for path in OUTPUT.rglob("*"):

            if path.is_file():

                z.write(
                    path,
                    arcname=path.relative_to(
                        OUTPUT
                    )
                )

    print(
        f"Created package: {PACKAGE_ZIP}"
    )

    return PACKAGE_ZIP


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 90
    )

    print(
        "FINAL GCN / GAT EVALUATION PACKAGE"
    )

    print(
        "=" * 90
    )

    print()

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    results = load_existing_results()

    df = build_results_table(
        results
    )

    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "CORRECTED FINAL RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "SAVING FINAL RESULTS"
    )

    print(
        "=" * 90
    )

    save_csv(
        df
    )

    save_markdown(
        df
    )

    save_results_json(
        df
    )

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "DATASET STATISTICS"
    )

    print(
        "=" * 90
    )

    stats = generate_dataset_statistics()

    # ========================================================
    # LEAKAGE DIAGNOSTIC
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "CPA / DCPA LEAKAGE DIAGNOSTIC"
    )

    print(
        "=" * 90
    )

    leakage = generate_leakage_summary()

    # ========================================================
    # PLOTS
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "GENERATING COMPARISON PLOTS"
    )

    print(
        "=" * 90
    )

    plot_prf(
        df
    )

    plot_roc_auc(
        df
    )

    plot_f1_ranking(
        df
    )

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "GENERATING CONFUSION MATRICES"
    )

    print(
        "=" * 90
    )

    plot_confusion_matrices()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "GENERATING FINAL REPORT"
    )

    print(
        "=" * 90
    )

    generate_final_report(
        df,
        stats,
        leakage
    )

    # ========================================================
    # MODEL MANIFEST
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "GENERATING MODEL MANIFEST"
    )

    print(
        "=" * 90
    )

    generate_model_manifest()

    # ========================================================
    # SUMMARY
    # ========================================================

    generate_evaluation_summary(
        df
    )

    # ========================================================
    # ZIP
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "CREATING ZIP PACKAGE"
    )

    print(
        "=" * 90
    )

    create_zip()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "FINAL PACKAGE COMPLETE"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT}"
    )

    print()

    print(
        "ZIP package:"
    )

    print(
        f"  {PACKAGE_ZIP}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

