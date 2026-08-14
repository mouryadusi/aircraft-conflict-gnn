import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("processed_data/ablation_results.csv")

metrics = ["precision", "recall", "f1", "roc_auc"]

for metric in metrics:

    plt.figure(figsize=(7,4))

    plt.bar(df["feature_set"], df[metric])

    plt.ylabel(metric.upper())

    plt.title(f"{metric.upper()} Comparison")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(f"processed_data/{metric}_comparison.png", dpi=300)

    plt.close()

print("Done.")