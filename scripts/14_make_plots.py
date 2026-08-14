import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("processed_data/final_results_table.csv")

metrics = ["Precision", "Recall", "F1", "ROC_AUC"]

for metric in metrics:

    plt.figure(figsize=(6,4))

    plt.bar(df["Model"] + "\n" + df["Features"], df[metric])

    plt.ylabel(metric)

    plt.title(metric + " Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(f"processed_data/{metric.lower()}_comparison.png", dpi=300)

    plt.close()

print("Plots saved.")
