import json
import pandas as pd
import matplotlib.pyplot as plt

results = [
    ("GCN", "processed_data/gcn_results.json"),
    ("GAT Baseline", "processed_data/gat_baseline_seeded_results.json"),
    ("GAT + Phase", "processed_data/gat_phase_results.json"),
    ("GAT + Wake", "processed_data/gat_wake_results.json"),
]

rows = []

for name, file in results:
    with open(file) as f:
        d = json.load(f)

    rows.append({
        "Model": name,
        "Precision": d["precision"],
        "Recall": d["recall"],
        "F1": d["f1"],
        "ROC_AUC": d["roc_auc"],
    })

df = pd.DataFrame(rows)

plt.figure(figsize=(8,5))
df.plot(
    x="Model",
    y=["Precision","Recall","F1","ROC_AUC"],
    kind="bar",
)
plt.tight_layout()
plt.savefig("processed_data/model_comparison.png", dpi=300)

print(df)
print("\nSaved processed_data/model_comparison.png")
