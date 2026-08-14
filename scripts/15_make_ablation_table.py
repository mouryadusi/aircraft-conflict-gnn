import json
import pandas as pd

files = [
    "processed_data/gat_baseline_seeded_results.json",
    "processed_data/gat_phase_results.json",
    "processed_data/gat_wake_results.json",
]

rows = []

for f in files:
    with open(f) as fp:
        rows.append(json.load(fp))

df = pd.DataFrame(rows)

cols = [
    "feature_set",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "best_val_f1",
]

df = df[cols]

print(df)

df.to_csv(
    "processed_data/ablation_results.csv",
    index=False,
)

print("\nSaved processed_data/ablation_results.csv")