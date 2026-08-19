from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "processed_data/candidate_pairs_no_leakage.csv"

df = pd.read_csv(p)

result = {
    "file": str(p),
    "rows": len(df),
    "columns": df.columns.tolist(),
}

if "y" in df:
    label = "y"
elif "label" in df:
    label = "label"
elif "conflict" in df:
    label = "conflict"
else:
    label = None

result["label_column"] = label

for feature in ["tcpa", "dcpa"]:
    if feature in df and label:
        x = pd.to_numeric(df[feature], errors="coerce")
        y = pd.to_numeric(df[label], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() and y[mask].nunique() == 2:
            result[feature] = {
                "min": float(x[mask].min()),
                "max": float(x[mask].max()),
                "positive_mean": float(x[mask & (y == 1)].mean()),
                "negative_mean": float(x[mask & (y == 0)].mean()),
            }

out = ROOT / "research_artifacts" / "cpa_label_diagnostic.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
