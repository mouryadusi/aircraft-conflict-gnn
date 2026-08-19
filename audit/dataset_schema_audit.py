from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "processed_data/state_vectors_clean.csv",
    "processed_data/trajectories.csv",
    "processed_data/candidate_pairs_no_leakage.csv",
    "processed_data/candidate_pairs.csv",
]

report = {}

for rel in FILES:
    p = ROOT / rel
    if not p.exists():
        report[rel] = {"exists": False}
        continue

    df = pd.read_csv(p, nrows=1000)

    report[rel] = {
        "exists": True,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "null_counts_first_1000": df.isna().sum().to_dict(),
        "rows_sampled": len(df),
    }

out = ROOT / "research_artifacts" / "dataset_schema_audit.json"
out.write_text(json.dumps(report, indent=2, default=str))
print(out)
