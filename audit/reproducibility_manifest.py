from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = [
    "processed_data/candidate_pairs_no_leakage.csv",
    "processed_data/graphs_large.pt",
    "processed_data/train_graphs.pt",
    "processed_data/val_graphs.pt",
    "processed_data/test_graphs.pt",
    "models/conflict_gat_no_cpa.pth",
]

manifest = {
    "python": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "files": {},
}

for rel in files:
    p = ROOT / rel
    if p.exists():
        manifest["files"][rel] = {
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }

try:
    manifest["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
except Exception:
    manifest["git_commit"] = None

out = ROOT / "research_artifacts" / "reproducibility_manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
