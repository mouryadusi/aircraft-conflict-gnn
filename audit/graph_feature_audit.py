from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parents[1]

files = {
    "train": ROOT / "processed_data/train_graphs.pt",
    "validation": ROOT / "processed_data/val_graphs.pt",
    "test": ROOT / "processed_data/test_graphs.pt",
}

report = {}

for name, path in files.items():

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    g = obj[0] if isinstance(obj, list) else obj

    entry = {
        "type": type(g).__name__,
        "attributes": list(g.keys()) if hasattr(g, "keys") else []
    }

    if hasattr(g, "x"):
        entry["node_feature_shape"] = list(g.x.shape)

    if hasattr(g, "edge_attr"):
        entry["edge_feature_shape"] = list(g.edge_attr.shape)

    if hasattr(g, "y"):
        entry["label_shape"] = list(g.y.shape)

    report[name] = entry


out = ROOT / "research_artifacts/graph_feature_audit.json"

out.write_text(
    json.dumps(report, indent=2)
)

print(json.dumps(report, indent=2))
