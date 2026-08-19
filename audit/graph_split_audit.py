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

    graphs = obj if isinstance(obj, list) else [obj]

    times = []
    edge_counts = []
    node_counts = []
    positive_edges = []

    for g in graphs:

        if hasattr(g, "time"):
            try:
                times.append(int(g.time))
            except:
                pass

        if hasattr(g, "edge_index"):
            edge_counts.append(
                int(g.edge_index.shape[1])
            )

        if hasattr(g, "x"):
            node_counts.append(
                int(g.x.shape[0])
            )

        if hasattr(g, "y"):
            try:
                positive_edges.append(
                    int((g.y == 1).sum())
                )
            except:
                pass


    report[name] = {

        "graphs": len(graphs),

        "time_min":
            min(times) if times else None,

        "time_max":
            max(times) if times else None,

        "total_edges":
            sum(edge_counts),

        "total_nodes":
            sum(node_counts),

        "positive_edges":
            sum(positive_edges),

        "positive_rate":
            (
                sum(positive_edges)
                /
                sum(edge_counts)
            )
            if sum(edge_counts)>0
            else None
    }


out = ROOT / "research_artifacts/graph_split_audit.json"

out.write_text(
    json.dumps(report, indent=2)
)

print(json.dumps(report, indent=2))
