import os
import torch
import pandas as pd
from torch_geometric.data import Data

os.makedirs("processed_data", exist_ok=True)

print("=" * 70)
print("BUILDING CPA GRAPH SNAPSHOTS")
print("=" * 70)

pairs = pd.read_pickle("processed_data/candidate_pairs_cpa.pkl")

print("Candidate pairs:", len(pairs))

graphs = []

times = sorted(pairs["time"].unique())

print("Unique timestamps:", len(times))

for i, t in enumerate(times):

    if i % 1000 == 0:
        print(f"{i}/{len(times)}")

    snap = pairs[pairs.time == t]

    aircraft = sorted(
        set(snap["icao24_1"]).union(
            set(snap["icao24_2"])
        )
    )

    mapping = {
        a: idx
        for idx, a in enumerate(aircraft)
    }

    node_dict = {}

    for _, r in snap.iterrows():

        if r["icao24_1"] not in node_dict:

            node_dict[r["icao24_1"]] = [

                r["lat1"],
                r["lon1"],
                r["alt1"],
                r["velocity1"],
                r["heading1"],
                0.0,

            ]

        if r["icao24_2"] not in node_dict:

            node_dict[r["icao24_2"]] = [

                r["lat2"],
                r["lon2"],
                r["alt2"],
                r["velocity2"],
                r["heading2"],
                0.0,

            ]

    x = torch.tensor(

        [node_dict[a] for a in aircraft],

        dtype=torch.float,

    )

    edge_index = []

    edge_attr = []

    labels = []

    for _, r in snap.iterrows():

        s = mapping[r["icao24_1"]]
        d = mapping[r["icao24_2"]]

        feat = [

            r["distance_now"],
            r["vertical_distance"],
            r["relative_speed"],
            r["relative_heading"],
            r["tcpa"],
            r["dcpa"],

        ]

        label = int(r["conflict"])

        edge_index.append([s, d])
        edge_index.append([d, s])

        edge_attr.append(feat)
        edge_attr.append(feat)

        labels.append(label)
        labels.append(label)

    edge_index = torch.tensor(

        edge_index,

        dtype=torch.long,

    ).t().contiguous()

    edge_attr = torch.tensor(

        edge_attr,

        dtype=torch.float,

    )

    y = torch.tensor(

        labels,

        dtype=torch.long,

    )

    g = Data(

        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,

    )

    g.time = int(t)

    graphs.append(g)

torch.save(

    graphs,

    "processed_data/graphs_large.pt",

)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print("Graphs built:", len(graphs))
print("Saved processed_data/graphs_large.pt")