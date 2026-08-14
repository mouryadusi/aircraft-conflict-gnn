"""
Script 2 (NEW)

Build graph snapshots directly from the CPA-labelled dataset.

Input:
    processed_data/candidate_pairs_cpa.pkl

Output:
    processed_data/graphs_large.pt
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

os.makedirs("processed_data", exist_ok=True)

print("=" * 70)
print("BUILDING GRAPH SNAPSHOTS")
print("=" * 70)

pairs = pd.read_pickle("processed_data/candidate_pairs_cpa.pkl")

print("Candidate pairs:", len(pairs))

graphs = []

times = sorted(pairs.time.unique())

print("Unique timestamps:", len(times))

for idx, t in enumerate(times):

    if idx % 1000 == 0:
        print(f"{idx}/{len(times)}")

    snap = pairs[pairs.time == t]

    aircraft = sorted(
        set(snap.icao24_1).union(set(snap.icao24_2))
    )

    mapping = {
        a: i
        for i, a in enumerate(aircraft)
    }

    node_features = []

    for a in aircraft:

        r = snap[
            (snap.icao24_1 == a) |
            (snap.icao24_2 == a)
        ].iloc[0]

        if r.icao24_1 == a:

            node_features.append([
                r.lat1,
                r.lon1,
                r.alt1,
                r.velocity1,
                r.heading1,
                0.0,
            ])

        else:

            node_features.append([
                r.lat2,
                r.lon2,
                r.alt2,
                r.velocity2,
                r.heading2,
                0.0,
            ])

    x = torch.tensor(
        node_features,
        dtype=torch.float,
    )

    edge_index = []

    edge_attr = []

    labels = []

    for _, r in snap.iterrows():

        s = mapping[r.icao24_1]
        d = mapping[r.icao24_2]

        edge_index.append([s, d])
        edge_index.append([d, s])

        feat = [

            r.distance_now,

            r.vertical_distance,

            r.relative_speed,

            r.relative_heading,

            r.tcpa,

            r.dcpa,

        ]

        edge_attr.append(feat)
        edge_attr.append(feat)

        labels.append(int(r.conflict))
        labels.append(int(r.conflict))

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long,
    ).t()

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

print("Graphs:", len(graphs))
print("Saved processed_data/graphs_large.pt")