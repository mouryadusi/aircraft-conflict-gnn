import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("BUILDING WAKE GRAPHS")
print("=" * 60)

print("\nLoading datasets...")

region = pd.read_pickle("processed_data/region_final_wake.pkl")
pairs = pd.read_pickle("processed_data/pairs_df_wake.pkl")

print("Region:", region.shape)
print("Pairs :", pairs.shape)

# -----------------------------
# Phase one-hot encoding
# -----------------------------# ----------------------------------------------------
# Recreate flight phase
# ----------------------------------------------------

region["vertrate_fpm"] = region["vertrate"] * 196.850394

def classify_phase(vfpm):
    if pd.isna(vfpm):
        return "Unknown"
    elif vfpm > 300:
        return "Climb"
    elif vfpm < -300:
        return "Descent"
    else:
        return "Cruise"

region["flight_phase"] = region["vertrate_fpm"].apply(classify_phase)

region["phase_climb"] = (region["flight_phase"] == "Climb").astype(float)
region["phase_cruise"] = (region["flight_phase"] == "Cruise").astype(float)
region["phase_descent"] = (region["flight_phase"] == "Descent").astype(float)
# -----------------------------
# Wake one-hot encoding
# -----------------------------
region["wake_light"] = (region["wake_category"] == "Light").astype(float)
region["wake_medium"] = (region["wake_category"] == "Medium").astype(float)
region["wake_heavy"] = (region["wake_category"] == "Heavy").astype(float)
region["wake_super"] = (region["wake_category"] == "Super").astype(float)

node_feature_cols = [
    "lat",
    "lon",
    "baroaltitude",
    "velocity",
    "heading",
    "vertrate",

    "phase_climb",
    "phase_cruise",
    "phase_descent",

    "wake_light",
    "wake_medium",
    "wake_heavy",
    "wake_super",
]

edge_feature_cols = [
    "h_dist_nm",
    "v_dist_ft",
    "closing_rate_nm_s",
    "bearing_diff",
    "time_to_cpa_s",

    "same_phase",

    "same_wake",
    "heavy_following_light",
    "super_following_light",
]

graphs = []
# ------------------------------------------------------------
# Build graph snapshots
# ------------------------------------------------------------

for (t, day), _ in region.groupby(["time", "source_day"]):

    nodes = region[
        (region["time"] == t) &
        (region["source_day"] == day)
    ].copy()

    edges = pairs[
        (pairs["time"] == t) &
        (pairs["source_day"] == day)
    ].copy()

    if len(nodes) < 2:
        continue

    if len(edges) == 0:
        continue

    node_map = {
        icao: idx
        for idx, icao in enumerate(nodes["icao24"])
    }

    edges = edges[
        edges["icao24_1"].isin(node_map) &
        edges["icao24_2"].isin(node_map)
    ]

    if len(edges) == 0:
        continue

    edge_index = torch.tensor(
        [
            [node_map[a] for a in edges["icao24_1"]],
            [node_map[b] for b in edges["icao24_2"]],
        ],
        dtype=torch.long,
    )

    x = torch.tensor(
        nodes[node_feature_cols]
        .fillna(0)
        .values,
        dtype=torch.float,
    )

    edge_attr = torch.tensor(
        edges[edge_feature_cols]
        .fillna(0)
        .values,
        dtype=torch.float,
    )

    if "label_conflict" in edges.columns:
        y = torch.tensor(
            edges["label_conflict"].astype(int).values,
            dtype=torch.long,
        )
    else:
        y = torch.zeros(len(edges), dtype=torch.long)

    graphs.append(
        Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            time=t,
            source_day=day,
        )
    )

print("\n============================================================")
print("WAKE GRAPH SUMMARY")
print("============================================================")

print(f"Graphs created : {len(graphs)}")
print(f"Node features  : {len(node_feature_cols)}")
print(f"Edge features  : {len(edge_feature_cols)}")

print("\nExample graph:")
print(graphs[0])

torch.save(
    graphs,
    "processed_data/graphs_wake.pt",
)

print("\nSaved:")
print("processed_data/graphs_wake.pt")

print("\nDONE: 13_build_wake_graphs.py")
