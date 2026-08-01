"""
Script 2: Build aircraft pairs, forward-looking conflict labels, and graph snapshots.
Run: python scripts/02_build_graphs.py
Input: processed_data/region_final.pkl
Output: processed_data/graphs_large.pt, processed_data/labelled_large.csv
"""
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from torch_geometric.data import Data
import os

os.makedirs("processed_data", exist_ok=True)

region_final = pd.read_pickle("processed_data/region_final.pkl")
print("Loaded region_final:", region_final.shape)


def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R_nm * np.arcsin(np.sqrt(a))


def bearing_diff(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)


def will_conflict_within(df, horizon_s=300):
    df = df.copy()
    df["conflict_now"] = (df["h_dist_nm"] < 5.0) & (df["v_dist_ft"] < 1000.0)
    df["label_conflict"] = False
    for (_, _, _), group in df.groupby(["icao24_1", "icao24_2", "source_day"]):
        idx = group.index.to_list()
        times = group["time"].to_numpy()
        conflicts = group["conflict_now"].to_numpy()
        for i in range(len(group)):
            future = (times >= times[i]) & (times <= times[i] + horizon_s)
            if conflicts[future].any():
                df.loc[idx[i], "label_conflict"] = True
    return df


# --- Build pairs_df ---
pair_records = []
for (t, day), group in region_final.groupby(["time", "source_day"]):
    aircraft = group.set_index("icao24")
    ids = aircraft.index.tolist()
    for id1, id2 in combinations(ids, 2):
        a1, a2 = aircraft.loc[id1], aircraft.loc[id2]
        h_dist = haversine_nm(a1["lat"], a1["lon"], a2["lat"], a2["lon"])
        v_dist = abs(a1["baroaltitude"] - a2["baroaltitude"]) * 3.28084
        pair_records.append({
            "time": t, "source_day": day, "icao24_1": id1, "icao24_2": id2,
            "h_dist_nm": h_dist, "v_dist_ft": v_dist,
            "heading_1": a1["heading"], "heading_2": a2["heading"]
        })
pairs_df = pd.DataFrame(pair_records)

pairs_df[["icao24_1", "icao24_2"]] = np.sort(pairs_df[["icao24_1", "icao24_2"]], axis=1)
pairs_df = pairs_df.sort_values(["icao24_1", "icao24_2", "source_day", "time"]).reset_index(drop=True)

pairs_df["h_dist_prev"] = pairs_df.groupby(["icao24_1", "icao24_2", "source_day"])["h_dist_nm"].shift(1)
pairs_df["time_prev"] = pairs_df.groupby(["icao24_1", "icao24_2", "source_day"])["time"].shift(1)
pairs_df["closing_rate_nm_s"] = (pairs_df["h_dist_prev"] - pairs_df["h_dist_nm"]) / (pairs_df["time"] - pairs_df["time_prev"])

dt = pairs_df["time"] - pairs_df["time_prev"]
pairs_df.loc[dt <= 0, "closing_rate_nm_s"] = np.nan
pairs_df.loc[pairs_df["closing_rate_nm_s"].abs() > 5, "closing_rate_nm_s"] = np.nan

pairs_df["bearing_diff"] = pairs_df.apply(
    lambda r: bearing_diff(r["heading_1"], r["heading_2"]) if pd.notna(r["heading_1"]) and pd.notna(r["heading_2"]) else np.nan,
    axis=1
)
pairs_df["time_to_cpa_s"] = np.where(
    pairs_df["closing_rate_nm_s"] > 0,
    pairs_df["h_dist_nm"] / pairs_df["closing_rate_nm_s"],
    np.nan
)
print("pairs_df:", pairs_df.shape)

# --- Label ---
labelled = will_conflict_within(pairs_df, horizon_s=300)
print("Positive labels:", labelled["label_conflict"].sum())
labelled.to_csv("processed_data/labelled_large.csv", index=False)

# --- Build graphs ---
node_feature_cols = ["lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]
edge_feature_cols = ["h_dist_nm", "v_dist_ft", "closing_rate_nm_s", "bearing_diff", "time_to_cpa_s"]

graphs = []
for (t, day), _ in region_final.groupby(["time", "source_day"]):
    nodes = region_final[(region_final["time"] == t) & (region_final["source_day"] == day)].copy()
    edges = labelled[(labelled["time"] == t) & (labelled["source_day"] == day)].copy()
    if len(nodes) < 2 or len(edges) == 0:
        continue
    node_map = {icao: idx for idx, icao in enumerate(nodes["icao24"])}
    edges = edges[edges["icao24_1"].isin(node_map) & edges["icao24_2"].isin(node_map)]
    if len(edges) == 0:
        continue
    edge_index = torch.tensor([
        [node_map[a] for a in edges["icao24_1"]],
        [node_map[b] for b in edges["icao24_2"]],
    ], dtype=torch.long)
    x = torch.tensor(nodes[node_feature_cols].fillna(0).values, dtype=torch.float)
    edge_attr = torch.tensor(edges[edge_feature_cols].fillna(0).values, dtype=torch.float)
    y = torch.tensor(edges["label_conflict"].astype(int).values, dtype=torch.long)
    graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, time=t, source_day=day))

print(f"Graph snapshots: {len(graphs)}")

torch.save(graphs, "processed_data/graphs_large.pt")
print("\nSaved processed_data/graphs_large.pt and processed_data/labelled_large.csv")
print("DONE: 02_build_graphs.py")
