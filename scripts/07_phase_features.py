import random
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from torch_geometric.data import Data

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

region_final = pd.read_pickle("processed_data/region_final.pkl")
print("Loaded region_final:", region_final.shape)

region_final["vertrate_fpm"] = region_final["vertrate"] * 196.850394

PHASE_CLIMB_THRESHOLD = 300
PHASE_DESCENT_THRESHOLD = -300


def classify_phase(vfpm):
    if pd.isna(vfpm):
        return "Unknown"
    if vfpm > PHASE_CLIMB_THRESHOLD:
        return "Climb"
    elif vfpm < PHASE_DESCENT_THRESHOLD:
        return "Descent"
    else:
        return "Cruise"


region_final["flight_phase"] = region_final["vertrate_fpm"].apply(classify_phase)

print("\nPhase distribution:")
print(region_final["flight_phase"].value_counts())

print("\nNOTE: Ground/Takeoff/Approach phases are structurally absent from this dataset")
print("due to the existing altitude filter (baroaltitude > 1000m), which restricts")
print("observations to en-route flight. This is a documented scope constraint.")


def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R_nm * np.arcsin(np.sqrt(a))


def bearing_diff(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)


def will_conflict_within(df, horizon_s=300):

    df = df.copy()

    df["conflict_now"] = (
        (df["h_dist_nm"] < 5.0)
        &
        (df["v_dist_ft"] < 1000.0)
    )

    df["label_conflict"] = False

    for (_, _, _), group in df.groupby(
        ["icao24_1", "icao24_2", "source_day"]
    ):

        idx = group.index.to_list()
        times = group["time"].to_numpy()
        conflicts = group["conflict_now"].to_numpy()

        for i in range(len(group)):

            future = (
                (times >= times[i])
                &
                (times <= times[i] + horizon_s)
            )

            if conflicts[future].any():
                df.loc[idx[i], "label_conflict"] = True

    return df


pair_records = []

for (t, day), group in region_final.groupby(
    ["time", "source_day"]
):

    aircraft = group.set_index("icao24")

    ids = aircraft.index.tolist()

    for id1, id2 in combinations(ids, 2):

        a1 = aircraft.loc[id1]
        a2 = aircraft.loc[id2]

        h_dist = haversine_nm(
            a1["lat"],
            a1["lon"],
            a2["lat"],
            a2["lon"],
        )

        v_dist = abs(
            a1["baroaltitude"]
            - a2["baroaltitude"]
        ) * 3.28084

        pair_records.append(
            {
                "time": t,
                "source_day": day,
                "icao24_1": id1,
                "icao24_2": id2,
                "h_dist_nm": h_dist,
                "v_dist_ft": v_dist,
                "heading_1": a1["heading"],
                "heading_2": a2["heading"],
                "phase_1": a1["flight_phase"],
                "phase_2": a2["flight_phase"],
            }
        )

pairs_df = pd.DataFrame(pair_records)

pairs_df[["icao24_1", "icao24_2"]] = np.sort(
    pairs_df[["icao24_1", "icao24_2"]],
    axis=1,
)

pairs_df = pairs_df.sort_values(
    [
        "icao24_1",
        "icao24_2",
        "source_day",
        "time",
    ]
).reset_index(drop=True)

pairs_df["h_dist_prev"] = pairs_df.groupby(
    [
        "icao24_1",
        "icao24_2",
        "source_day",
    ]
)["h_dist_nm"].shift(1)

pairs_df["time_prev"] = pairs_df.groupby(
    [
        "icao24_1",
        "icao24_2",
        "source_day",
    ]
)["time"].shift(1)

pairs_df["closing_rate_nm_s"] = (
    (
        pairs_df["h_dist_prev"]
        - pairs_df["h_dist_nm"]
    )
    /
    (
        pairs_df["time"]
        - pairs_df["time_prev"]
    )
)

dt = pairs_df["time"] - pairs_df["time_prev"]

pairs_df.loc[
    dt <= 0,
    "closing_rate_nm_s",
] = np.nan

pairs_df.loc[
    pairs_df["closing_rate_nm_s"].abs() > 5,
    "closing_rate_nm_s",
] = np.nan

pairs_df["bearing_diff"] = pairs_df.apply(
    lambda r:
    bearing_diff(
        r["heading_1"],
        r["heading_2"],
    )
    if pd.notna(r["heading_1"])
    and pd.notna(r["heading_2"])
    else np.nan,
    axis=1,
)

pairs_df["time_to_cpa_s"] = np.where(
    pairs_df["closing_rate_nm_s"] > 0,
    pairs_df["h_dist_nm"]
    /
    pairs_df["closing_rate_nm_s"],
    np.nan,
)

pairs_df["same_phase"] = (
    pairs_df["phase_1"]
    ==
    pairs_df["phase_2"]
).astype(int)

PHASE_MAP = {
    "Climb": 0,
    "Cruise": 1,
    "Descent": 2,
    "Unknown": -1,
}

pairs_df["phase_1_code"] = pairs_df["phase_1"].map(PHASE_MAP)
pairs_df["phase_2_code"] = pairs_df["phase_2"].map(PHASE_MAP)

print("\npairs_df with phase features:", pairs_df.shape)
print(pairs_df["same_phase"].value_counts())

labelled = will_conflict_within(
    pairs_df,
    horizon_s=300,
)

print("\nPositive labels:", labelled["label_conflict"].sum())

# Save labelled CSV
labelled.to_csv(
    "processed_data/labelled_phase.csv",
    index=False,
)

# NEW: Save pair dataframe for wake feature engineering
labelled.to_pickle(
    "processed_data/pairs_df.pkl"
)

print("Saved processed_data/pairs_df.pkl")

node_feature_cols_base = [
    "lat",
    "lon",
    "baroaltitude",
    "velocity",
    "heading",
    "vertrate",
]

region_final["phase_climb"] = (
    region_final["flight_phase"] == "Climb"
).astype(float)

region_final["phase_cruise"] = (
    region_final["flight_phase"] == "Cruise"
).astype(float)

region_final["phase_descent"] = (
    region_final["flight_phase"] == "Descent"
).astype(float)

node_feature_cols = (
    node_feature_cols_base
    +
    [
        "phase_climb",
        "phase_cruise",
        "phase_descent",
    ]
)

edge_feature_cols = [
    "h_dist_nm",
    "v_dist_ft",
    "closing_rate_nm_s",
    "bearing_diff",
    "time_to_cpa_s",
    "same_phase",
]

graphs = []

for (t, day), _ in region_final.groupby(
    ["time", "source_day"]
):

    nodes = region_final[
        (region_final["time"] == t)
        &
        (region_final["source_day"] == day)
    ].copy()

    edges = labelled[
        (labelled["time"] == t)
        &
        (labelled["source_day"] == day)
    ].copy()

    if len(nodes) < 2 or len(edges) == 0:
        continue

    node_map = {
        icao: idx
        for idx, icao in enumerate(nodes["icao24"])
    }

    edges = edges[
        edges["icao24_1"].isin(node_map)
        &
        edges["icao24_2"].isin(node_map)
    ]

    if len(edges) == 0:
        continue

    edge_index = torch.tensor(
        [
            [
                node_map[a]
                for a in edges["icao24_1"]
            ],
            [
                node_map[b]
                for b in edges["icao24_2"]
            ],
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

    y = torch.tensor(
        edges["label_conflict"]
        .astype(int)
        .values,
        dtype=torch.long,
    )

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

print(f"\nGraph snapshots (phase-extended): {len(graphs)}")
print("Node feature dim:", len(node_feature_cols))
print("Edge feature dim:", len(edge_feature_cols))
print(graphs[0])

torch.save(
    graphs,
    "processed_data/graphs_phase.pt",
)

print("\nSaved processed_data/graphs_phase.pt")
print("DONE: 07_phase_features.py")