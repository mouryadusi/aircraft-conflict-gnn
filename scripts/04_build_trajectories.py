import pandas as pd
import numpy as np

print("=" * 70)
print("BUILDING AIRCRAFT TRAJECTORIES")
print("=" * 70)

df = pd.read_pickle("processed_data/state_vectors_clean.pkl")

print("Loaded:", df.shape)

df = df.sort_values(
    ["icao24", "time"]
).reset_index(drop=True)
df["dt"] = (
    df.groupby("icao24")["time"]
      .diff()
      .fillna(0)
)

print(df["dt"].describe())
MAX_GAP = 900

df["new_track"] = (
    df["dt"] > MAX_GAP
).astype(int)

df["trajectory_id"] = (
    df.groupby("icao24")["new_track"]
      .cumsum()
)

df["trajectory_id"] = (
    df["icao24"]
    + "_"
    + df["trajectory_id"].astype(str)
)
counts = (
    df.groupby("trajectory_id")
      .size()
)

valid_tracks = counts[counts >= 20].index

df = df[
    df["trajectory_id"].isin(valid_tracks)
]

print("Valid trajectories:", len(valid_tracks))
df = df.sort_values(
    ["trajectory_id", "time"]
).reset_index(drop=True)
print("=" * 70)

print("Trajectory statistics")

print("=" * 70)

print("Rows:", len(df))

print("Aircraft:", df["icao24"].nunique())

print("Trajectories:", df["trajectory_id"].nunique())

sizes = (
    df.groupby("trajectory_id")
      .size()
)

print()

print(sizes.describe())
df.to_pickle(
    "processed_data/trajectories.pkl"
)

df.to_csv(
    "processed_data/trajectories.csv",
    index=False,
)

print()

print("Saved:")

print("processed_data/trajectories.pkl")

print("processed_data/trajectories.csv")

print()

print("DONE: 04_build_trajectories.py")