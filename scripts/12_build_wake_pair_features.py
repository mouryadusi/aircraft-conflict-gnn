import pandas as pd
import numpy as np

print("=" * 60)
print("BUILDING WAKE PAIR FEATURES")
print("=" * 60)

region = pd.read_pickle("processed_data/region_final_wake.pkl")

pairs = pd.read_pickle("processed_data/pairs_df.pkl")

print("Region:", region.shape)
print("Pairs :", pairs.shape)

lookup = region[
    ["icao24", "time", "wake_category"]
].copy()

lookup = lookup.rename(
    columns={
        "icao24": "icao24_1",
        "wake_category": "wake_1"
    }
)

pairs = pairs.merge(
    lookup,
    on=["icao24_1", "time"],
    how="left"
)

lookup = lookup.rename(
    columns={
        "icao24_1": "icao24_2",
        "wake_1": "wake_2"
    }
)

pairs = pairs.merge(
    lookup,
    on=["icao24_2", "time"],
    how="left"
)

pairs["same_wake"] = (
    pairs["wake_1"] == pairs["wake_2"]
).astype(int)

pairs["heavy_following_light"] = (
    (pairs["wake_1"] == "Heavy") &
    (pairs["wake_2"] == "Medium")
).astype(int)

pairs["super_following_light"] = (
    (pairs["wake_1"] == "Super") &
    (pairs["wake_2"] == "Medium")
).astype(int)

print("\nWake Pair Statistics\n")

print(pairs[
    [
        "same_wake",
        "heavy_following_light",
        "super_following_light"
    ]
].sum())

pairs.to_pickle(
    "processed_data/pairs_df_wake.pkl"
)

print("\nSaved processed_data/pairs_df_wake.pkl")