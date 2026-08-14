import pandas as pd

print("=" * 70)
print("JOINING WEATHER WITH CANDIDATE PAIRS")
print("=" * 70)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------

pairs = pd.read_pickle(
    "processed_data/candidate_pairs_cpa.pkl"
)

weather = pd.read_pickle(
    "processed_data/weather_processed.pkl"
)

print(f"Candidate pairs : {len(pairs):,}")
print(f"Weather records : {len(weather):,}")

# -------------------------------------------------------
# Round coordinates to ERA5 grid (0.25°)
# -------------------------------------------------------

pairs["lat_grid"] = (
    (pairs["lat1"] / 0.25).round() * 0.25
)

pairs["lon_grid"] = (
    (pairs["lon1"] / 0.25).round() * 0.25
)

weather["lat_grid"] = weather["lat"]
weather["lon_grid"] = weather["lon"]

# -------------------------------------------------------
# Convert timestamps
# -------------------------------------------------------

# -------------------------------------------------------
# Convert timestamps
# -------------------------------------------------------

pairs["time"] = pd.to_datetime(
    pairs["time"],
    unit="s"
).dt.floor("h")

weather["time"] = pd.to_datetime(
    weather["time"]
).dt.floor("h")
# -------------------------------------------------------
# DEBUG
# -------------------------------------------------------

print("\nPair key dtypes")
print(pairs[["time", "lat_grid", "lon_grid"]].dtypes)

print("\nWeather key dtypes")
print(weather[["time", "lat_grid", "lon_grid"]].dtypes)

print("\nFirst aircraft keys")
print(
    pairs[
        ["time", "lat_grid", "lon_grid"]
    ].head()
)

print("\nFirst weather keys")
print(
    weather[
        ["time", "lat_grid", "lon_grid"]
    ].head()
)

# -------------------------------------------------------
# Merge
# -------------------------------------------------------

pairs_weather = pairs.merge(
    weather[
        [
            "time",
            "lat_grid",
            "lon_grid",
            "temperature",
            "wind_speed",
            "pressure",
            "precipitation",
            "tcc",
        ]
    ],
    on=[
        "time",
        "lat_grid",
        "lon_grid",
    ],
    how="left",
)

# -------------------------------------------------------
# Missing values
# -------------------------------------------------------

print("\nMissing weather values")

print(
    pairs_weather[
        [
            "temperature",
            "wind_speed",
            "pressure",
            "precipitation",
            "tcc",
        ]
    ].isna().sum()
)

# -------------------------------------------------------
# Save
# -------------------------------------------------------

pairs_weather.to_pickle(
    "processed_data/candidate_pairs_weather.pkl"
)

pairs_weather.to_csv(
    "processed_data/candidate_pairs_weather.csv",
    index=False,
)

print("\nSaved:")
print("processed_data/candidate_pairs_weather.pkl")
print("processed_data/candidate_pairs_weather.csv")

print("\nRows:", len(pairs_weather))

print("\nDONE")


