import pandas as pd

print("=" * 70)
print("CLEANING OPENSKY STATE VECTORS")
print("=" * 70)

# ---------------------------------------------------------
# Load raw Trino data
# ---------------------------------------------------------

df = pd.read_pickle("raw_data/trino_day.pkl")

print("\nOriginal shape:", df.shape)

# ---------------------------------------------------------
# Remove duplicate records
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["icao24", "time"]
)

print(f"Removed duplicates : {before-len(df)}")

# ---------------------------------------------------------
# Remove missing coordinates
# ---------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=[
        "lat",
        "lon",
        "baroaltitude",
        "velocity"
    ]
)

print(f"Removed NaNs       : {before-len(df)}")

# ---------------------------------------------------------
# Remove invalid coordinates
# ---------------------------------------------------------

before = len(df)

df = df[
    (df["lat"] >= -90)
    & (df["lat"] <= 90)
    & (df["lon"] >= -180)
    & (df["lon"] <= 180)
]

print(f"Invalid coords     : {before-len(df)}")

# ---------------------------------------------------------
# Remove aircraft on ground
# ---------------------------------------------------------

before = len(df)

df = df[df["onground"] == False]

print(f"Ground aircraft    : {before-len(df)}")

# ---------------------------------------------------------
# Remove stationary aircraft
# ---------------------------------------------------------

before = len(df)

df = df[df["velocity"] > 20]

print(f"Very slow aircraft : {before-len(df)}")

# ---------------------------------------------------------
# Remove unrealistic altitude
# ---------------------------------------------------------

before = len(df)

df = df[
    (df["baroaltitude"] > 0)
    &
    (df["baroaltitude"] < 18000)
]

print(f"Bad altitude       : {before-len(df)}")

# ---------------------------------------------------------
# Sort chronologically
# ---------------------------------------------------------

df = df.sort_values(
    ["icao24", "time"]
)

df = df.reset_index(drop=True)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print("\nClean dataset")

print(df.shape)

print("\nAircraft:")

print(df["icao24"].nunique())

print("\nTime range:")

print(
    pd.to_datetime(df.time.min(), unit="s"),
    "->",
    pd.to_datetime(df.time.max(), unit="s"),
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_pickle(
    "processed_data/state_vectors_clean.pkl"
)

df.to_csv(
    "processed_data/state_vectors_clean.csv",
    index=False,
)

print("\nSaved:")

print("processed_data/state_vectors_clean.pkl")

print("processed_data/state_vectors_clean.csv")

print("\nDONE: 03_clean_state_vectors.py")