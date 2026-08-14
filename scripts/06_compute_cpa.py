import numpy as np
import pandas as pd

print("=" * 70)
print("COMPUTING CLOSEST POINT OF APPROACH (CPA)")
print("=" * 70)

pairs = pd.read_pickle(
    "processed_data/candidate_pairs.pkl"
)

print("Candidate pairs:", len(pairs))


EARTH_RADIUS = 6371000.0


def latlon_to_xy(lat, lon, ref_lat):

    x = np.radians(lon) * EARTH_RADIUS * np.cos(np.radians(ref_lat))

    y = np.radians(lat) * EARTH_RADIUS

    return x, y


# ----------------------------------------------------------
# Aircraft A
# ----------------------------------------------------------

pairs["x1"], pairs["y1"] = latlon_to_xy(
    pairs["lat1"],
    pairs["lon1"],
    (pairs["lat1"] + pairs["lat2"]) / 2
)

# ----------------------------------------------------------
# Aircraft B
# ----------------------------------------------------------

pairs["x2"], pairs["y2"] = latlon_to_xy(
    pairs["lat2"],
    pairs["lon2"],
    (pairs["lat1"] + pairs["lat2"]) / 2
)

# ----------------------------------------------------------
# Velocity vectors
# ----------------------------------------------------------

heading1 = np.radians(pairs["heading1"])

heading2 = np.radians(pairs["heading2"])

pairs["vx1"] = pairs["velocity1"] * np.sin(heading1)

pairs["vy1"] = pairs["velocity1"] * np.cos(heading1)

pairs["vx2"] = pairs["velocity2"] * np.sin(heading2)

pairs["vy2"] = pairs["velocity2"] * np.cos(heading2)

# ----------------------------------------------------------
# Relative position
# ----------------------------------------------------------

rx = pairs["x2"] - pairs["x1"]

ry = pairs["y2"] - pairs["y1"]

# ----------------------------------------------------------
# Relative velocity
# ----------------------------------------------------------

rvx = pairs["vx2"] - pairs["vx1"]

rvy = pairs["vy2"] - pairs["vy1"]

rv2 = rvx**2 + rvy**2

rv2 = np.where(rv2 < 1e-6, 1e-6, rv2)

# ----------------------------------------------------------
# Time to CPA
# ----------------------------------------------------------

tcpa = -(rx * rvx + ry * rvy) / rv2

tcpa = np.maximum(tcpa, 0)

pairs["tcpa"] = tcpa

# ----------------------------------------------------------
# Distance at CPA
# ----------------------------------------------------------

cx = rx + rvx * tcpa

cy = ry + rvy * tcpa

pairs["dcpa"] = np.sqrt(cx**2 + cy**2)

# ----------------------------------------------------------
# Horizontal conflict
# ----------------------------------------------------------

pairs["horizontal_conflict"] = (
    pairs["dcpa"] < 5000
)

# ----------------------------------------------------------
# Vertical conflict
# ----------------------------------------------------------

pairs["vertical_conflict"] = (
    pairs["vertical_distance"] < 300
)

# ----------------------------------------------------------
# Final conflict label
# ----------------------------------------------------------

pairs["conflict"] = (
    pairs["horizontal_conflict"] &
    pairs["vertical_conflict"]
).astype(int)

# ----------------------------------------------------------
# Statistics
# ----------------------------------------------------------

print()

print("Conflict statistics")

print(pairs["conflict"].value_counts())

print()

print("TCPA")

print(pairs["tcpa"].describe())

print()

print("DCPA")

print(pairs["dcpa"].describe())

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

pairs.to_pickle(
    "processed_data/candidate_pairs_cpa.pkl"
)

pairs.to_csv(
    "processed_data/candidate_pairs_cpa.csv",
    index=False
)

print()

print("=" * 70)

print("DONE")

print("=" * 70)

print()

print("Saved:")

print("processed_data/candidate_pairs_cpa.pkl")

print("processed_data/candidate_pairs_cpa.csv")