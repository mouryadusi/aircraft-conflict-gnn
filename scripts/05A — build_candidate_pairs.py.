import os
import numpy as np
import pandas as pd
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "processed_data/trajectories.pkl"

OUTPUT_DIR = "processed_data/candidate_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_HORIZONTAL_DISTANCE = 10000      # meters
CHUNK_SIZE = 1000                    # timestamps per output file

EARTH_RADIUS = 6371000

# ============================================================
# HAVERSINE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS * c


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("GENERATING CANDIDATE PAIRS")
print("=" * 70)

traj = pd.read_pickle(INPUT_FILE)

print(f"Trajectory rows : {len(traj):,}")

groups = list(traj.groupby("time"))

print(f"Unique timestamps : {len(groups):,}")

# ============================================================
# PROCESS
# ============================================================

chunk = []

chunk_id = 1

processed = 0

total_pairs = 0

for timestamp, group in groups:

    processed += 1

    if processed % 100 == 0:
        print(f"Timestamp {processed:,}/{len(groups):,}", end="\r")

    if len(group) < 2:
        continue

    rows = group.reset_index(drop=True)

    for i, j in combinations(range(len(rows)), 2):

        a = rows.iloc[i]
        b = rows.iloc[j]

        distance = haversine(
            a.lat,
            a.lon,
            b.lat,
            b.lon,
        )

        if distance > MAX_HORIZONTAL_DISTANCE:
            continue

        vertical = abs(
            a.baroaltitude - b.baroaltitude
        )

        rel_speed = abs(
            a.velocity - b.velocity
        )

        rel_heading = abs(
            a.heading - b.heading
        )

        if rel_heading > 180:
            rel_heading = 360 - rel_heading

        chunk.append({

            "time": timestamp,

            "icao24_1": a.icao24,
            "icao24_2": b.icao24,

            "lat1": a.lat,
            "lon1": a.lon,

            "lat2": b.lat,
            "lon2": b.lon,

            "alt1": a.baroaltitude,
            "alt2": b.baroaltitude,

            "velocity1": a.velocity,
            "velocity2": b.velocity,

            "heading1": a.heading,
            "heading2": b.heading,

            "distance_now": distance,

            "vertical_distance": vertical,

            "relative_speed": rel_speed,

            "relative_heading": rel_heading,

        })

        total_pairs += 1

    # --------------------------------------------------------

    if processed % CHUNK_SIZE == 0:

        df = pd.DataFrame(chunk)

        outfile = os.path.join(
            OUTPUT_DIR,
            f"candidate_pairs_part_{chunk_id:04d}.pkl",
        )

        df.to_pickle(outfile)

        print()

        print(f"Saved {outfile}")

        print(f"Rows : {len(df):,}")

        chunk = []

        chunk_id += 1

# ============================================================
# SAVE REMAINING
# ============================================================

if len(chunk) > 0:

    df = pd.DataFrame(chunk)

    outfile = os.path.join(
        OUTPUT_DIR,
        f"candidate_pairs_part_{chunk_id:04d}.pkl",
    )

    df.to_pickle(outfile)

    print()

    print(f"Saved {outfile}")

    print(f"Rows : {len(df):,}")

print()

print("=" * 70)

print("FINISHED")

print("=" * 70)

print(f"Candidate pairs : {total_pairs:,}")

print(f"Chunks created  : {chunk_id}")
