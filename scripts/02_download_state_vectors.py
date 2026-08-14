import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import trino
from trino.auth import OAuth2Authentication

# ============================================================
# CONFIGURATION
# ============================================================

USERNAME = "mdusi"

OUTPUT_DIR = "raw_data/trino_hourly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START = datetime(2017, 6, 5, 0, 0, tzinfo=timezone.utc)
END   = datetime(2017, 6, 5, 23, 0, tzinfo=timezone.utc)

# Dubai FIR (change later if required)
LAT_MIN = 23.0
LAT_MAX = 26.8

LON_MIN = 54.0
LON_MAX = 56.8

MAX_RETRIES = 3

# ============================================================
# CONNECT
# ============================================================

print("=" * 70)
print("CONNECTING TO OPENSKY TRINO")
print("=" * 70)

conn = trino.dbapi.connect(
    host="trino.opensky-network.org",
    port=443,
    http_scheme="https",
    user=USERNAME,
    auth=OAuth2Authentication(),
    catalog="minio",
    schema="osky",
)

cursor = conn.cursor()

print("Authentication successful.")

# ============================================================
# DOWNLOAD
# ============================================================

current = START

while current <= END:

    hour_partition = int(current.timestamp())

    outfile = os.path.join(
        OUTPUT_DIR,
        f"{hour_partition}.pkl"
    )

    if os.path.exists(outfile):
        print(f"Skipping existing hour {current}")
        current += timedelta(hours=1)
        continue

    print("\n------------------------------------------------")
    print(f"Downloading {current}")
    print("------------------------------------------------")

    query = f"""
    SELECT

        time,
        icao24,
        callsign,
        lat,
        lon,
        velocity,
        heading,
        vertrate,
        baroaltitude,
        geoaltitude,
        onground,
        lastcontact

    FROM state_vectors_data4

    WHERE hour = {hour_partition}

      AND time - lastcontact <= 15

      AND lat BETWEEN {LAT_MIN} AND {LAT_MAX}

      AND lon BETWEEN {LON_MIN} AND {LON_MAX}
    """

    success = False

    for attempt in range(MAX_RETRIES):

        try:

            cursor.execute(query)

            rows = cursor.fetchall()

            df = pd.DataFrame(
                rows,
                columns=[
                    "time",
                    "icao24",
                    "callsign",
                    "lat",
                    "lon",
                    "velocity",
                    "heading",
                    "vertrate",
                    "baroaltitude",
                    "geoaltitude",
                    "onground",
                    "lastcontact",
                ],
            )

            print(f"Rows downloaded: {len(df):,}")

            df.to_pickle(outfile)

            success = True

            break

        except Exception as e:

            print(
                f"Attempt {attempt+1}/{MAX_RETRIES} failed"
            )

            print(e)

            time.sleep(5)

    if not success:

        print("Failed after retries.")

    current += timedelta(hours=1)

# ============================================================
# MERGE
# ============================================================

print("\nMerging hourly files...")

frames = []

files = sorted(
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".pkl")
)

for f in files:

    frames.append(
        pd.read_pickle(
            os.path.join(OUTPUT_DIR, f)
        )
    )

dataset = pd.concat(
    frames,
    ignore_index=True
)

os.makedirs("raw_data", exist_ok=True)

dataset.to_pickle("raw_data/trino_day.pkl")
dataset.to_csv(
    "raw_data/trino_day.csv",
    index=False,
)

print("\n================================================")
print("DOWNLOAD COMPLETE")
print("================================================")

print(dataset.shape)

print("\nSaved:")

print("raw_data/trino_day.pkl")

print("raw_data/trino_day.csv")