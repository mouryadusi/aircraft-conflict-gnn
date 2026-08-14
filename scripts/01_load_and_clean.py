"""
Script 1: Load raw OpenSky avro files, filter to region, clean.
Run: python scripts/01_load_and_clean.py
Output: processed_data/region_final.pkl
"""
import numpy as np
import pandas as pd
import fastavro
from pathlib import Path
import pickle
import os

os.makedirs("processed_data", exist_ok=True)

# --- Load both avro files ---
file_path_1 = Path("data/raw/opensky_avro/states_2017-06-05-02.avro")
print("File 1 exists:", file_path_1.exists())
with open(file_path_1, "rb") as f:
    records1 = list(fastavro.reader(f))
df = pd.DataFrame(records1)
print("df shape:", df.shape)

file_path_2 = Path("data/raw/opensky_avro/states_2018-05-28-00.avro")
print("File 2 exists:", file_path_2.exists())
with open(file_path_2, "rb") as f:
    records2 = list(fastavro.reader(f))
df2 = pd.DataFrame(records2)
print("df2 shape:", df2.shape)

# --- Filter to region, tag source day, combine ---
region = df[(df['lat'].between(48, 53)) & (df['lon'].between(5, 15))].copy()
region2 = df2[(df2['lat'].between(48, 53)) & (df2['lon'].between(5, 15))].copy()
region["source_day"] = "2017-06-05"
region2["source_day"] = "2018-05-28"

combined_region = pd.concat([region, region2], ignore_index=True)
print("combined_region:", combined_region.shape, combined_region["icao24"].nunique(), "aircraft")

# --- Clean: ground filter, altitude filter, climb-rate plausibility filter ---
region_final = combined_region[
    (combined_region["onground"] == False) &
    (combined_region["baroaltitude"] > 1000)
].copy()

region_final = region_final.sort_values(["icao24", "source_day", "time"])
region_final["dt"] = region_final.groupby(["icao24", "source_day"])["time"].diff()
region_final["dalt"] = region_final.groupby(["icao24", "source_day"])["baroaltitude"].diff()
region_final["climb_rate_fpm"] = (region_final["dalt"] * 3.28084) / region_final["dt"] * 60

MAX_CLIMB_RATE = 6000
region_final = region_final[
    (region_final["climb_rate_fpm"].abs() <= MAX_CLIMB_RATE) | (region_final["climb_rate_fpm"].isna())
].copy()

region_final["vertrate"] = region_final.groupby(["icao24", "source_day"])["vertrate"].transform(lambda s: s.ffill().bfill())
region_final["geoaltitude"] = region_final.groupby(["icao24", "source_day"])["geoaltitude"].transform(lambda s: s.ffill().bfill())
region_final = region_final.dropna(subset=["lat", "lon", "baroaltitude"])

print("region_final:", region_final.shape, region_final["icao24"].nunique(), "aircraft")

# --- Save checkpoint ---
region_final.to_pickle("processed_data/region_final.pkl")
print("\nSaved processed_data/region_final.pkl")
print("DONE: 01_load_and_clean.py")
