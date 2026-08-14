import pandas as pd

print("=" * 60)
print("JOINING WAKE FEATURES")
print("=" * 60)

print("\nLoading wake lookup...")
wake = pd.read_csv("processed_data/aircraft_wake_lookup.csv")
wake["icao24"] = wake["icao24"].astype(str).str.lower()

print("Loading region_final.pkl...")
region = pd.read_pickle("processed_data/region_final.pkl")

print("Shape:", region.shape)
print("Columns:", region.columns.tolist())

# Detect ICAO column
possible = ["icao24", "ICAO24", "icao", "hex", "aircraft"]

icao_col = None
for c in possible:
    if c in region.columns:
        icao_col = c
        break

if icao_col is None:
    raise Exception("Could not find ICAO column.")

print("Using ICAO column:", icao_col)

region[icao_col] = region[icao_col].astype(str).str.lower()

region = region.merge(
    wake[["icao24", "wake_category"]],
    left_on=icao_col,
    right_on="icao24",
    how="left",
)

coverage = region["wake_category"].notna().mean() * 100

print(f"\nWake coverage: {coverage:.2f}%")
print(region["wake_category"].value_counts(dropna=False))

region.to_pickle("processed_data/region_final_wake.pkl")
region.to_csv("processed_data/region_final_wake.csv", index=False)

print("\nSaved:")
print("processed_data/region_final_wake.pkl")
print("processed_data/region_final_wake.csv")