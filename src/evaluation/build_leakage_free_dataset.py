import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "processed_data" / "candidate_pairs_weather.csv"
OUTPUT = ROOT / "processed_data" / "candidate_pairs_no_leakage.csv"

print("=" * 70)
print("BUILDING LEAKAGE-FREE DATASET")
print("=" * 70)

print(f"\nInput:  {INPUT}")
print(f"Output: {OUTPUT}")

# ------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print("\nOriginal shape:")
print(df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. REMOVE FEATURES THAT ARE DIRECTLY RELATED TO THE
#    LABEL-GENERATION / CPA LOGIC
#
# IMPORTANT:
# DO NOT REMOVE 'conflict'
# It is the TARGET LABEL.
# ------------------------------------------------------------

DROP_COLS = [
    "distance_now",
    "vertical_distance",
    "tcpa",
    "dcpa",
    "horizontal_conflict",
    "vertical_conflict",
    "conflict_now",
]

existing_drop_cols = [
    c for c in DROP_COLS
    if c in df.columns
]

print("\nRemoving:")
for c in existing_drop_cols:
    print(f"  - {c}")

df = df.drop(columns=existing_drop_cols)

# ------------------------------------------------------------
# 3. VERIFY TARGET STILL EXISTS
# ------------------------------------------------------------

if "conflict" not in df.columns:
    raise ValueError(
        "TARGET 'conflict' is missing. "
        "The leakage-free dataset must retain the target."
    )

# ------------------------------------------------------------
# 4. VERIFY REMOVED FEATURES ARE GONE
# ------------------------------------------------------------

remaining_leakage_cols = [
    c for c in DROP_COLS
    if c in df.columns
]

if remaining_leakage_cols:
    raise ValueError(
        f"Leakage-related columns still present: "
        f"{remaining_leakage_cols}"
    )

# ------------------------------------------------------------
# 5. CHECK TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\nTarget distribution:")

print(df["conflict"].value_counts(dropna=False))

print("\nTarget proportions:")

print(
    df["conflict"]
    .value_counts(normalize=True, dropna=False)
)

# ------------------------------------------------------------
# 6. FINAL DATASET
# ------------------------------------------------------------

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 70)
print("LEAKAGE-FREE DATASET CREATED")
print("=" * 70)