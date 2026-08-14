"""
Verify whether no-CPA features overlap with label-generation features.

Checks:
1. Reconstruct horizontal/vertical separation independently from raw data.
2. Compare with distance_now and vertical_distance.
3. Measure correlation and scaling ratio.

Repo: aircraft-conflict-gnn
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# CHANGE THIS IF YOUR FILE IS DIFFERENT
DATA_PATH = REPO_ROOT / "processed_data" / "candidate_pairs_weather.csv"
# ============================================================
# LOAD DATA
# ============================================================

print("Loading:", DATA_PATH)

if DATA_PATH.suffix == ".csv":
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.read_pickle(DATA_PATH)


print("\nLoaded:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "lat1",
    "lon1",
    "lat2",
    "lon2",
    "alt1",
    "alt2",
    "distance_now",
    "vertical_distance",
    "conflict",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"\nMissing columns: {missing}\n"
        "Use the dataframe containing lat/lon/alt + distance_now + vertical_distance."
    )


# ============================================================
# RECONSTRUCT DISTANCE
# ============================================================

def haversine_nm(lat1, lon1, lat2, lon2):
    """
    Calculate horizontal distance in nautical miles.
    """

    R_nm = 3440.065

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R_nm * c


print("\nReconstructing distances...")


df["h_dist_nm_reconstructed"] = haversine_nm(
    df["lat1"],
    df["lon1"],
    df["lat2"],
    df["lon2"],
)


# ADS-B altitude normally in feet

df["v_dist_ft_reconstructed"] = (
    df["alt1"] - df["alt2"]
).abs()


# ============================================================
# CHECK LABEL RECONSTRUCTION
# ============================================================

print("\n========== LABEL CHECK ==========")


reconstructed_label = (
    (df["h_dist_nm_reconstructed"] < 5)
    &
    (df["v_dist_ft_reconstructed"] < 1000)
).astype(int)


agreement = (
    reconstructed_label
    ==
    df["conflict"].astype(int)
).mean()


print(
    "Reconstructed label agreement:",
    f"{agreement:.4%}"
)


# ============================================================
# FEATURE OVERLAP CHECK
# ============================================================

print("\n========== FEATURE OVERLAP CHECK ==========")


checks = [
    (
        "distance_now",
        "h_dist_nm_reconstructed"
    ),
    (
        "vertical_distance",
        "v_dist_ft_reconstructed"
    ),
]


for feature, reconstructed in checks:

    temp = df[
        [
            feature,
            reconstructed
        ]
    ].dropna()


    correlation = temp[feature].corr(
        temp[reconstructed]
    )


    ratio = (
        temp[feature]
        /
        temp[reconstructed].replace(0, np.nan)
    ).median()


    print("\nFeature:")
    print(feature)

    print(
        "Correlation:",
        round(correlation, 6)
    )

    print(
        "Median ratio:",
        round(ratio, 6)
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n========== INTERPRETATION ==========")

print(
"""
Correlation near 1.0:
    Feature is likely the same measurement as label-rule input.

Correlation below ~0.9:
    Feature is likely independently derived.

A constant ratio indicates only unit conversion
(example: meters vs feet, km vs nautical miles).
"""
)