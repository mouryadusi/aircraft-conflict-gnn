import xarray as xr
import pandas as pd

print("=" * 70)
print("PROCESSING ERA5 WEATHER")
print("=" * 70)

# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------

instant = xr.open_dataset(
    "weather_data/extracted/data_stream-oper_stepType-instant.nc"
)

accum = xr.open_dataset(
    "weather_data/extracted/data_stream-oper_stepType-accum.nc"
)

# -------------------------------------------------------
# Convert to DataFrames
# -------------------------------------------------------

instant_df = (
    instant[
        [
            "t2m",
            "sp",
            "u10",
            "v10",
            "tcc",
        ]
    ]
    .to_dataframe()
    .reset_index()
)

accum_df = (
    accum[
        [
            "tp"
        ]
    ]
    .to_dataframe()
    .reset_index()
)

# -------------------------------------------------------
# Merge
# -------------------------------------------------------

weather = instant_df.merge(
    accum_df[
        [
            "valid_time",
            "latitude",
            "longitude",
            "tp",
        ]
    ],
    on=[
        "valid_time",
        "latitude",
        "longitude",
    ],
    how="left",
)

# -------------------------------------------------------
# Rename columns
# -------------------------------------------------------

weather = weather.rename(
    columns={
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "t2m": "temperature",
        "u10": "wind_u",
        "v10": "wind_v",
        "sp": "pressure",
        "tp": "precipitation",
    }
)

# -------------------------------------------------------
# Wind speed
# -------------------------------------------------------

weather["wind_speed"] = (
    weather["wind_u"]**2
    + weather["wind_v"]**2
) ** 0.5

# -------------------------------------------------------
# Convert units
# -------------------------------------------------------

# Kelvin → Celsius
weather["temperature"] = weather["temperature"] - 273.15

# Pa → hPa
weather["pressure"] = weather["pressure"] / 100

# metres → mm
weather["precipitation"] = weather["precipitation"] * 1000

# -------------------------------------------------------
# Keep useful columns
# -------------------------------------------------------

weather = weather[
    [
        "time",
        "lat",
        "lon",
        "temperature",
        "wind_speed",
        "pressure",
        "precipitation",
        "tcc",
    ]
]

print("\nProcessed weather")
print(weather.shape)
print(weather.head())

# -------------------------------------------------------
# Save
# -------------------------------------------------------

weather.to_pickle(
    "processed_data/weather_processed.pkl"
)

weather.to_csv(
    "processed_data/weather_processed.csv",
    index=False,
)

print("\nSaved:")
print("processed_data/weather_processed.pkl")
print("processed_data/weather_processed.csv")

print("\nDONE")




