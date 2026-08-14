import cdsapi
import os

print("=" * 70)
print("DOWNLOADING ERA5 WEATHER")
print("=" * 70)

os.makedirs("weather_data", exist_ok=True)

client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",

        "variable": [
            "2m_temperature",
            "surface_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
            "total_cloud_cover",
        ],

        "year": "2017",
        "month": "06",
        "day": "05",

        "time": [
            "00:00","01:00","02:00","03:00",
            "04:00","05:00","06:00","07:00",
            "08:00","09:00","10:00","11:00",
            "12:00","13:00","14:00","15:00",
            "16:00","17:00","18:00","19:00",
            "20:00","21:00","22:00","23:00",
        ],

        "area": [
            26.8,   # North
            54.0,   # West
            23.0,   # South
            56.8,   # East
        ],

        "format": "netcdf",
    },
    "weather_data/era5_2017_06_05.nc",
)

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("Saved:")
print("weather_data/era5_2017_06_05.nc")