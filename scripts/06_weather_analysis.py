import pandas as pd
import numpy as np

weather1 = pd.read_csv("data/raw/weather/weather_2017_06_05.csv")
weather2 = pd.read_csv("data/raw/weather/weather_2018_05_28.csv")
weather1["source_day"] = "2017-06-05"
weather2["source_day"] = "2018-05-28"
weather = pd.concat([weather1, weather2], ignore_index=True)

weather["valid"] = pd.to_datetime(weather["valid"], utc=True)
weather["time"] = weather["valid"].astype("int64") // 10**9
weather = weather[(weather["lat"].between(48, 53)) & (weather["lon"].between(5, 15))].copy()

print("Total weather obs in region:", weather.shape)
print("Unique stations:", weather["station"].nunique())
print(weather["source_day"].value_counts())

weather["wind_speed_kt"] = pd.to_numeric(weather["sknt"], errors="coerce")
weather["visibility_mi"] = pd.to_numeric(weather["vsby"], errors="coerce")
weather["wind_dir_deg"] = pd.to_numeric(weather["drct"], errors="coerce")
weather["temp_f"] = pd.to_numeric(weather["tmpf"], errors="coerce")
weather["gust_kt"] = pd.to_numeric(weather["gust"], errors="coerce")

print(weather[["wind_speed_kt", "visibility_mi", "temp_f"]].describe())

def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R_nm * np.arcsin(np.sqrt(a))

labelled = pd.read_csv("processed_data/labelled_large.csv")
region_final = pd.read_pickle("processed_data/region_final.pkl")

results = []
for day in labelled["source_day"].unique():
    day_weather = weather[weather["source_day"] == day].copy()
    day_labelled = labelled[labelled["source_day"] == day].copy()
    day_region = region_final[region_final["source_day"] == day]

    for t in day_labelled["time"].unique():
        snapshot = day_region[day_region["time"] == t]
        if len(snapshot) == 0:
            continue
        anchor_lat, anchor_lon = snapshot["lat"].mean(), snapshot["lon"].mean()

        day_weather["time_diff"] = (day_weather["time"] - t).abs()
        nearest_time = day_weather.nsmallest(20, "time_diff")
        if len(nearest_time) == 0:
            continue
        nearest_time = nearest_time.copy()
        nearest_time["dist_nm"] = haversine_nm(anchor_lat, anchor_lon, nearest_time["lat"], nearest_time["lon"])
        best = nearest_time.nsmallest(1, "dist_nm").iloc[0]

        results.append({
            "time": t, "source_day": day,
            "wind_speed_kt": best["wind_speed_kt"],
            "visibility_mi": best["visibility_mi"],
            "gust_kt": best["gust_kt"],
            "weather_station_dist_nm": best["dist_nm"],
        })

weather_by_timestep = pd.DataFrame(results)
print("Weather-by-timestep shape:", weather_by_timestep.shape)
print(weather_by_timestep.describe())

labelled_weather = labelled.merge(weather_by_timestep, on=["time", "source_day"], how="left")

print("\nMean weather by conflict label:")
print(labelled_weather.groupby("label_conflict")[["wind_speed_kt", "visibility_mi"]].mean())

print("\nCorrelation (closing rate vs wind speed):")
print(labelled_weather[["closing_rate_nm_s", "wind_speed_kt"]].corr())

labelled_weather.to_csv("processed_data/labelled_with_weather.csv", index=False)
print("Saved processed_data/labelled_with_weather.csv")
print("DONE: 06_weather_analysis.py")
