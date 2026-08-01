from traffic.data import opensky

print("Downloading aircraft data...")

flights = opensky.history(
    start="2026-07-01 10:00",
    stop="2026-07-01 10:30",
    bounds=(10, 20, 70, 80)
)

print(flights)

flights.to_parquet(
    "data/raw/flights.parquet"
)

print("Saved data/raw/flights.parquet")
