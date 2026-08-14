import pandas as pd

aircraft = pd.read_csv("data/raw/aircraftDatabase.csv")

aircraft["icao24"] = aircraft["icao24"].astype(str).str.lower()

wake_lookup = {

    "A388":"Super",

    "B748":"Heavy",
    "B744":"Heavy",
    "B742":"Heavy",
    "B743":"Heavy",
    "B741":"Heavy",
    "B77W":"Heavy",
    "B773":"Heavy",
    "B772":"Heavy",
    "B789":"Heavy",
    "B788":"Heavy",
    "A359":"Heavy",
    "A346":"Heavy",
    "A345":"Heavy",
    "A343":"Heavy",
    "A342":"Heavy",
    "A333":"Heavy",
    "A332":"Heavy",
    "A339":"Heavy",
    "A310":"Heavy",

    "A321":"Medium",
    "A320":"Medium",
    "A319":"Medium",
    "A318":"Medium",
    "B739":"Medium",
    "B738":"Medium",
    "B737":"Medium",
    "B736":"Medium",
    "B735":"Medium",
    "B734":"Medium",
    "B733":"Medium",
    "E190":"Medium",
    "E195":"Medium",
    "E170":"Medium",
    "E175":"Medium",
    "CRJ9":"Medium",
    "CRJ7":"Medium",
    "AT76":"Medium",
    "DH8D":"Medium",

    "C172":"Light",
    "C152":"Light",
    "PA28":"Light",
    "BE36":"Light",
    "PA31":"Light",
}

aircraft["wake_category"] = aircraft["typecode"].map(wake_lookup)

print("\nAircraft in database:", len(aircraft))

print("\nWake category counts:")
print(aircraft["wake_category"].value_counts(dropna=False))

aircraft[
    ["icao24", "typecode", "wake_category"]
].to_csv(
    "processed_data/aircraft_wake_lookup.csv",
    index=False
)

print("\nSaved processed_data/aircraft_wake_lookup.csv")