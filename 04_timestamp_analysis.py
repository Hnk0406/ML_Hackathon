
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TRIP_ID = "00000001"

track_file = (
    BASE_DIR / "train" / TRIP_ID / f"{TRIP_ID}.track"
)

track = pd.read_csv(
    track_file,
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)


def haversine(lon1, lat1, lon2, lat2):

    earth_radius = 6371000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c


track["time_gap"] = track["timestamp"].diff()

track["distance_m"] = haversine(
    track["longitude"].shift(),
    track["latitude"].shift(),
    track["longitude"],
    track["latitude"]
)

track["speed_mps"] = (
    track["distance_m"] / track["time_gap"]
)

track["speed_kmh"] = track["speed_mps"] * 3.6

print("\n========== TIMESTAMP ANALYSIS ==========")

print("Total GPS observations:", len(track))
print("Total time intervals:", track["time_gap"].count())

print("Minimum time gap:", track["time_gap"].min())
print("Maximum time gap:", track["time_gap"].max())
print("Average time gap:", track["time_gap"].mean())

print("\n========== INTERVAL DISTRIBUTION ==========")

print(track["time_gap"].value_counts().sort_index())

print("\n========== GPS GAP DETAILS ==========")

gaps = track[
    track["time_gap"] > 1
].copy()

gaps["previous_timestamp"] = (
    track["timestamp"].shift()
)

gaps["previous_longitude"] = (
    track["longitude"].shift()
)

gaps["previous_latitude"] = (
    track["latitude"].shift()
)

gap_report = gaps[
    [
        "previous_timestamp",
        "timestamp",
        "time_gap",
        "previous_longitude",
        "previous_latitude",
        "longitude",
        "latitude",
        "distance_m",
        "speed_kmh"
    ]
]

print(gap_report.to_string(index=False))

print("\n========== SUMMARY ==========")

print("Normal 1-second intervals:",
      (track["time_gap"] == 1).sum())

print("Intervals greater than 1 second:",
      (track["time_gap"] > 1).sum())

print("Total GPS gaps:", len(gaps))

OUTPUT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / "gps_gap_analysis.csv"

gap_report.to_csv(output_file, index=False)

print("\nGap analysis saved successfully!")
print("Output:", output_file)
