
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS = 6371000

def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))

results = []

for partition in ["train", "student_test"]:
    partition_dir = BASE_DIR / partition

    for trip_dir in sorted(partition_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        track_file = trip_dir / f"{trip_dir.name}.track"

        if not track_file.exists():
            continue

        track = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        if len(track) < 2:
            continue

        distances = haversine_distance(
            track["longitude"].iloc[:-1].to_numpy(),
            track["latitude"].iloc[:-1].to_numpy(),
            track["longitude"].iloc[1:].to_numpy(),
            track["latitude"].iloc[1:].to_numpy()
        )

        time_differences = track["timestamp"].diff().iloc[1:].to_numpy()

        total_distance = distances.sum()

        valid_intervals = time_differences > 0
        total_time = time_differences[valid_intervals].sum()

        average_speed = (
            total_distance / total_time * 3.6
            if total_time > 0 else np.nan
        )

        results.append({
            "partition": partition,
            "trip_id": trip_dir.name,
            "gps_observations": len(track),
            "min_longitude": track["longitude"].min(),
            "max_longitude": track["longitude"].max(),
            "min_latitude": track["latitude"].min(),
            "max_latitude": track["latitude"].max(),
            "total_distance_km": total_distance / 1000,
            "average_time_interval_sec": time_differences.mean(),
            "average_speed_kmh": average_speed
        })

report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_DIR / "gps_trajectory_analysis.csv",
    index=False
)

print("========== GPS TRAJECTORY ANALYSIS ==========")
print("Total trips analysed:", len(report))
print("\nTrips by partition:")
print(report["partition"].value_counts())

print("\nOverall statistics:")
print(report[[
    "gps_observations",
    "total_distance_km",
    "average_time_interval_sec",
    "average_speed_kmh"
]].describe().round(3))

print("\nReport saved successfully!")
