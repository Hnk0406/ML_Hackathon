
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

        time_diff = track["timestamp"].diff().iloc[1:].to_numpy()

        distances = haversine_distance(
            track["longitude"].iloc[:-1].to_numpy(),
            track["latitude"].iloc[:-1].to_numpy(),
            track["longitude"].iloc[1:].to_numpy(),
            track["latitude"].iloc[1:].to_numpy()
        )

        valid_time = time_diff > 0

        speeds = np.full(len(time_diff), np.nan)
        speeds[valid_time] = (
            distances[valid_time] / time_diff[valid_time] * 3.6
        )

        results.append({
            "partition": partition,
            "trip_id": trip_dir.name,
            "total_intervals": len(time_diff),
            "intervals_over_1_sec": int((time_diff > 1).sum()),
            "intervals_over_5_sec": int((time_diff > 5).sum()),
            "intervals_over_10_sec": int((time_diff > 10).sum()),
            "zero_or_negative_intervals": int((time_diff <= 0).sum()),
            "maximum_gap_sec": float(np.max(time_diff)),
            "average_gap_sec": float(np.mean(time_diff)),
            "maximum_segment_speed_kmh": (
                float(np.nanmax(speeds))
                if np.isfinite(speeds).any() else np.nan
            )
        })

report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_DIR / "gps_interval_quality.csv",
    index=False
)

print("========== GPS INTERVAL QUALITY ANALYSIS ==========")
print("Total trips analysed:", len(report))

print("\nTotal irregular intervals:")
print("Over 1 second:", report["intervals_over_1_sec"].sum())
print("Over 5 seconds:", report["intervals_over_5_sec"].sum())
print("Over 10 seconds:", report["intervals_over_10_sec"].sum())
print("Zero or negative:", report["zero_or_negative_intervals"].sum())

print("\nLargest timestamp gaps:")
print(
    report.nlargest(10, "maximum_gap_sec")[
        ["partition", "trip_id", "maximum_gap_sec"]
    ].to_string(index=False)
)

print("\nTrips with the highest segment speeds:")
print(
    report.nlargest(10, "maximum_segment_speed_kmh")[
        ["partition", "trip_id", "maximum_segment_speed_kmh"]
    ].to_string(index=False)
)

print("\nReport saved successfully!")
