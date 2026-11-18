
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS = 6371000
GAP_THRESHOLD = 10
SPEED_THRESHOLD = 160

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

anomalies = []

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

        for i in range(1, len(track)):
            previous = track.iloc[i - 1]
            current = track.iloc[i]

            time_gap = current["timestamp"] - previous["timestamp"]

            distance = haversine_distance(
                previous["longitude"],
                previous["latitude"],
                current["longitude"],
                current["latitude"]
            )

            speed = (
                distance / time_gap * 3.6
                if time_gap > 0 else np.nan
            )

            reasons = []

            if time_gap > GAP_THRESHOLD:
                reasons.append("Large timestamp gap")

            if np.isfinite(speed) and speed > SPEED_THRESHOLD:
                reasons.append("High segment speed")

            if reasons:
                anomalies.append({
                    "partition": partition,
                    "trip_id": trip_dir.name,
                    "previous_point_index": i - 1,
                    "current_point_index": i,
                    "previous_timestamp": previous["timestamp"],
                    "current_timestamp": current["timestamp"],
                    "time_gap_sec": time_gap,
                    "distance_m": distance,
                    "speed_kmh": speed,
                    "previous_longitude": previous["longitude"],
                    "previous_latitude": previous["latitude"],
                    "current_longitude": current["longitude"],
                    "current_latitude": current["latitude"],
                    "reason": "; ".join(reasons)
                })

report = pd.DataFrame(anomalies)

output_file = OUTPUT_DIR / "gps_anomaly_details.csv"
report.to_csv(output_file, index=False)

print("========== GPS ANOMALY INSPECTION ==========")
print("Total suspicious segments:", len(report))

if not report.empty:
    print("\nAnomalies by reason:")
    print(report["reason"].value_counts())

    print("\nTop 15 highest-speed segments:")
    print(
        report.nlargest(15, "speed_kmh")[
            ["trip_id", "time_gap_sec", "distance_m", "speed_kmh", "reason"]
        ].to_string(index=False)
    )

    print("\nLargest timestamp gaps:")
    print(
        report.nlargest(10, "time_gap_sec")[
            ["trip_id", "previous_timestamp", "current_timestamp", "time_gap_sec"]
        ].to_string(index=False)
    )

print("\nDetailed report saved at:", output_file)
