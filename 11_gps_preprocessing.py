
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "processed"

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

processing_summary = []

for partition in ["train", "student_test"]:
    partition_dir = BASE_DIR / partition

    for trip_dir in sorted(partition_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name
        track_file = trip_dir / f"{trip_id}.track"

        if not track_file.exists():
            continue

        track = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        track["time_gap_sec"] = track["timestamp"].diff()

        distances = np.full(len(track), np.nan)

        if len(track) > 1:
            distances[1:] = haversine_distance(
                track["longitude"].iloc[:-1].to_numpy(),
                track["latitude"].iloc[:-1].to_numpy(),
                track["longitude"].iloc[1:].to_numpy(),
                track["latitude"].iloc[1:].to_numpy()
            )

        track["distance_from_previous_m"] = distances

        track["speed_from_previous_kmh"] = (
            track["distance_from_previous_m"]
            / track["time_gap_sec"]
            * 3.6
        )

        track.loc[
            track["time_gap_sec"] <= 0,
            "speed_from_previous_kmh"
        ] = np.nan

        track["large_time_gap"] = (
            track["time_gap_sec"] > GAP_THRESHOLD
        )

        track["high_speed_segment"] = (
            track["speed_from_previous_kmh"] > SPEED_THRESHOLD
        )

        track["suspicious_segment"] = (
            track["large_time_gap"] |
            track["high_speed_segment"]
        )

        output_partition = OUTPUT_DIR / partition
        output_partition.mkdir(parents=True, exist_ok=True)

        output_file = output_partition / f"{trip_id}.csv"
        track.to_csv(output_file, index=False)

        processing_summary.append({
            "partition": partition,
            "trip_id": trip_id,
            "original_points": len(track),
            "large_time_gaps": int(track["large_time_gap"].sum()),
            "high_speed_segments": int(track["high_speed_segment"].sum()),
            "flagged_segments": int(track["suspicious_segment"].sum()),
            "processed_points": len(track)
        })

summary = pd.DataFrame(processing_summary)

summary.to_csv(
    BASE_DIR / "outputs" / "reports" / "gps_preprocessing_summary.csv",
    index=False
)

print("========== GPS PREPROCESSING ==========")
print("Trips processed:", len(summary))
print("Original GPS points:", summary["original_points"].sum())
print("Processed GPS points:", summary["processed_points"].sum())
print("Large timestamp gaps:", summary["large_time_gaps"].sum())
print("High-speed segments:", summary["high_speed_segments"].sum())
print("Flagged segments:", summary["flagged_segments"].sum())
print("Original GPS observations preserved:", 
      (summary["original_points"] == summary["processed_points"]).all())

print("\nProcessed files saved in:", OUTPUT_DIR)
print("Preprocessing summary saved successfully!")
