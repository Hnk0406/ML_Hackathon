
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "outputs" / "processed"

results = []

for partition in ["train", "student_test"]:
    original_dir = BASE_DIR / partition
    processed_dir = PROCESSED_DIR / partition

    for trip_dir in sorted(original_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

        original_file = trip_dir / f"{trip_id}.track"
        processed_file = processed_dir / f"{trip_id}.csv"

        if not original_file.exists() or not processed_file.exists():
            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "status": "MISSING FILE"
            })
            continue

        original = pd.read_csv(
            original_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        processed = pd.read_csv(processed_file)

        coordinates_match = np.allclose(
            original[["longitude", "latitude", "timestamp"]].to_numpy(),
            processed[["longitude", "latitude", "timestamp"]].to_numpy(),
            equal_nan=True
        )

        valid_coordinates = (
            processed["longitude"].between(-180, 180).all()
            and processed["latitude"].between(-90, 90).all()
        )

        flags_match = (
            processed["large_time_gap"].fillna(False)
            == (processed["time_gap_sec"] > 10)
        ).all() and (
            processed["high_speed_segment"].fillna(False)
            == (processed["speed_from_previous_kmh"] > 160)
        ).all()

        passed = (
            len(original) == len(processed)
            and coordinates_match
            and valid_coordinates
            and flags_match
        )

        results.append({
            "partition": partition,
            "trip_id": trip_id,
            "original_points": len(original),
            "processed_points": len(processed),
            "coordinates_and_timestamps_match": coordinates_match,
            "valid_coordinates": valid_coordinates,
            "flags_match": flags_match,
            "status": "PASS" if passed else "FAIL"
        })

report = pd.DataFrame(results)

output_file = BASE_DIR / "outputs" / "reports" / "processed_data_validation.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(output_file, index=False)

print("========== PROCESSED DATA VALIDATION ==========")
print("Total trips checked:", len(report))
print("\nValidation results:")
print(report["status"].value_counts())

print("\nFailed trips:")
print(report[report["status"] != "PASS"].to_string(index=False))

print("\nAll trips passed:", (report["status"] == "PASS").all())
print("Validation report saved:", output_file)
