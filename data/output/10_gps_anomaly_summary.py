
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "outputs" / "reports"

anomalies = pd.read_csv(
    REPORT_DIR / "gps_anomaly_details.csv",
    dtype={"trip_id": str}
)

summary = anomalies.groupby(
    ["partition", "trip_id"]
).agg(
    total_anomalies=("trip_id", "count"),
    large_gaps=("time_gap_sec", lambda x: (x > 10).sum()),
    high_speed_segments=("speed_kmh", lambda x: (x > 160).sum()),
    maximum_gap_sec=("time_gap_sec", "max"),
    maximum_speed_kmh=("speed_kmh", "max")
).reset_index()

summary = summary.sort_values(
    by="total_anomalies",
    ascending=False
)

output_file = REPORT_DIR / "gps_anomaly_summary.csv"

summary.to_csv(output_file, index=False)

print("========== GPS ANOMALY SUMMARY ==========")
print("Trips containing anomalies:", len(summary))

print("\nTotal large gaps:", summary["large_gaps"].sum())
print("Total high-speed segments:", summary["high_speed_segments"].sum())

print("\nTop 15 trips with the most anomalies:")
print(summary.head(15).to_string(index=False))

print("\nSummary saved successfully!")
print(output_file)
