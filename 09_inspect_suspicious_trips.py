
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "outputs" / "reports"

anomalies = pd.read_csv(
    REPORT_DIR / "gps_anomaly_details.csv",
    dtype={"trip_id": str}
)

def inspect_trip(partition, trip_id, anomaly_type, top_n=3):
    trip_anomalies = anomalies[
        (anomalies["partition"] == partition) &
        (anomalies["trip_id"] == trip_id)
    ].copy()

    if anomaly_type == "gap":
        trip_anomalies = trip_anomalies[
            trip_anomalies["time_gap_sec"] > 10
        ].nlargest(top_n, "time_gap_sec")

    else:
        trip_anomalies = trip_anomalies[
            trip_anomalies["speed_kmh"] > 160
        ].nlargest(top_n, "speed_kmh")

    track_file = (
        BASE_DIR / partition / trip_id / f"{trip_id}.track"
    )

    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    print(f"\n========== TRIP {trip_id} ==========")

    if trip_anomalies.empty:
        print("No matching suspicious segments found.")
        return

    for _, anomaly in trip_anomalies.iterrows():
        previous_index = int(anomaly["previous_point_index"])
        current_index = int(anomaly["current_point_index"])

        print("\nSuspicious segment:")
        print("Reason:", anomaly["reason"])
        print("Time gap:", anomaly["time_gap_sec"], "seconds")
        print("Distance:", round(anomaly["distance_m"], 2), "metres")
        print("Speed:", round(anomaly["speed_kmh"], 2), "km/h")

        start = max(0, previous_index - 2)
        end = min(len(track), current_index + 3)

        print("\nOriginal GPS observations:")
        print(track.iloc[start:end].to_string(index=True))

inspect_trip("train", "00000090", "gap", 3)
inspect_trip("train", "00000079", "speed", 3)
inspect_trip("train", "00000013", "speed", 3)
