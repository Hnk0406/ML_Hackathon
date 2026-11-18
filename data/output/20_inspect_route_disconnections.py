
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "outputs" / "reports"

details = pd.read_csv(
    REPORT_DIR / "disconnected_route_details.csv",
    dtype={"trip_id": str}
)

EARTH_RADIUS = 6371000

def haversine(lon1, lat1, lon2, lat2):
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

for _, item in details.iterrows():
    partition = item["partition"]
    trip_id = item["trip_id"]

    trip_dir = BASE_DIR / partition / trip_id

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    current_node = nodes.iloc[int(item["current_to_node"])]
    next_node = nodes.iloc[int(item["next_from_node"])]

    distance = haversine(
        current_node["longitude"],
        current_node["latitude"],
        next_node["longitude"],
        next_node["latitude"]
    )

    results.append({
        "partition": partition,
        "trip_id": trip_id,
        "current_arc_id": item["current_arc_id"],
        "next_arc_id": item["next_arc_id"],
        "current_to_node": item["current_to_node"],
        "next_from_node": item["next_from_node"],
        "distance_between_nodes_m": distance
    })

report = pd.DataFrame(results)

report.to_csv(
    REPORT_DIR / "route_disconnection_distances.csv",
    index=False
)

print("========== ROUTE DISCONNECTION DISTANCE ANALYSIS ==========")
print("Total disconnected transitions:", len(report))

print("\nDistance statistics:")
print(report["distance_between_nodes_m"].describe().round(3))

print("\nDetailed results:")
print(report.to_string(index=False))

print("\nReport saved successfully!")
