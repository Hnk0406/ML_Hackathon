from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TRIP_ID = "00000001"
PARTITION = "train"

TRIP_DIR = BASE_DIR / PARTITION / TRIP_ID

track = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

nodes = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

arcs = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.arcs",
    sep=r"\s+",
    header=None,
    names=["from_node_id", "to_node_id"]
)

route = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.route",
    sep=r"\s+",
    header=None,
    names=["arc_id"]
)

print("\n========== GPS DATA QUALITY ==========")

print("\nMissing values:")
print(track.isnull().sum())

print("\nDuplicate GPS observations:")
print(track.duplicated().sum())

print("\nInvalid longitude values:")
print((~track["longitude"].between(-180, 180)).sum())

print("\nInvalid latitude values:")
print((~track["latitude"].between(-90, 90)).sum())

print("\nNon-finite GPS values:")
print((~np.isfinite(track.select_dtypes(include="number"))).sum())

print("\nTimestamp analysis:")

time_diff = track["timestamp"].diff().dropna()

print("Minimum interval:", time_diff.min())
print("Maximum interval:", time_diff.max())
print("Average interval:", time_diff.mean())
print("Intervals not equal to 1 second:", (time_diff != 1).sum())
print("Non-increasing timestamps:", (time_diff <= 0).sum())

print("\n========== ROAD NODE VALIDATION ==========")

print("Total nodes:", len(nodes))

invalid_node_coordinates = (
    ~nodes["longitude"].between(-180, 180)
    | ~nodes["latitude"].between(-90, 90)
)

print("Invalid node coordinates:", invalid_node_coordinates.sum())

print("\n========== ROAD ARC VALIDATION ==========")

print("Total arcs:", len(arcs))

invalid_arcs = (
    (arcs["from_node_id"] < 0)
    | (arcs["from_node_id"] >= len(nodes))
    | (arcs["to_node_id"] < 0)
    | (arcs["to_node_id"] >= len(nodes))
)

print("Invalid arc connections:", invalid_arcs.sum())

print("\n========== REFERENCE ROUTE VALIDATION ==========")

print("Total route segments:", len(route))

invalid_route_ids = (
    (route["arc_id"] < 0)
    | (route["arc_id"] >= len(arcs))
)

print("Invalid route arc IDs:", invalid_route_ids.sum())

print("\n========== DATA QUALITY SUMMARY ==========")

print("GPS observations:", len(track))
print("Road nodes:", len(nodes))
print("Road arcs:", len(arcs))
print("Reference route segments:", len(route))

print("\nData quality analysis completed!")