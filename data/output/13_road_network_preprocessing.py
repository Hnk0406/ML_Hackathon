
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRIP_ID = "00000001"

TRIP_DIR = BASE_DIR / "train" / TRIP_ID
OUTPUT_DIR = BASE_DIR / "outputs" / "processed" / "road_network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS = 6371000

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

arcs["arc_id"] = np.arange(len(arcs))

from_nodes = nodes.iloc[arcs["from_node_id"]].reset_index(drop=True)
to_nodes = nodes.iloc[arcs["to_node_id"]].reset_index(drop=True)

lon1 = np.radians(from_nodes["longitude"].to_numpy())
lat1 = np.radians(from_nodes["latitude"].to_numpy())
lon2 = np.radians(to_nodes["longitude"].to_numpy())
lat2 = np.radians(to_nodes["latitude"].to_numpy())

dlon = lon2 - lon1
dlat = lat2 - lat1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
)

distances = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))

arcs["length_m"] = distances

arcs["self_loop"] = (
    arcs["from_node_id"] == arcs["to_node_id"]
)

arcs["duplicate_directed_arc"] = arcs.duplicated(
    subset=["from_node_id", "to_node_id"],
    keep="first"
)

arcs["valid_length"] = (
    np.isfinite(arcs["length_m"])
    & (arcs["length_m"] > 0)
)

output_file = OUTPUT_DIR / f"{TRIP_ID}_processed_arcs.csv"
arcs.to_csv(output_file, index=False)

print("========== ROAD NETWORK PREPROCESSING ==========")
print("Trip ID:", TRIP_ID)
print("Total nodes:", len(nodes))
print("Total arcs:", len(arcs))
print("Self-loops:", int(arcs["self_loop"].sum()))
print("Duplicate directed arcs:", int(arcs["duplicate_directed_arc"].sum()))
print("Valid arc lengths:", int(arcs["valid_length"].sum()))
print("Invalid arc lengths:", int((~arcs["valid_length"]).sum()))

print("\nArc length statistics (metres):")
print(arcs["length_m"].describe().round(3))

print("\nFirst 10 processed arcs:")
print(arcs.head(10).to_string(index=False))

print("\nProcessed road network saved at:")
print(output_file)
