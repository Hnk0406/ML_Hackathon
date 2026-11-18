
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRIP_ID = "00000001"

INPUT_DIR = BASE_DIR / "outputs" / "processed" / "road_network"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

arcs = pd.read_csv(
    INPUT_DIR / f"{TRIP_ID}_processed_arcs.csv"
)

nodes = pd.read_csv(
    BASE_DIR / "train" / TRIP_ID / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

invalid_arcs = arcs[
    (~arcs["valid_length"]) |
    (~np.isfinite(arcs["length_m"]))
]

duplicate_arcs = arcs[
    arcs.duplicated(
        subset=["from_node_id", "to_node_id"],
        keep=False
    )
].sort_values(["from_node_id", "to_node_id"])

long_arcs = arcs.nlargest(20, "length_m")

print("========== ROAD NETWORK QUALITY ANALYSIS ==========")

print("\n1. INVALID ARCS")
print("Total:", len(invalid_arcs))
print(invalid_arcs.to_string(index=False))

print("\n2. DUPLICATE DIRECTED ARCS")
print("Total rows involved:", len(duplicate_arcs))
print(duplicate_arcs.head(20).to_string(index=False))

print("\n3. LONGEST ROAD ARCS")
print(long_arcs[
    ["arc_id", "from_node_id", "to_node_id", "length_m"]
].to_string(index=False))

print("\n4. ZERO-LENGTH ARCS")
zero_arcs = arcs[arcs["length_m"] == 0]
print("Total:", len(zero_arcs))
print(zero_arcs.to_string(index=False))

invalid_arcs.to_csv(
    OUTPUT_DIR / "invalid_road_arcs.csv",
    index=False
)

duplicate_arcs.to_csv(
    OUTPUT_DIR / "duplicate_road_arcs.csv",
    index=False
)

long_arcs.to_csv(
    OUTPUT_DIR / "long_road_arcs.csv",
    index=False
)

print("\nReports saved successfully!")
