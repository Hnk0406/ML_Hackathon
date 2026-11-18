
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRIP_ID = "00000001"

TRIP_DIR = BASE_DIR / "train" / TRIP_ID
INPUT_DIR = BASE_DIR / "outputs" / "processed" / "road_network"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

arcs = pd.read_csv(
    INPUT_DIR / f"{TRIP_ID}_processed_arcs.csv"
)

nodes = pd.read_csv(
    TRIP_DIR / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

invalid_arcs = arcs[arcs["length_m"] == 0]

duplicate_arcs = arcs[
    arcs.duplicated(
        subset=["from_node_id", "to_node_id"],
        keep=False
    )
]

print("========== ZERO-LENGTH ARC INSPECTION ==========")

for _, arc in invalid_arcs.iterrows():
    start_id = int(arc["from_node_id"])
    end_id = int(arc["to_node_id"])

    start_node = nodes.iloc[start_id]
    end_node = nodes.iloc[end_id]

    print(f"\nArc ID: {int(arc['arc_id'])}")
    print("Start node:", start_id)
    print("End node:", end_id)
    print("Start coordinates:", tuple(start_node))
    print("End coordinates:", tuple(end_node))
    print(
        "Coordinates identical:",
        start_node.equals(end_node)
    )

print("\n========== DUPLICATE ARC INSPECTION ==========")

duplicate_groups = duplicate_arcs.groupby(
    ["from_node_id", "to_node_id"]
)

print("Total duplicate groups:", len(duplicate_groups))

print("\nFirst 15 duplicate groups:")

for (from_id, to_id), group in list(duplicate_groups)[:15]:
    print({
        "from_node": from_id,
        "to_node": to_id,
        "arc_ids": group["arc_id"].tolist(),
        "lengths": group["length_m"].round(3).tolist()
    })

print("\n========== INSPECTION COMPLETED ==========")
