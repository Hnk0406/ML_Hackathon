
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "processed" / "road_network"

EARTH_RADIUS = 6371000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

summary = []

for partition in ["train", "student_test"]:
    partition_dir = BASE_DIR / partition

    for trip_dir in sorted(partition_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

        nodes_file = trip_dir / f"{trip_id}.nodes"
        arcs_file = trip_dir / f"{trip_id}.arcs"

        if not nodes_file.exists() or not arcs_file.exists():
            continue

        nodes = pd.read_csv(
            nodes_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        arcs = pd.read_csv(
            arcs_file,
            sep=r"\s+",
            header=None,
            names=["from_node_id", "to_node_id"]
        )

        arcs["arc_id"] = np.arange(len(arcs))

        valid_connections = (
            arcs["from_node_id"].between(0, len(nodes) - 1)
            & arcs["to_node_id"].between(0, len(nodes) - 1)
        )

        arcs["valid_connection"] = valid_connections

        lengths = np.full(len(arcs), np.nan)

        valid_arcs = arcs.loc[valid_connections]

        if len(valid_arcs) > 0:
            from_nodes = nodes.iloc[
                valid_arcs["from_node_id"].to_numpy()
            ]

            to_nodes = nodes.iloc[
                valid_arcs["to_node_id"].to_numpy()
            ]

            lengths[valid_connections.to_numpy()] = haversine_distance(
                from_nodes["longitude"].to_numpy(),
                from_nodes["latitude"].to_numpy(),
                to_nodes["longitude"].to_numpy(),
                to_nodes["latitude"].to_numpy()
            )

        arcs["length_m"] = lengths

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
            & arcs["valid_connection"]
        )

        trip_output = OUTPUT_DIR / partition / trip_id
        trip_output.mkdir(parents=True, exist_ok=True)

        nodes_output = nodes.copy()
        nodes_output.insert(0, "node_id", np.arange(len(nodes)))

        nodes_output.to_csv(
            trip_output / f"{trip_id}_processed_nodes.csv",
            index=False
        )

        arcs.to_csv(
            trip_output / f"{trip_id}_processed_arcs.csv",
            index=False
        )

        summary.append({
            "partition": partition,
            "trip_id": trip_id,
            "total_nodes": len(nodes),
            "total_arcs": len(arcs),
            "invalid_connections": int((~valid_connections).sum()),
            "self_loops": int(arcs["self_loop"].sum()),
            "duplicate_directed_arcs": int(
                arcs["duplicate_directed_arc"].sum()
            ),
            "zero_length_arcs": int(
                (arcs["length_m"] == 0).sum()
            ),
            "valid_arcs": int(arcs["valid_length"].sum())
        })

summary_df = pd.DataFrame(summary)

REPORT_DIR = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

summary_df.to_csv(
    REPORT_DIR / "all_road_networks_summary.csv",
    index=False
)

print("========== ALL ROAD NETWORKS PREPROCESSING ==========")
print("Trips processed:", len(summary_df))
print("Training trips:", (summary_df["partition"] == "train").sum())
print("Student testing trips:", (summary_df["partition"] == "student_test").sum())
print("Total nodes:", summary_df["total_nodes"].sum())
print("Total arcs:", summary_df["total_arcs"].sum())
print("Invalid connections:", summary_df["invalid_connections"].sum())
print("Self-loops:", summary_df["self_loops"].sum())
print("Duplicate directed arcs:", summary_df["duplicate_directed_arcs"].sum())
print("Zero-length arcs:", summary_df["zero_length_arcs"].sum())
print("Valid arcs:", summary_df["valid_arcs"].sum())

print("\nSummary saved successfully!")
print("Processed road networks saved at:", OUTPUT_DIR)
