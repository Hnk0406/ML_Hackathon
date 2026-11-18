
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "outputs" / "processed" / "road_network"

results = []

for partition in ["train", "student_test"]:
    partition_dir = BASE_DIR / partition

    for trip_dir in sorted(partition_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

        original_nodes_file = trip_dir / f"{trip_id}.nodes"
        original_arcs_file = trip_dir / f"{trip_id}.arcs"

        processed_dir = PROCESSED_DIR / partition / trip_id

        processed_nodes_file = (
            processed_dir / f"{trip_id}_processed_nodes.csv"
        )

        processed_arcs_file = (
            processed_dir / f"{trip_id}_processed_arcs.csv"
        )

        if not processed_nodes_file.exists() or not processed_arcs_file.exists():
            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "status": "MISSING FILE"
            })
            continue

        original_nodes = pd.read_csv(
            original_nodes_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        original_arcs = pd.read_csv(
            original_arcs_file,
            sep=r"\s+",
            header=None,
            names=["from_node_id", "to_node_id"]
        )

        processed_nodes = pd.read_csv(processed_nodes_file)
        processed_arcs = pd.read_csv(processed_arcs_file)

        node_count_match = len(original_nodes) == len(processed_nodes)
        arc_count_match = len(original_arcs) == len(processed_arcs)

        node_ids_match = np.array_equal(
            processed_nodes["node_id"].to_numpy(),
            np.arange(len(original_nodes))
        )

        node_coordinates_match = np.allclose(
            original_nodes[["longitude", "latitude"]].to_numpy(),
            processed_nodes[["longitude", "latitude"]].to_numpy()
        )

        arc_ids_match = np.array_equal(
            processed_arcs["arc_id"].to_numpy(),
            np.arange(len(original_arcs))
        )

        connections_match = np.array_equal(
            original_arcs[["from_node_id", "to_node_id"]].to_numpy(),
            processed_arcs[["from_node_id", "to_node_id"]].to_numpy()
        )

        lengths_valid = (
            np.isfinite(processed_arcs["length_m"]).all()
            and (processed_arcs["length_m"] >= 0).all()
        )

        passed = all([
            node_count_match,
            arc_count_match,
            node_ids_match,
            node_coordinates_match,
            arc_ids_match,
            connections_match,
            lengths_valid
        ])

        results.append({
            "partition": partition,
            "trip_id": trip_id,
            "node_count_match": node_count_match,
            "arc_count_match": arc_count_match,
            "node_ids_match": node_ids_match,
            "node_coordinates_match": node_coordinates_match,
            "arc_ids_match": arc_ids_match,
            "connections_match": connections_match,
            "lengths_valid": lengths_valid,
            "status": "PASS" if passed else "FAIL"
        })

report = pd.DataFrame(results)

report_dir = BASE_DIR / "outputs" / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

report.to_csv(
    report_dir / "road_network_validation.csv",
    index=False
)

print("========== ROAD NETWORK VALIDATION ==========")
print("Total trips checked:", len(report))
print("\nValidation results:")
print(report["status"].value_counts())

print("\nFailed trips:")
print(report[report["status"] != "PASS"].to_string(index=False))

print("\nAll trips passed:", (report["status"] == "PASS").all())
print("Validation report saved successfully!")
