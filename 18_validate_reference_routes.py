
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "outputs" / "processed" / "road_network"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = []

for partition in ["train", "student_test"]:
    partition_dir = BASE_DIR / partition

    for trip_dir in sorted(partition_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        trip_id = trip_dir.name

        route_file = trip_dir / f"{trip_id}.route"
        original_arcs_file = trip_dir / f"{trip_id}.arcs"

        processed_arcs_file = (
            PROCESSED_DIR / partition / trip_id /
            f"{trip_id}_processed_arcs.csv"
        )

        if not all([
            route_file.exists(),
            original_arcs_file.exists(),
            processed_arcs_file.exists()
        ]):
            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "status": "MISSING FILE"
            })
            continue

        route = pd.read_csv(
            route_file,
            sep=r"\s+",
            header=None,
            names=["arc_id"]
        )

        original_arcs = pd.read_csv(
            original_arcs_file,
            sep=r"\s+",
            header=None,
            names=["from_node_id", "to_node_id"]
        )

        processed_arcs = pd.read_csv(processed_arcs_file)

        route_ids = route["arc_id"].to_numpy()

        valid_ids = (
            (route_ids >= 0) &
            (route_ids < len(original_arcs))
        )

        route_ids_valid = bool(valid_ids.all())

        route_sequence_preserved = (
            len(route) > 0
            and np.array_equal(
                route_ids,
                route["arc_id"].to_numpy()
            )
        )

        if route_ids_valid:
            route_connections = original_arcs.iloc[
                route_ids
            ].reset_index(drop=True)

            if len(route_connections) > 1:
                transitions_connected = (
                    route_connections["to_node_id"].iloc[:-1].to_numpy()
                    ==
                    route_connections["from_node_id"].iloc[1:].to_numpy()
                )

                disconnected_transitions = int(
                    (~transitions_connected).sum()
                )
            else:
                disconnected_transitions = 0
        else:
            disconnected_transitions = np.nan

        processed_arc_ids = set(
            processed_arcs["arc_id"].to_numpy()
        )

        processed_ids_exist = all(
            int(arc_id) in processed_arc_ids
            for arc_id in route_ids
        )

        passed = (
            route_ids_valid
            and route_sequence_preserved
            and processed_ids_exist
        )

        results.append({
            "partition": partition,
            "trip_id": trip_id,
            "route_entries": len(route),
            "invalid_route_ids": int((~valid_ids).sum()),
            "route_sequence_preserved": route_sequence_preserved,
            "processed_arc_ids_exist": processed_ids_exist,
            "disconnected_transitions": disconnected_transitions,
            "status": "PASS" if passed else "FAIL"
        })

report = pd.DataFrame(results)

report.to_csv(
    REPORT_DIR / "reference_route_validation.csv",
    index=False
)

print("========== REFERENCE ROUTE VALIDATION ==========")
print("Total trips checked:", len(report))

print("\nValidation results:")
print(report["status"].value_counts())

print("\nTotal route entries:", 
      report["route_entries"].sum())

print("Invalid route IDs:",
      report["invalid_route_ids"].sum())

print("Disconnected consecutive transitions:",
      report["disconnected_transitions"].sum())

print("\nFailed trips:")
print(report[report["status"] != "PASS"].to_string(index=False))

print("\nAll trips passed:", (report["status"] == "PASS").all())

print("\nValidation report saved:")
print(REPORT_DIR / "reference_route_validation.csv")
