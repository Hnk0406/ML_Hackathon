
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "outputs" / "reports"

validation = pd.read_csv(
    REPORT_DIR / "reference_route_validation.csv",
    dtype={"trip_id": str}
)

disconnected = validation[
    validation["disconnected_transitions"] > 0
]

results = []

for _, row in disconnected.iterrows():
    partition = row["partition"]
    trip_id = row["trip_id"]

    trip_dir = BASE_DIR / partition / trip_id

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route_connections = arcs.iloc[
        route["arc_id"].to_numpy()
    ].reset_index(drop=True)

    for i in range(len(route_connections) - 1):
        current_to = int(
            route_connections.iloc[i]["to_node_id"]
        )

        next_from = int(
            route_connections.iloc[i + 1]["from_node_id"]
        )

        if current_to != next_from:
            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "transition_index": i,
                "current_arc_id": int(route.iloc[i]["arc_id"]),
                "next_arc_id": int(route.iloc[i + 1]["arc_id"]),
                "current_to_node": current_to,
                "next_from_node": next_from
            })

details = pd.DataFrame(results)

details.to_csv(
    REPORT_DIR / "disconnected_route_details.csv",
    index=False
)

print("========== DISCONNECTED ROUTE INSPECTION ==========")
print("Affected trips:", len(disconnected))
print("Disconnected transitions:", len(details))

if not details.empty:
    print("\nDisconnected transition details:")
    print(details.to_string(index=False))

print("\nReport saved successfully!")
