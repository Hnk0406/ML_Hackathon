
from pathlib import Path
import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
TRIP_ID = "00000001"

TRIP_DIR = BASE_DIR / "train" / TRIP_ID

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

G = nx.DiGraph()

G.add_nodes_from(range(len(nodes)))

G.add_edges_from(
    arcs[["from_node_id", "to_node_id"]].itertuples(
        index=False,
        name=None
    )
)

components = list(nx.weakly_connected_components(G))

isolated_nodes = list(nx.isolates(G))

self_loops = list(nx.selfloop_edges(G))

duplicate_arcs = arcs.duplicated().sum()

component_sizes = sorted(
    [len(component) for component in components],
    reverse=True
)

print("\n========== ROAD NETWORK ANALYSIS ==========")

print("Total nodes:", len(nodes))
print("Total arcs:", len(arcs))

print("Unique connected nodes:", G.number_of_nodes())

print("Nodes with at least one connection:",
      sum(1 for node in G.nodes if G.degree(node) > 0))

print("Isolated nodes:", len(isolated_nodes))

print("Self-loops:", len(self_loops))

print("Duplicate directed arcs:", duplicate_arcs)

print("\n========== CONNECTED COMPONENTS ==========")

print("Number of weakly connected components:", len(components))

print("Largest component size:",
      component_sizes[0] if component_sizes else 0)

print("Smallest component size:",
      component_sizes[-1] if component_sizes else 0)

if components:

    largest_component = max(components, key=len)

    print("Largest component percentage:",
          round(len(largest_component) / len(nodes) * 100, 2), "%")

print("\n========== COMPONENT DISTRIBUTION ==========")

print("Top 10 component sizes:", component_sizes[:10])

OUTPUT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

component_report = pd.DataFrame({
    "component_id": range(len(component_sizes)),
    "number_of_nodes": component_sizes
})

component_report.to_csv(
    OUTPUT_DIR / "road_component_analysis.csv",
    index=False
)

print("\nRoad network analysis completed!")
print("Report saved successfully.")
