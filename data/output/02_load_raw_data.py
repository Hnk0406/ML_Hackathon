
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

TRIP_ID = "00000001"
PARTITION = "train"

TRIP_DIR = BASE_DIR / PARTITION / TRIP_ID

track_file = TRIP_DIR / f"{TRIP_ID}.track"
nodes_file = TRIP_DIR / f"{TRIP_ID}.nodes"
arcs_file = TRIP_DIR / f"{TRIP_ID}.arcs"
route_file = TRIP_DIR / f"{TRIP_ID}.route"

track = pd.read_csv(
    track_file,
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

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

route = pd.read_csv(
    route_file,
    sep=r"\s+",
    header=None,
    names=["arc_id"]
)

datasets = {
    "GPS Track": track,
    "Road Nodes": nodes,
    "Road Arcs": arcs,
    "Reference Route": route
}

for name, df in datasets.items():

    print(f"\n========== {name} ==========")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

print("\nAll four datasets loaded successfully!")
