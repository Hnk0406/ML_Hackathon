"""
data_loader.py - Load and parse PathSnap raw data files
"""
import numpy as np
import os


def load_track(filepath):
    """Load .track file -> (N, 3) array: lon, lat, timestamp"""
    data = np.loadtxt(filepath)
    return data  # columns: lon, lat, timestamp


def load_nodes(filepath):
    """Load .nodes file -> (M, 2) array: lon, lat. Row index = node_id"""
    data = np.loadtxt(filepath)
    return data  # columns: lon, lat


def load_arcs(filepath):
    """Load .arcs file -> (A, 2) int array: from_node, to_node. Row index = arc_id"""
    data = np.loadtxt(filepath, dtype=int)
    return data  # columns: from_node_id, to_node_id


def load_route(filepath):
    """Load .route file -> 1D int array of ground-truth arc_ids in order"""
    data = np.loadtxt(filepath, dtype=int)
    return data


def load_trip(trip_dir, trip_id):
    """Load all files for a single trip. Returns dict."""
    prefix = os.path.join(trip_dir, trip_id)
    trip = {
        "trip_id": trip_id,
        "track": load_track(f"{prefix}.track"),
        "nodes": load_nodes(f"{prefix}.nodes"),
        "arcs": load_arcs(f"{prefix}.arcs"),
    }
    route_path = f"{prefix}.route"
    if os.path.exists(route_path):
        trip["route"] = load_route(route_path)
    else:
        trip["route"] = None
    return trip


def get_trip_ids(partition_dir):
    """Get sorted list of trip IDs from a partition folder."""
    ids = sorted(set(
        f.split(".")[0] for f in os.listdir(partition_dir) if "." in f
    ))
    if not ids:
        ids = sorted([
            d for d in os.listdir(partition_dir)
            if os.path.isdir(os.path.join(partition_dir, d))
        ])
    return ids


if __name__ == "__main__":
    # Quick test
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    train_dir = os.path.join(base, "train", "00000001")
    trip = load_trip(train_dir, "00000001")
    print(f"Track shape: {trip['track'].shape}")
    print(f"Nodes shape: {trip['nodes'].shape}")
    print(f"Arcs shape: {trip['arcs'].shape}")
    print(f"Route shape: {trip['route'].shape}")
    print(f"First 3 GPS points:\n{trip['track'][:3]}")
    print(f"First 3 nodes:\n{trip['nodes'][:3]}")
    print(f"First 3 arcs:\n{trip['arcs'][:3]}")
    print(f"First 10 route arcs: {trip['route'][:10]}")
    print("Data loader OK!")
