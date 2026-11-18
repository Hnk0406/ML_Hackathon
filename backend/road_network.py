"""
road_network.py - Build road network graph and spatial index for candidate generation
"""
import numpy as np
import networkx as nx
from math import radians, sin, cos, sqrt, atan2


def haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in meters between two points."""
    R = 6371000  # Earth radius in meters
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def haversine_vec(lon1, lat1, lon2, lat2):
    """Vectorized haversine for arrays."""
    R = 6371000
    lon1, lat1, lon2, lat2 = [np.radians(x) for x in [lon1, lat1, lon2, lat2]]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    Approximate distance from point (px,py) to line segment (ax,ay)-(bx,by).
    Uses projection in lon/lat space then converts to meters via haversine.
    Returns distance in meters and the projection parameter t.
    """
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return haversine(px, py, ax, ay), 0.0

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj_lon = ax + t * dx
    proj_lat = ay + t * dy
    dist = haversine(px, py, proj_lon, proj_lat)
    return dist, t


def build_graph(nodes, arcs):
    """Build a NetworkX directed graph from nodes and arcs arrays."""
    G = nx.DiGraph()
    for node_id in range(len(nodes)):
        G.add_node(node_id, lon=nodes[node_id, 0], lat=nodes[node_id, 1])

    for arc_id in range(len(arcs)):
        from_node, to_node = arcs[arc_id]
        from_lon, from_lat = nodes[from_node]
        to_lon, to_lat = nodes[to_node]
        length = haversine(from_lon, from_lat, to_lon, to_lat)
        G.add_edge(from_node, to_node, arc_id=arc_id, length=length)

    return G


def build_arc_midpoints(nodes, arcs):
    """Compute midpoint of each arc for fast spatial lookup. Returns (A, 2) array."""
    from_nodes = nodes[arcs[:, 0]]
    to_nodes = nodes[arcs[:, 1]]
    midpoints = (from_nodes + to_nodes) / 2.0
    return midpoints  # (num_arcs, 2) -> lon, lat


def find_candidate_arcs(gps_lon, gps_lat, nodes, arcs, arc_midpoints, k=10, radius_m=200):
    """
    Find the k nearest candidate arcs for a GPS point.
    Uses midpoint pre-filter then exact point-to-segment distance.
    Returns list of (arc_id, distance_m, proj_t).
    """
    # Vectorized distance from GPS point to all arc midpoints
    dists_to_mid = haversine_vec(
        gps_lon, gps_lat,
        arc_midpoints[:, 0], arc_midpoints[:, 1]
    )

    # Pre-filter: take top k*3 by midpoint distance
    pre_k = min(k * 3, len(arcs))
    candidate_indices = np.argpartition(dists_to_mid, pre_k)[:pre_k]

    # Exact point-to-segment distance
    results = []
    for arc_id in candidate_indices:
        from_node, to_node = arcs[arc_id]
        from_lon, from_lat = nodes[from_node]
        to_lon, to_lat = nodes[to_node]
        dist, t = point_to_segment_distance(
            gps_lon, gps_lat, from_lon, from_lat, to_lon, to_lat
        )
        results.append((int(arc_id), dist, t))

    # Sort by distance and take top k
    results.sort(key=lambda x: x[1])
    results = results[:k]

    # Also filter by radius
    results = [r for r in results if r[1] <= radius_m]

    # If nothing within radius, keep at least the closest one
    if not results:
        results.sort(key=lambda x: x[1])
        results = results[:1]

    return results


def generate_all_candidates(track, nodes, arcs, k=10, radius_m=200):
    """
    Generate candidates for all GPS points in a trip.
    Returns list of lists: candidates[gps_idx] = [(arc_id, dist_m, proj_t), ...]
    """
    arc_midpoints = build_arc_midpoints(nodes, arcs)
    all_candidates = []

    for i in range(len(track)):
        gps_lon, gps_lat = track[i, 0], track[i, 1]
        candidates = find_candidate_arcs(
            gps_lon, gps_lat, nodes, arcs, arc_midpoints, k=k, radius_m=radius_m
        )
        all_candidates.append(candidates)

    return all_candidates


if __name__ == "__main__":
    from data_loader import load_trip
    import os
    import time

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    trip = load_trip(os.path.join(base, "train", "00000001"), "00000001")

    print(f"Building graph for {len(trip['nodes'])} nodes, {len(trip['arcs'])} arcs...")
    t0 = time.time()
    G = build_graph(trip["nodes"], trip["arcs"])
    print(f"Graph built in {time.time()-t0:.2f}s")

    print(f"\nGenerating candidates for {len(trip['track'])} GPS points...")
    t0 = time.time()
    # Test with just first 20 GPS points for speed
    candidates = generate_all_candidates(
        trip["track"][:20], trip["nodes"], trip["arcs"], k=8
    )
    print(f"Candidates generated in {time.time()-t0:.2f}s")

    for i in range(min(5, len(candidates))):
        print(f"  GPS[{i}]: {len(candidates[i])} candidates, "
              f"closest arc={candidates[i][0][0]} dist={candidates[i][0][1]:.1f}m")

    # Check if ground truth arc is in candidates
    if trip["route"] is not None:
        hits = 0
        for i in range(min(20, len(candidates))):
            gt_arc = trip["route"][i] if i < len(trip["route"]) else -1
            cand_arcs = [c[0] for c in candidates[i]]
            if gt_arc in cand_arcs:
                hits += 1
        print(f"\nGround truth in candidates: {hits}/{min(20, len(candidates))}")

    print("Road network module OK!")
