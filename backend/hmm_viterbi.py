"""
hmm_viterbi.py - HMM-based route reconstruction using Viterbi decoding
"""
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import networkx as nx


def haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in meters."""
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def emission_prob(distance_m, sigma=50.0):
    """
    Emission probability: how likely a GPS observation is given a candidate road segment.
    Uses Gaussian model on distance.
    sigma controls GPS noise tolerance (meters).
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance_m / sigma) ** 2)


def transition_prob(route_dist, great_circle_dist, beta=20.0):
    """
    Transition probability between consecutive candidate arcs.
    Based on difference between route distance and great-circle distance.
    Uses exponential distribution.
    beta controls tolerance for detours.
    """
    dt = abs(route_dist - great_circle_dist)
    return (1.0 / beta) * np.exp(-dt / beta)


def get_arc_center(arc_id, nodes, arcs):
    """Get the midpoint of an arc in (lon, lat)."""
    from_node, to_node = arcs[arc_id]
    from_coords = nodes[from_node]
    to_coords = nodes[to_node]
    return (from_coords + to_coords) / 2.0


def compute_route_distance(arc_id_from, arc_id_to, nodes, arcs, graph, max_hops=8):
    """
    Approximate route distance between two arcs using shortest path on the graph.
    Returns distance in meters. If no path found, returns a large penalty.
    """
    to_node_of_from_arc = arcs[arc_id_from][1]  # end node of first arc
    from_node_of_to_arc = arcs[arc_id_to][0]    # start node of second arc

    # Same node - arcs are directly connected
    if to_node_of_from_arc == from_node_of_to_arc:
        return 0.0

    # Same arc
    if arc_id_from == arc_id_to:
        return 0.0

    try:
        path = nx.shortest_path(
            graph, to_node_of_from_arc, from_node_of_to_arc, weight="length"
        )
        dist = sum(
            graph[path[i]][path[i+1]]["length"]
            for i in range(len(path) - 1)
        )
        return dist
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 2000.0  # large penalty for disconnected segments


def viterbi_decode(candidates_per_obs, track, nodes, arcs, graph,
                   sigma=50.0, beta=20.0, use_model_probs=False, model_probs=None):
    """
    Viterbi algorithm for HMM map-matching.

    Parameters:
    - candidates_per_obs: list of lists, each inner list = [(arc_id, dist_m, proj_t), ...]
    - track: (N, 3) array of GPS points (lon, lat, timestamp)
    - nodes, arcs: road network data
    - graph: NetworkX directed graph
    - sigma: emission noise parameter
    - beta: transition smoothness parameter
    - use_model_probs: if True, incorporate ML model probabilities
    - model_probs: dict mapping (obs_idx, arc_id) -> probability from classifier

    Returns:
    - best_route: list of arc_ids, one per GPS observation
    """
    T = len(candidates_per_obs)
    if T == 0:
        return []

    # Filter out observations with no candidates
    valid_obs = [(i, cands) for i, cands in enumerate(candidates_per_obs) if len(cands) > 0]
    if not valid_obs:
        return [0] * T

    # Initialize Viterbi tables
    # V[t] = dict: arc_id -> log probability
    # backpointer[t] = dict: arc_id -> previous arc_id
    V = [{} for _ in range(len(valid_obs))]
    backpointer = [{} for _ in range(len(valid_obs))]

    # === Initialization (t=0) ===
    t0_idx, t0_cands = valid_obs[0]
    for arc_id, dist_m, proj_t in t0_cands:
        emit = emission_prob(dist_m, sigma)
        log_emit = np.log(max(emit, 1e-300))

        if use_model_probs and model_probs and (t0_idx, arc_id) in model_probs:
            ml_prob = model_probs[(t0_idx, arc_id)]
            log_emit += np.log(max(ml_prob, 1e-300))

        V[0][arc_id] = log_emit
        backpointer[0][arc_id] = None

    # === Recursion ===
    for t in range(1, len(valid_obs)):
        curr_obs_idx, curr_cands = valid_obs[t]
        prev_obs_idx, prev_cands = valid_obs[t - 1]

        # Great-circle distance between consecutive GPS points
        gc_dist = haversine(
            track[prev_obs_idx, 0], track[prev_obs_idx, 1],
            track[curr_obs_idx, 0], track[curr_obs_idx, 1]
        )

        for arc_id_curr, dist_curr, proj_t_curr in curr_cands:
            best_log_prob = -np.inf
            best_prev_arc = None

            emit = emission_prob(dist_curr, sigma)
            log_emit = np.log(max(emit, 1e-300))

            if use_model_probs and model_probs and (curr_obs_idx, arc_id_curr) in model_probs:
                ml_prob = model_probs[(curr_obs_idx, arc_id_curr)]
                log_emit += np.log(max(ml_prob, 1e-300))

            for arc_id_prev, dist_prev, proj_t_prev in prev_cands:
                if arc_id_prev not in V[t - 1]:
                    continue

                # Compute route distance between arcs
                route_dist = compute_route_distance(
                    arc_id_prev, arc_id_curr, nodes, arcs, graph, max_hops=8
                )
                trans = transition_prob(route_dist, gc_dist, beta)
                log_trans = np.log(max(trans, 1e-300))

                log_prob = V[t - 1][arc_id_prev] + log_trans + log_emit

                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_prev_arc = arc_id_prev

            if best_prev_arc is not None:
                V[t][arc_id_curr] = best_log_prob
                backpointer[t][arc_id_curr] = best_prev_arc

    # === Termination: backtrack ===
    # Find best final state
    last_V = V[-1]
    if not last_V:
        # Fallback: use closest candidate for each observation
        result = []
        for i in range(T):
            if candidates_per_obs[i]:
                result.append(candidates_per_obs[i][0][0])
            else:
                result.append(0)
        return result

    best_last_arc = max(last_V, key=last_V.get)

    # Backtrack
    decoded_arcs_valid = [best_last_arc]
    for t in range(len(valid_obs) - 1, 0, -1):
        prev_arc = backpointer[t].get(decoded_arcs_valid[-1])
        if prev_arc is None:
            # Fallback
            prev_cands = valid_obs[t - 1][1]
            prev_arc = prev_cands[0][0] if prev_cands else 0
        decoded_arcs_valid.append(prev_arc)
    decoded_arcs_valid.reverse()

    # Map back to full observation indices
    result = [0] * T
    valid_indices = [idx for idx, _ in valid_obs]
    for i, obs_idx in enumerate(valid_indices):
        result[obs_idx] = decoded_arcs_valid[i]

    # Fill gaps (observations with no candidates) with nearest valid prediction
    for i in range(T):
        if not candidates_per_obs[i]:
            # Find nearest valid observation
            best_dist = float("inf")
            best_arc = 0
            for j in valid_indices:
                if abs(j - i) < best_dist:
                    best_dist = abs(j - i)
                    best_arc = result[j]
            result[i] = best_arc

    return result


def evaluate_route(predicted_route, ground_truth_route):
    """
    Compare predicted route with ground truth.
    Returns accuracy metrics.
    """
    min_len = min(len(predicted_route), len(ground_truth_route))
    if min_len == 0:
        return {"accuracy": 0.0, "matched": 0, "total": 0, "continuity": 0.0}

    pred = predicted_route[:min_len]
    gt = ground_truth_route[:min_len]

    matches = sum(1 for p, g in zip(pred, gt) if p == g)
    accuracy = matches / min_len

    # Route continuity: fraction of consecutive predictions that share a node
    continuity_count = 0
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            continuity_count += 1

    continuity = continuity_count / max(len(pred) - 1, 1)

    return {
        "accuracy": round(accuracy, 4),
        "matched": matches,
        "total": min_len,
        "continuity": round(continuity, 4),
        "pred_length": len(predicted_route),
        "gt_length": len(ground_truth_route),
    }


if __name__ == "__main__":
    from data_loader import load_trip
    from road_network import build_graph, generate_all_candidates
    import os
    import time

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    trip = load_trip(os.path.join(base, "train", "00000001"), "00000001")

    print("Building graph...")
    graph = build_graph(trip["nodes"], trip["arcs"])

    # Use first 50 GPS points for quick test
    n_test = 50
    test_track = trip["track"][:n_test]

    print(f"Generating candidates for {n_test} GPS points...")
    t0 = time.time()
    candidates = generate_all_candidates(test_track, trip["nodes"], trip["arcs"], k=8, radius_m=300)
    print(f"Candidates: {time.time()-t0:.2f}s")

    print("Running Viterbi...")
    t0 = time.time()
    predicted = viterbi_decode(candidates, test_track, trip["nodes"], trip["arcs"], graph)
    print(f"Viterbi: {time.time()-t0:.2f}s")

    if trip["route"] is not None:
        gt = trip["route"][:n_test].tolist()
        metrics = evaluate_route(predicted, gt)
        print(f"\nResults: {metrics}")
    else:
        print(f"Predicted first 10 arcs: {predicted[:10]}")

    print("HMM Viterbi module OK!")
