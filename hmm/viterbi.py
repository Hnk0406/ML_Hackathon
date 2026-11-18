"""
viterbi.py - HMM-based Route Reconstruction using ML Classifier Probabilities
Author: Person 3 (HMM / Viterbi & Backend)

Runs standalone via:
    python hmm/viterbi.py
"""

import os
import sys
import time
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd
import networkx as nx
import folium
import joblib

# Paths setup (relative to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
STUDENT_TEST_DIR = os.path.join(DATA_DIR, "student_test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_IMG_DIR = os.path.join(PROJECT_ROOT, "reports", "images")
os.makedirs(REPORTS_IMG_DIR, exist_ok=True)


# ==========================================
# 1. DISTANCE & SPATIAL UTILITIES
# ==========================================

def haversine(lon1, lat1, lon2, lat2):
    """Haversine distance in meters between two GPS coordinates."""
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    return R * 2.0 * atan2(sqrt(a), sqrt(max(0.0, 1.0 - a)))


def haversine_vec(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance for fast candidate filtering."""
    R = 6371000.0
    lon1, lat1, lon2, lat2 = [np.radians(x) for x in [lon1, lat1, lon2, lat2]]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Computes orthogonal distance and projection fraction t from P to segment AB."""
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


# ==========================================
# 2. DATA & ROAD NETWORK LOADERS
# ==========================================

def load_trip(trip_dir, trip_id):
    prefix = os.path.join(trip_dir, trip_id)
    return {
        "trip_id": trip_id,
        "track": np.loadtxt(f"{prefix}.track"),
        "nodes": np.loadtxt(f"{prefix}.nodes"),
        "arcs": np.loadtxt(f"{prefix}.arcs", dtype=int),
        "route": np.loadtxt(f"{prefix}.route", dtype=int) if os.path.exists(f"{prefix}.route") else None,
    }


def build_graph(nodes, arcs):
    """Build directed road network graph."""
    G = nx.DiGraph()
    for nid in range(len(nodes)):
        G.add_node(nid, lon=nodes[nid, 0], lat=nodes[nid, 1])

    for aid in range(len(arcs)):
        fn, tn = arcs[aid]
        length = haversine(nodes[fn, 0], nodes[fn, 1], nodes[tn, 0], nodes[tn, 1])
        G.add_edge(fn, tn, arc_id=aid, length=length)
    return G


def build_arc_midpoints(nodes, arcs):
    return (nodes[arcs[:, 0]] + nodes[arcs[:, 1]]) / 2.0


def find_candidates(gps_lon, gps_lat, nodes, arcs, midpoints, k=10, radius_m=300):
    """Returns top-k candidate arcs sorted by distance."""
    mid_dists = haversine_vec(gps_lon, gps_lat, midpoints[:, 0], midpoints[:, 1])
    pre_k = min(k * 4, len(arcs))
    candidate_indices = np.argpartition(mid_dists, pre_k)[:pre_k]

    results = []
    for aid in candidate_indices:
        fn, tn = arcs[aid]
        dist, t = point_to_segment_distance(
            gps_lon, gps_lat, nodes[fn, 0], nodes[fn, 1], nodes[tn, 0], nodes[tn, 1]
        )
        results.append((int(aid), dist, t))

    results.sort(key=lambda x: x[1])
    filtered = [r for r in results[:k] if r[1] <= radius_m]
    return filtered if filtered else results[:1]


def generate_all_candidates(track, nodes, arcs, k=10, radius_m=300):
    midpoints = build_arc_midpoints(nodes, arcs)
    return [
        find_candidates(track[i, 0], track[i, 1], nodes, arcs, midpoints, k=k, radius_m=radius_m)
        for i in range(len(track))
    ]


# ==========================================
# 3. FEATURE EXTRACTION FOR ML CLASSIFIER
# ==========================================

def extract_candidate_features(track, obs_idx, arc_id, dist_m, proj_t, nodes, arcs):
    """
    Constructs feature vector for candidate arc matching Person 1's features:
    1. Distance from GPS observation to arc (meters)
    2. Arc length (meters)
    3. Heading / bearing difference
    4. Projection fraction t
    5. Observation speed (m/s)
    """
    fn, tn = arcs[arc_id]
    arc_len = haversine(nodes[fn, 0], nodes[fn, 1], nodes[tn, 0], nodes[tn, 1])

    # Arc bearing
    d_lon = nodes[tn, 0] - nodes[fn, 0]
    d_lat = nodes[tn, 1] - nodes[fn, 1]
    arc_heading = np.degrees(np.arctan2(d_lon, d_lat)) % 360.0

    # GPS movement heading
    if obs_idx > 0:
        gps_dlon = track[obs_idx, 0] - track[obs_idx - 1, 0]
        gps_dlat = track[obs_idx, 1] - track[obs_idx - 1, 1]
        gps_heading = np.degrees(np.arctan2(gps_dlon, gps_dlat)) % 360.0
        time_diff = max(track[obs_idx, 2] - track[obs_idx - 1, 2], 1.0)
        gps_dist = haversine(track[obs_idx - 1, 0], track[obs_idx - 1, 1],
                             track[obs_idx, 0], track[obs_idx, 1])
        speed = gps_dist / time_diff
    else:
        gps_heading = arc_heading
        speed = 0.0

    heading_diff = abs(arc_heading - gps_heading)
    if heading_diff > 180.0:
        heading_diff = 360.0 - heading_diff

    return [dist_m, arc_len, heading_diff, proj_t, speed]


def load_best_ml_model():
    """Attempts to load Person 2's best model from `models/`."""
    possible_names = ["best_model.pkl", "best_model.joblib", "model.pkl", "model.joblib", "rf_model.pkl", "xgb_model.pkl"]
    for name in possible_names:
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                print(f"[+] Loaded ML model from: {p}")
                return model
            except Exception as e:
                print(f"[!] Found model at {p} but failed loading: {e}")
    return None


def get_ml_candidate_probabilities(model, candidates_per_obs, track, nodes, arcs):
    if model is None:
        return None

    features_list = []
    keys = []

    for obs_idx, cands in enumerate(candidates_per_obs):
        for arc_id, dist_m, proj_t in cands:
            feats = extract_candidate_features(track, obs_idx, arc_id, dist_m, proj_t, nodes, arcs)
            features_list.append(feats)
            keys.append((obs_idx, arc_id))

    if not features_list:
        return {}

    X = np.array(features_list)
    try:
        probs = model.predict_proba(X)
        scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    except Exception:
        scores = model.predict(X)

    return {k: float(s) for k, s in zip(keys, scores)}


# ==========================================
# 4. HMM & VITERBI DECODING ENGINE
# ==========================================

def emission_prob(distance_m, sigma=50.0):
    """Gaussian emission probability."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * (distance_m / sigma) ** 2)


def transition_prob(route_dist, gc_dist, beta=20.0):
    """Exponential transition probability."""
    dt = abs(route_dist - gc_dist)
    return (1.0 / beta) * np.exp(-dt / beta)


def compute_route_distance(arc_from, arc_to, arcs, graph, penalty=2000.0):
    end_node = arcs[arc_from][1]
    start_node = arcs[arc_to][0]

    if end_node == start_node or arc_from == arc_to:
        return 0.0

    try:
        path = nx.shortest_path(graph, end_node, start_node, weight="length")
        return sum(graph[path[i]][path[i + 1]]["length"] for i in range(len(path) - 1))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return penalty


def viterbi_decode(candidates_per_obs, track, nodes, arcs, graph,
                   sigma=50.0, beta=20.0, ml_probs=None, ml_weight=1.5):
    """
    Viterbi route decoding with integrated ML classifier emission probabilities:
        log P(Emission) = log P_spatial(dist) + ml_weight * log P_ml(arc | features)
    """
    T = len(candidates_per_obs)
    if T == 0:
        return []

    valid_obs = [(i, c) for i, c in enumerate(candidates_per_obs) if len(c) > 0]
    if not valid_obs:
        return [0] * T

    V = [{} for _ in range(len(valid_obs))]
    backpointer = [{} for _ in range(len(valid_obs))]

    def calc_emission_log(obs_i, arc_i, dist_m):
        log_e = np.log(max(emission_prob(dist_m, sigma), 1e-300))
        if ml_probs and (obs_i, arc_i) in ml_probs:
            p = max(ml_probs[(obs_i, arc_i)], 1e-4)
            log_e += ml_weight * np.log(p)
        return log_e

    # 1. Initialization (t=0)
    idx0, cands0 = valid_obs[0]
    for arc_id, dist_m, _ in cands0:
        V[0][arc_id] = calc_emission_log(idx0, arc_id, dist_m)
        backpointer[0][arc_id] = None

    # 2. Recurrence
    for t in range(1, len(valid_obs)):
        curr_idx, curr_cands = valid_obs[t]
        prev_idx, prev_cands = valid_obs[t - 1]

        gc_dist = haversine(
            track[prev_idx, 0], track[prev_idx, 1],
            track[curr_idx, 0], track[curr_idx, 1]
        )

        for arc_curr, dist_curr, _ in curr_cands:
            log_e = calc_emission_log(curr_idx, arc_curr, dist_curr)
            best_val = -np.inf
            best_prev = None

            for arc_prev, _, _ in prev_cands:
                if arc_prev not in V[t - 1]:
                    continue
                rd = compute_route_distance(arc_prev, arc_curr, arcs, graph)
                log_t = np.log(max(transition_prob(rd, gc_dist, beta), 1e-300))
                prob = V[t - 1][arc_prev] + log_t + log_e
                if prob > best_val:
                    best_val = prob
                    best_prev = arc_prev

            if best_prev is not None:
                V[t][arc_curr] = best_val
                backpointer[t][arc_curr] = best_prev

    # 3. Backtrack
    if not V[-1]:
        return [c[0][0] if c else 0 for c in candidates_per_obs]

    best_last = max(V[-1], key=V[-1].get)
    decoded = [best_last]
    for t in range(len(valid_obs) - 1, 0, -1):
        prev = backpointer[t].get(decoded[-1])
        if prev is None:
            prev = valid_obs[t - 1][1][0][0] if valid_obs[t - 1][1] else 0
        decoded.append(prev)
    decoded.reverse()

    # 4. Map back
    result = [0] * T
    valid_idxs = [i for i, _ in valid_obs]
    for i, vi in enumerate(valid_idxs):
        result[vi] = decoded[i]

    for i in range(T):
        if not candidates_per_obs[i]:
            closest = min(valid_idxs, key=lambda j: abs(j - i))
            result[i] = result[closest]

    return result


def evaluate_route(predicted_route, ground_truth_route):
    n = min(len(predicted_route), len(ground_truth_route))
    if n == 0:
        return {"accuracy": 0.0, "matched": 0, "total": 0, "continuity": 0.0}

    pred = predicted_route[:n]
    gt = ground_truth_route[:n]
    matches = sum(1 for p, g in zip(pred, gt) if p == g)
    accuracy = matches / n

    contig = sum(1 for i in range(1, len(pred)) if pred[i] == pred[i - 1])
    continuity = contig / max(len(pred) - 1, 1)

    return {
        "accuracy": round(accuracy, 4),
        "matched": matches,
        "total": n,
        "continuity": round(continuity, 4)
    }


# ==========================================
# 5. EXPERIMENTATION & COMPARISON RUNNER
# ==========================================

def run_evaluation_pipeline(trip_id="00000001", num_obs=100):
    print("=" * 70)
    print("PATHSNAP: END-TO-END ML + HMM ROUTE RECONSTRUCTION (P3)")
    print("=" * 70)

    # 1. Load Data
    trip_dir = os.path.join(TRAIN_DIR, trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = os.path.join(STUDENT_TEST_DIR, trip_id)

    print(f"\n[1] Loading Trip: {trip_id} from {trip_dir}")
    trip = load_trip(trip_dir, trip_id)
    eval_track = trip["track"][:num_obs]

    # 2. Road Network Graph
    print("\n[2] Constructing Road Graph...")
    graph = build_graph(trip["nodes"], trip["arcs"])

    # 3. Candidate Generation
    print(f"\n[3] Generating Candidates for {num_obs} GPS Points (k=10, radius=300m)...")
    candidates = generate_all_candidates(eval_track, trip["nodes"], trip["arcs"], k=10, radius_m=300)

    # 4. Check for ML model
    print("\n[4] Checking for Person 2's Trained ML Model in models/...")
    ml_model = load_best_ml_model()
    ml_probs = None
    if ml_model is not None:
        print("    -> ML Model found! Extracting features & calculating candidate probabilities...")
        ml_probs = get_ml_candidate_probabilities(ml_model, candidates, eval_track, trip["nodes"], trip["arcs"])
    else:
        print("    [Note] Person 2's model not in models/ yet. Running spatial baseline + spatial HMM.")

    # 5. Model Comparisons (Required for Report Section 5)
    print("\n[5] Executing 3-Way Model Comparison...")

    # Method A: Closest Arc (Spatial Baseline)
    pred_closest = [c[0][0] if c else 0 for c in candidates]

    # Method B: ML Classifier-Only (Argmax of ML predictions)
    if ml_probs:
        pred_ml_only = []
        for i, cands in enumerate(candidates):
            best_arc = max(cands, key=lambda c: ml_probs.get((i, c[0]), -1))[0]
            pred_ml_only.append(best_arc)
    else:
        pred_ml_only = pred_closest

    # Method C: Full System (ML Classifier + HMM / Viterbi)
    t0 = time.time()
    pred_hmm = viterbi_decode(
        candidates, eval_track, trip["nodes"], trip["arcs"], graph,
        sigma=50.0, beta=20.0, ml_probs=ml_probs
    )
    hmm_time = time.time() - t0

    # 6. Evaluation Table Output
    if trip["route"] is not None:
        gt = trip["route"][:num_obs].tolist()
        m_closest = evaluate_route(pred_closest, gt)
        m_ml = evaluate_route(pred_ml_only, gt)
        m_hmm = evaluate_route(pred_hmm, gt)

        print("\n" + "=" * 70)
        print("REPORT SECTION 5: MODEL EXPERIMENTATION & COMPARISON TABLE")
        print("=" * 70)
        print(f"{'Method / Configuration':<32} | {'Top-1 Accuracy':<15} | {'Continuity':<12}")
        print("-" * 70)
        print(f"{'1. Baseline: Nearest Road Arc':<32} | {m_closest['accuracy']:<15.4f} | {m_closest['continuity']:<12.4f}")
        if ml_probs:
            print(f"{'2. ML Classifier-Only (Argmax)':<32} | {m_ml['accuracy']:<15.4f} | {m_ml['continuity']:<12.4f}")
        print(f"{'3. PathSnap: ML + HMM Viterbi':<32} | {m_hmm['accuracy']:<15.4f} | {m_hmm['continuity']:<12.4f}")
        print("-" * 70)
        print(f"Decoding Time: {hmm_time:.2f}s for {num_obs} observations")
        print("=" * 70)

    # 7. Generate Folium Map for reports
    out_map = os.path.join(REPORTS_IMG_DIR, f"trip_{trip_id}_route_reconstruction.html")
    m = folium.Map(location=[np.mean(eval_track[:, 1]), np.mean(eval_track[:, 0])], zoom_start=15)

    gps_fg = folium.FeatureGroup(name="GPS (Blue)")
    for i in range(len(eval_track)):
        folium.CircleMarker([eval_track[i, 1], eval_track[i, 0]], radius=3, color="blue", fill=True).add_to(gps_fg)
    gps_fg.add_to(m)

    pred_fg = folium.FeatureGroup(name="ML + HMM Route (Red)")
    for aid in pred_hmm:
        if 0 <= aid < len(trip["arcs"]):
            fn, tn = trip["arcs"][aid]
            folium.PolyLine(
                [[trip["nodes"][fn, 1], trip["nodes"][fn, 0]], [trip["nodes"][tn, 1], trip["nodes"][tn, 0]]],
                color="red", weight=4, opacity=0.85
            ).add_to(pred_fg)
    pred_fg.add_to(m)

    if trip["route"] is not None:
        gt_fg = folium.FeatureGroup(name="Ground Truth (Green)")
        for aid in trip["route"][:num_obs]:
            if 0 <= aid < len(trip["arcs"]):
                fn, tn = trip["arcs"][aid]
                folium.PolyLine(
                    [[trip["nodes"][fn, 1], trip["nodes"][fn, 0]], [trip["nodes"][tn, 1], trip["nodes"][tn, 0]]],
                    color="green", weight=2.5, opacity=0.75, dash_array="5, 5"
                ).add_to(gt_fg)
        gt_fg.add_to(m)

    folium.LayerControl().add_to(m)
    try:
        m.save(out_map)
        print(f"\n[✔] Map saved to: {out_map}")
    except Exception as e:
        print(f"\n[!] Note: Could not save HTML map ({e}). Proceeding.")


if __name__ == "__main__":
    run_evaluation_pipeline()
