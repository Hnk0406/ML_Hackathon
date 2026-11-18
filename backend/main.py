"""
main.py - FastAPI backend for PathSnap
Runs on Laptop 1. Hosts model, HMM/Viterbi, evaluation.
"""
import os
import time
import logging
import numpy as np
import folium
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from data_loader import load_trip, get_trip_ids
from road_network import build_graph, generate_all_candidates, haversine
from hmm_viterbi import viterbi_decode, evaluate_route
import joblib

# --- Config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_BASE = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_BASE, "train")
STUDENT_TEST_DIR = os.path.join(DATA_BASE, "student_test")
EVALUATOR_TEST_DIR = os.path.join(DATA_BASE, "evaluator_test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# HMM parameters
SIGMA = 50.0        # emission noise (meters)
BETA = 20.0         # transition smoothness
CANDIDATE_K = 10    # number of candidate arcs per GPS point
CANDIDATE_RADIUS = 300  # max distance in meters

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap")

# --- App ---
app = FastAPI(title="PathSnap API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cache for loaded data ---
trip_cache = {}
graph_cache = {}
loaded_model = None


def load_best_ml_model():
    """Attempt to load trained ML model from models/ directory."""
    global loaded_model
    if loaded_model is not None:
        return loaded_model

    candidate_files = ["best_model.pkl", "best_model.joblib", "model.pkl", "model.joblib", "rf_model.pkl", "xgb_model.pkl"]
    for fname in candidate_files:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                loaded_model = joblib.load(path)
                logger.info(f"Loaded trained ML model from {path}")
                return loaded_model
            except Exception as e:
                logger.warning(f"Could not load model at {path}: {e}")
    return None


def compute_ml_candidate_probs(model, candidates, track, nodes, arcs):
    """Computes ML probabilities for (observation, candidate) pairs."""
    if model is None:
        return None

    features = []
    keys = []
    for obs_idx, cands in enumerate(candidates):
        for arc_id, dist_m, proj_t in cands:
            fn, tn = arcs[arc_id]
            arc_len = haversine(nodes[fn, 0], nodes[fn, 1], nodes[tn, 0], nodes[tn, 1])
            d_lon = nodes[tn, 0] - nodes[fn, 0]
            d_lat = nodes[tn, 1] - nodes[fn, 1]
            arc_heading = np.degrees(np.arctan2(d_lon, d_lat)) % 360.0

            if obs_idx > 0:
                gps_dlon = track[obs_idx, 0] - track[obs_idx - 1, 0]
                gps_dlat = track[obs_idx, 1] - track[obs_idx - 1, 1]
                gps_heading = np.degrees(np.arctan2(gps_dlon, gps_dlat)) % 360.0
                dt = max(track[obs_idx, 2] - track[obs_idx - 1, 2], 1.0)
                speed = haversine(track[obs_idx - 1, 0], track[obs_idx - 1, 1],
                                  track[obs_idx, 0], track[obs_idx, 1]) / dt
            else:
                gps_heading = arc_heading
                speed = 0.0

            heading_diff = abs(arc_heading - gps_heading)
            if heading_diff > 180.0:
                heading_diff = 360.0 - heading_diff

            features.append([dist_m, arc_len, heading_diff, proj_t, speed])
            keys.append((obs_idx, arc_id))

    if not features:
        return {}

    X = np.array(features)
    try:
        probs = model.predict_proba(X)
        scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    except Exception:
        scores = model.predict(X)

    return {k: float(s) for k, s in zip(keys, scores)}


def align_gps_to_route_arcs(track, nodes, arcs, route_arcs):
    """Map each GPS point monotonically to its closest arc along the ground-truth route."""
    if len(route_arcs) == len(track):
        return [int(a) for a in route_arcs]

    route_mids = np.array([
        (nodes[arcs[a, 0]] + nodes[arcs[a, 1]]) / 2.0 for a in route_arcs
    ])
    assigned = []
    curr_idx = 0
    window = max(15, len(route_arcs) // 5)
    for i in range(len(track)):
        end_idx = min(len(route_arcs), curr_idx + window)
        sub_mids = route_mids[curr_idx:end_idx]
        d = (sub_mids[:, 0] - track[i, 0]) ** 2 + (sub_mids[:, 1] - track[i, 1]) ** 2
        best_offset = int(np.argmin(d))
        curr_idx = curr_idx + best_offset
        assigned.append(int(route_arcs[curr_idx]))
    return assigned


def get_trip_dir(trip_id):
    """Find which partition a trip belongs to (evaluator_test, student_test, or train)."""
    for partition in [EVALUATOR_TEST_DIR, STUDENT_TEST_DIR, TRAIN_DIR]:
        if os.path.isdir(partition):
            trip_path = os.path.join(partition, trip_id)
            if os.path.isdir(trip_path):
                return trip_path
    return None


def load_and_cache_trip(trip_id):
    """Load trip data with caching."""
    if trip_id in trip_cache:
        return trip_cache[trip_id], graph_cache[trip_id]

    trip_dir = get_trip_dir(trip_id)
    if trip_dir is None:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    logger.info(f"Loading trip {trip_id} from {trip_dir}...")
    trip = load_trip(trip_dir, trip_id)

    logger.info(f"Building graph ({len(trip['nodes'])} nodes, {len(trip['arcs'])} arcs)...")
    graph = build_graph(trip["nodes"], trip["arcs"])

    trip_cache[trip_id] = trip
    graph_cache[trip_id] = graph
    return trip, graph


def make_folium_map(trip, predicted_route, ground_truth=None):
    """Generate a Folium map with GPS points, predicted route, and ground truth."""
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    center_lat = np.mean(track[:, 1])
    center_lon = np.mean(track[:, 0])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # GPS points (blue) - downsample if track is very large for fast rendering
    gps_group = folium.FeatureGroup(name="GPS Points")
    step = max(1, len(track) // 600)
    for i in range(0, len(track), step):
        folium.CircleMarker(
            location=[track[i, 1], track[i, 0]],
            radius=3, color="blue", fill=True, fill_opacity=0.6,
            popup=f"GPS[{i}] ({track[i,0]:.5f}, {track[i,1]:.5f})"
        ).add_to(gps_group)
    gps_group.add_to(m)

    # Predicted route (red)
    if predicted_route:
        pred_group = folium.FeatureGroup(name="Predicted Route")
        pred_coords = []
        for arc_id in predicted_route:
            if 0 <= arc_id < len(arcs):
                from_node = arcs[arc_id][0]
                to_node = arcs[arc_id][1]
                from_coord = [nodes[from_node][1], nodes[from_node][0]]
                to_coord = [nodes[to_node][1], nodes[to_node][0]]
                pred_coords.append(from_coord)
                pred_coords.append(to_coord)
        if pred_coords:
            folium.PolyLine(pred_coords, color="red", weight=4, opacity=0.8).add_to(pred_group)
        pred_group.add_to(m)

    # Ground truth route (green)
    if ground_truth is not None:
        gt_group = folium.FeatureGroup(name="Ground Truth")
        gt_coords = []
        for arc_id in ground_truth:
            if 0 <= arc_id < len(arcs):
                from_node = arcs[arc_id][0]
                to_node = arcs[arc_id][1]
                from_coord = [nodes[from_node][1], nodes[from_node][0]]
                to_coord = [nodes[to_node][1], nodes[to_node][0]]
                gt_coords.append(from_coord)
                gt_coords.append(to_coord)
        if gt_coords:
            folium.PolyLine(gt_coords, color="green", weight=3, opacity=0.7).add_to(gt_group)
        gt_group.add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


# =====================
# API ENDPOINTS
# =====================

@app.get("/")
def root():
    return {"status": "PathSnap API is running", "version": "1.0"}


@app.get("/trips")
def list_trips(partition: str = "evaluator_test"):
    """List available trips for a given partition (evaluator_test, student_test, or train)."""
    mapping = {
        "evaluator_test": EVALUATOR_TEST_DIR,
        "student_test": STUDENT_TEST_DIR,
        "train": TRAIN_DIR,
    }
    target_dir = mapping.get(partition, EVALUATOR_TEST_DIR)
    partition_name = partition if partition in mapping else "evaluator_test"

    trip_ids = []
    if os.path.isdir(target_dir):
        for tid in sorted(os.listdir(target_dir)):
            if os.path.isdir(os.path.join(target_dir, tid)):
                trip_ids.append(tid)
    logger.info(f"Listed {len(trip_ids)} trips from {partition_name}")
    return {"trips": trip_ids, "count": len(trip_ids), "partition": partition_name}


@app.get("/predict/{trip_id}")
def predict_route(trip_id: str):
    """
    Full prediction pipeline for a trip:
    1. Load trip data
    2. Generate candidates
    3. Run HMM/Viterbi decoding
    4. Evaluate against ground truth (server-side)
    5. Return predicted route + metrics + map
    """
    total_start = time.time()
    logger.info(f"=== PREDICT REQUEST: trip {trip_id} ===")

    # 1. Load data
    trip, graph = load_and_cache_trip(trip_id)
    logger.info(f"Trip loaded: {len(trip['track'])} GPS points")

    # 2. Generate candidates
    t0 = time.time()
    candidates = generate_all_candidates(
        trip["track"], trip["nodes"], trip["arcs"],
        k=CANDIDATE_K, radius_m=CANDIDATE_RADIUS
    )
    cand_time = time.time() - t0
    logger.info(f"Candidates generated in {cand_time:.2f}s")

    # 3. Check ML model & calculate probabilities if available
    ml_model = load_best_ml_model()
    ml_probs = None
    use_model = False
    if ml_model is not None:
        ml_probs = compute_ml_candidate_probs(ml_model, candidates, trip["track"], trip["nodes"], trip["arcs"])
        use_model = True if ml_probs else False
        logger.info(f"Scored candidates with ML model ({len(ml_probs)} candidates scored)")

    # 4. Viterbi decode
    t0 = time.time()
    predicted_route = viterbi_decode(
        candidates, trip["track"], trip["nodes"], trip["arcs"], graph,
        sigma=SIGMA, beta=BETA, use_model_probs=use_model, model_probs=ml_probs
    )
    viterbi_time = time.time() - t0
    logger.info(f"Viterbi decoded in {viterbi_time:.2f}s (use_ml={use_model})")

    # 4. Evaluate (server-side only - ground truth never sent to client)
    metrics = {}
    ground_truth = None
    if trip["route"] is not None:
        ground_truth = trip["route"].tolist()
        aligned_gt = align_gps_to_route_arcs(trip["track"], trip["nodes"], trip["arcs"], trip["route"])
        metrics = evaluate_route(predicted_route, aligned_gt)
        logger.info(f"Evaluation: Top-1 accuracy={metrics['accuracy']}, "
                    f"matched={metrics['matched']}/{metrics['total']}, "
                    f"continuity={metrics['continuity']}")
    else:
        logger.info("No ground truth available for this trip")

    # 5. Generate Folium map
    t0 = time.time()
    map_html = make_folium_map(trip, predicted_route, ground_truth)
    map_time = time.time() - t0
    logger.info(f"Map generated in {map_time:.2f}s")

    total_time = time.time() - total_start
    logger.info(f"=== TOTAL TIME: {total_time:.2f}s ===")

    return {
        "trip_id": trip_id,
        "predicted_route": predicted_route,
        "num_gps_points": len(trip["track"]),
        "num_predicted_arcs": len(predicted_route),
        "metrics": metrics,
        "timing": {
            "candidate_generation_s": round(cand_time, 2),
            "viterbi_decoding_s": round(viterbi_time, 2),
            "map_generation_s": round(map_time, 2),
            "total_s": round(total_time, 2),
        },
        "map_html": map_html,
    }


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def get_map(trip_id: str):
    """Return just the Folium map HTML for a trip (for iframe embedding)."""
    result = predict_route(trip_id)
    return HTMLResponse(content=result["map_html"])


@app.get("/health")
def health():
    return {"status": "ok", "time": time.strftime("%H:%M:%S")}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PathSnap API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
