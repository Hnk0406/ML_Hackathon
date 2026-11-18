import os
import sys
import json
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("backend"))
from data_loader import load_trip, get_trip_ids
from road_network import generate_all_candidates, haversine

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

DATA_BASE = "data"
TRAIN_DIR = os.path.join(DATA_BASE, "train")
STUDENT_TEST_DIR = os.path.join(DATA_BASE, "student_test")
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = ["dist_m", "arc_len", "heading_diff", "proj_t", "speed"]


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


def build_dataset(partition_dir, max_obs_per_trip=250, is_train=True):
    """Extract candidate features matching backend/main.py compute_ml_candidate_probs()."""
    rows = []
    trip_ids = get_trip_ids(partition_dir)
    for trip_id in trip_ids:
        trip = load_trip(os.path.join(partition_dir, trip_id), trip_id)
        track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
        if route is None or len(route) == 0:
            continue

        gt_per_obs = align_gps_to_route_arcs(track, nodes, arcs, route)
        step = max(1, len(track) // max_obs_per_trip) if is_train else 1
        obs_indices = list(range(0, len(track), step))
        sub_track = track[obs_indices]
        candidates = generate_all_candidates(sub_track, nodes, arcs, k=10, radius_m=300)

        for local_i, orig_obs_idx in enumerate(obs_indices):
            gt_arc = gt_per_obs[orig_obs_idx]
            cands = candidates[local_i]
            cand_arc_ids = [c[0] for c in cands]

            if is_train and gt_arc not in cand_arc_ids:
                fn, tn = arcs[gt_arc]
                mid_lon = (nodes[fn, 0] + nodes[tn, 0]) / 2.0
                mid_lat = (nodes[fn, 1] + nodes[tn, 1]) / 2.0
                d_gt = haversine(track[orig_obs_idx, 0], track[orig_obs_idx, 1], mid_lon, mid_lat)
                cands = cands[:-1] + [(int(gt_arc), d_gt, 0.5)]

            for arc_id, dist_m, proj_t in cands:
                fn, tn = arcs[arc_id]
                arc_len = haversine(nodes[fn, 0], nodes[fn, 1], nodes[tn, 0], nodes[tn, 1])
                d_lon = nodes[tn, 0] - nodes[fn, 0]
                d_lat = nodes[tn, 1] - nodes[fn, 1]
                arc_heading = np.degrees(np.arctan2(d_lon, d_lat)) % 360.0

                if orig_obs_idx > 0:
                    gps_dlon = track[orig_obs_idx, 0] - track[orig_obs_idx - 1, 0]
                    gps_dlat = track[orig_obs_idx, 1] - track[orig_obs_idx - 1, 1]
                    gps_heading = np.degrees(np.arctan2(gps_dlon, gps_dlat)) % 360.0
                    dt = max(track[orig_obs_idx, 2] - track[orig_obs_idx - 1, 2], 1.0)
                    speed = haversine(
                        track[orig_obs_idx - 1, 0], track[orig_obs_idx - 1, 1],
                        track[orig_obs_idx, 0], track[orig_obs_idx, 1]
                    ) / dt
                else:
                    gps_heading = arc_heading
                    speed = 0.0

                heading_diff = abs(arc_heading - gps_heading)
                if heading_diff > 180.0:
                    heading_diff = 360.0 - heading_diff

                rows.append({
                    "trip_id": trip_id,
                    "obs_idx": orig_obs_idx,
                    "candidate_arc_id": int(arc_id),
                    "from_node_id": int(fn),
                    "to_node_id": int(tn),
                    "dist_m": float(dist_m),
                    "arc_len": float(arc_len),
                    "heading_diff": float(heading_diff),
                    "proj_t": float(proj_t),
                    "speed": float(speed),
                    "label": 1 if int(arc_id) == int(gt_arc) else 0,
                })
    return pd.DataFrame(rows)


def evaluate_predictions(df, probs):
    """Compute candidate-level and route-level metrics required by Report Section 3.1."""
    eval_df = df.copy()
    eval_df["prob"] = probs
    eval_df["pred"] = (probs >= 0.5).astype(int)

    macro_f1 = f1_score(eval_df["label"], eval_df["pred"], average="macro")
    prec = precision_score(eval_df["label"], eval_df["pred"], average="macro", zero_division=0)
    rec = recall_score(eval_df["label"], eval_df["pred"], average="macro", zero_division=0)
    ll = log_loss(eval_df["label"], np.clip(probs, 1e-6, 1 - 1e-6))

    # Observation-level Top-1 Accuracy
    top1_idx = eval_df.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    top1_df = eval_df.loc[top1_idx].sort_values(["trip_id", "obs_idx"])
    top1_acc = float((top1_df["label"] == 1).mean())

    # Route Continuity (% of consecutive Top-1 predicted arcs connected in road network)
    connected = 0
    total_transitions = 0
    for _, trip_group in top1_df.groupby("trip_id"):
        arcs_seq = trip_group["candidate_arc_id"].values
        from_nodes = trip_group["from_node_id"].values
        to_nodes = trip_group["to_node_id"].values
        for i in range(len(arcs_seq) - 1):
            total_transitions += 1
            if arcs_seq[i] == arcs_seq[i + 1] or to_nodes[i] == from_nodes[i + 1]:
                connected += 1
    continuity = connected / max(total_transitions, 1)

    return macro_f1, prec, rec, top1_acc, continuity, ll


def main():
    print("Extracting features from 80 training trips and 10 student-test trips...")
    train_df = build_dataset(TRAIN_DIR, max_obs_per_trip=250, is_train=True)
    test_df = build_dataset(STUDENT_TEST_DIR, max_obs_per_trip=10000, is_train=False)
    print(f"Train rows: {len(train_df)} | Student-Test rows: {len(test_df)}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    groups = train_df["trip_id"].values
    X_test = test_df[FEATURE_COLS].values

    models = {
        "Logistic Regression": ("Linear/Probabilistic", "C=1.0, balanced", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])),
        "Distance-Weighted KNN": ("Instance-Based", "k=15, distance weights", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1))
        ])),
        "Random Forest": ("Ensemble (Bagging)", "n_estimators=120, max_depth=12", RandomForestClassifier(
            n_estimators=120, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
        )),
        "LightGBM": ("Boosting", "n_estimators=150, lr=0.05", lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
        )),
        "MLP Neural Net": ("Neural Network", "layers=(64,32), early_stopping", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, early_stopping=True, random_state=42))
        ])),
    }

    gkf = GroupKFold(n_splits=5)
    results = []
    mlp_loss_curve = None
    n_test_trips = test_df["trip_id"].nunique()

    for name, (family, cfg, model) in models.items():
        print(f"\n--- Evaluating {name} ---")
        cv_f1s = []
        for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
            model.fit(X_train[tr_idx], y_train[tr_idx])
            va_preds = model.predict(X_train[va_idx])
            cv_f1s.append(f1_score(y_train[va_idx], va_preds, average="macro"))

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        if name == "MLP Neural Net":
            mlp_loss_curve = model.named_steps["clf"].loss_curve_

        t0 = time.perf_counter()
        test_probs = model.predict_proba(X_test)[:, 1]
        inf_time_ms = ((time.perf_counter() - t0) / max(n_test_trips, 1)) * 1000.0

        test_f1, prec, rec, top1_acc, continuity, ll = evaluate_predictions(test_df, test_probs)

        results.append({
            "Model": name,
            "Algorithm Family": family,
            "Key Configuration": cfg,
            "CV Macro F1-score": round(float(np.mean(cv_f1s)), 4),
            "CV Macro F1 Std": round(float(np.std(cv_f1s)), 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Route Continuity": round(continuity, 4),
            "Student-Test Macro F1-score": round(test_f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Log Loss": round(ll, 4),
            "Train Time (s)": round(train_time, 3),
            "Inference Time (ms/trip)": round(inf_time_ms, 3),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\n=== OVERALL MODEL COMPARISON (REPORT SECTION 3.1) ===")
    print(results_df.to_string(index=False))

    # Bayesian Hyperparameter Optimization (Optuna) on LightGBM
    print("\n=== RUNNING OPTUNA BAYESIAN OPTIMIZATION (25 TRIALS) ===")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 350),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[va_idx])
            fold_scores.append(f1_score(y_train[va_idx], preds, average="macro"))
        return float(np.mean(fold_scores))

    t0_opt = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=25)
    opt_time = time.perf_counter() - t0_opt

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(RESULTS_DIR, "bayesian_trials.csv"), index=False)

    best_params = {
        **study.best_params,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    tuned_model = lgb.LGBMClassifier(**best_params)
    tuned_model.fit(X_train, y_train)

    # Save best model directly to models/best_model.joblib for backend/main.py
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(tuned_model, model_path)
    print(f"Saved tuned model to {model_path}")

    t0 = time.perf_counter()
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]
    tuned_inf_ms = ((time.perf_counter() - t0) / max(n_test_trips, 1)) * 1000.0
    t_f1, _, _, t_top1, t_cont, t_ll = evaluate_predictions(test_df, tuned_probs)

    untuned_row = results_df[results_df["Model"] == "LightGBM"].iloc[0]
    tuning_comparison = pd.DataFrame([
        {
            "Version": "Untuned LightGBM",
            "CV Macro F1-score": untuned_row["CV Macro F1-score"],
            "Student-Test Top-1 Accuracy": untuned_row["Top-1 Accuracy"],
            "Student-Test Macro F1-score": untuned_row["Student-Test Macro F1-score"],
            "Log Loss": untuned_row["Log Loss"],
            "Inference Time (ms/trip)": untuned_row["Inference Time (ms/trip)"],
        },
        {
            "Version": "Tuned LightGBM (Optuna)",
            "CV Macro F1-score": round(float(study.best_value), 4),
            "Student-Test Top-1 Accuracy": round(t_top1, 4),
            "Student-Test Macro F1-score": round(t_f1, 4),
            "Log Loss": round(t_ll, 4),
            "Inference Time (ms/trip)": round(tuned_inf_ms, 3),
        },
    ])
    tuning_comparison.to_csv(os.path.join(RESULTS_DIR, "tuning_comparison.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump({"best_params": study.best_params, "best_cv_macro_f1": study.best_value, "optimization_time_s": round(opt_time, 2)}, f, indent=2)

    print("\n=== UNTUNED VS TUNED COMPARISON (REPORT SECTION 4) ===")
    print(tuning_comparison.to_string(index=False))

    # Generate Compulsory Section 3.3 & Section 4 Plots
    # 1. CV Macro F1 Comparison
    plt.figure(figsize=(8, 4.5))
    plt.bar(results_df["Model"], results_df["CV Macro F1-score"], yerr=results_df["CV Macro F1 Std"], capsize=4, color="#2b6cb0")
    plt.title("5-Fold GroupKFold CV Macro F1-Score Comparison")
    plt.ylabel("CV Macro F1-Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cv_macro_f1_comparison.png"), dpi=200)
    plt.close()

    # 2. Student-Test Top-1 Accuracy vs Macro F1
    x = np.arange(len(results_df))
    width = 0.35
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, results_df["Top-1 Accuracy"], width, label="Top-1 Accuracy", color="#38a169")
    plt.bar(x + width / 2, results_df["Student-Test Macro F1-score"], width, label="Student-Test Macro F1", color="#dd6b20")
    plt.xticks(x, results_df["Model"], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Student-Testing Top-1 Accuracy & Macro F1 Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "student_test_comparison.png"), dpi=200)
    plt.close()

    # 3. Log Loss Comparison
    plt.figure(figsize=(8, 4.5))
    plt.bar(results_df["Model"], results_df["Log Loss"], color="#805ad5")
    plt.title("Student-Testing Log Loss Comparison (Lower is Better)")
    plt.ylabel("Log Loss")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "log_loss_comparison.png"), dpi=200)
    plt.close()

    # 4. MLP Epoch-Wise Loss Curve (genuine iterative history only)
    if mlp_loss_curve:
        plt.figure(figsize=(6.5, 4))
        plt.plot(range(1, len(mlp_loss_curve) + 1), mlp_loss_curve, marker="o", markersize=3, color="#d53f8c")
        plt.title("MLP Neural Network Epoch-Wise Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "mlp_training_loss.png"), dpi=200)
        plt.close()

    # 5. Optuna Trial History
    plt.figure(figsize=(7, 4))
    plt.plot(trials_df["number"] + 1, trials_df["value"], marker="o", linestyle="-", color="#319795", label="Trial CV Macro F1")
    plt.plot(trials_df["number"] + 1, trials_df["value"].cummax(), linestyle="--", color="#e53e3e", label="Best So Far")
    plt.title("Bayesian Hyperparameter Optimization History (Optuna TPE)")
    plt.xlabel("Trial Number")
    plt.ylabel("5-Fold GroupKFold Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "optuna_history.png"), dpi=200)
    plt.close()

    print("\nALL DONE! Artifacts saved to models/best_model.joblib and results/.")


if __name__ == "__main__":
    main()