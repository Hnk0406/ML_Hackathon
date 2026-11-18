import os, sys, json, time, joblib, optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath("models"))
from train_and_tune import build_dataset, evaluate_predictions, TRAIN_DIR, STUDENT_TEST_DIR, FEATURE_COLS, MODELS_DIR, RESULTS_DIR

print("Loading P1 preprocessed train & student-test features...")
train_df = build_dataset(TRAIN_DIR, max_obs_per_trip=250, is_train=True)
test_df = build_dataset(STUDENT_TEST_DIR, max_obs_per_trip=10000, is_train=False)

X_train, y_train, groups = train_df[FEATURE_COLS].values, train_df["label"].values, train_df["trip_id"].values
X_test = test_df[FEATURE_COLS].values
n_test_trips = test_df["trip_id"].nunique()
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_train, y_train, groups))

optuna.logging.set_verbosity(optuna.logging.WARNING)
ARCH_CHOICES = {"(64, 32)": (64, 32), "(96, 48)": (96, 48), "(128, 64)": (128, 64)}

def eval_fold(tr_idx, va_idx, arch, alpha, lr, batch_size):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=arch,
            alpha=alpha,
            learning_rate_init=lr,
            batch_size=batch_size,
            max_iter=45,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42
        ))
    ])
    pipe.fit(X_train[tr_idx], y_train[tr_idx])
    return float(f1_score(y_train[va_idx], pipe.predict(X_train[va_idx]), average="macro"))

def objective(trial):
    arch_key = trial.suggest_categorical("hidden_layers", list(ARCH_CHOICES.keys()))
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    lr = trial.suggest_float("learning_rate_init", 1e-3, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    scores = Parallel(n_jobs=-1)(
        delayed(eval_fold)(tr, va, ARCH_CHOICES[arch_key], alpha, lr, batch_size) for tr, va in splits
    )
    return float(np.mean(scores))

print("Running 8-trial 5-Fold GroupKFold Bayesian Optimization on MLP (with batch_size)...")
t0_opt = time.perf_counter()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=8)
opt_time = time.perf_counter() - t0_opt

trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(RESULTS_DIR, "bayesian_trials.csv"), index=False)

bp = study.best_params
best_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=ARCH_CHOICES[bp["hidden_layers"]],
        alpha=bp["alpha"],
        learning_rate_init=bp["learning_rate_init"],
        batch_size=bp["batch_size"],
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=42
    ))
])
best_mlp.fit(X_train, y_train)
joblib.dump(best_mlp, os.path.join(MODELS_DIR, "best_model.joblib"))

plt.figure(figsize=(6.5, 4))
plt.plot(range(1, len(best_mlp.named_steps["clf"].loss_curve_) + 1), best_mlp.named_steps["clf"].loss_curve_, marker="o", markersize=3, color="#d53f8c")
plt.title("Tuned MLP Epoch-Wise Training Loss (P1 Preprocessed Data)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mlp_training_loss.png"), dpi=200)
plt.close()

t0 = time.perf_counter()
tuned_probs = best_mlp.predict_proba(X_test)[:, 1]
tuned_inf_ms = ((time.perf_counter() - t0) / max(n_test_trips, 1)) * 1000.0
t_f1, _, _, t_top1, _, t_ll = evaluate_predictions(test_df, tuned_probs)

comp_df = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
untuned_mlp = comp_df[comp_df["Model"] == "MLP Neural Net"].iloc[0]

tuning_comparison = pd.DataFrame([
    {
        "Version": "Untuned MLP Neural Net",
        "CV Macro F1-score": untuned_mlp["CV Macro F1-score"],
        "Student-Test Top-1 Accuracy": untuned_mlp["Top-1 Accuracy"],
        "Student-Test Macro F1-score": untuned_mlp["Student-Test Macro F1-score"],
        "Log Loss": untuned_mlp["Log Loss"],
        "Inference Time (ms/trip)": untuned_mlp["Inference Time (ms/trip)"],
    },
    {
        "Version": "Tuned MLP Neural Net (Optuna)",
        "CV Macro F1-score": round(float(study.best_value), 4),
        "Student-Test Top-1 Accuracy": round(t_top1, 4),
        "Student-Test Macro F1-score": round(t_f1, 4),
        "Log Loss": round(t_ll, 4),
        "Inference Time (ms/trip)": round(tuned_inf_ms, 3),
    },
])
tuning_comparison.to_csv(os.path.join(RESULTS_DIR, "tuning_comparison.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "best_hyperparameters.json"), "w") as f:
    json.dump({"best_params": bp, "best_cv_macro_f1": study.best_value, "optimization_time_s": round(opt_time, 2)}, f, indent=2)

plt.figure(figsize=(7, 4))
plt.plot(trials_df["number"] + 1, trials_df["value"], marker="o", linestyle="-", color="#319795", label="Trial CV Macro F1")
plt.plot(trials_df["number"] + 1, trials_df["value"].cummax(), linestyle="--", color="#e53e3e", label="Best So Far")
plt.title("Bayesian Hyperparameter Optimization History (Optuna on MLP)")
plt.xlabel("Trial Number")
plt.ylabel("5-Fold GroupKFold Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "optuna_history.png"), dpi=200)
plt.close()

print("\n=== BEST HYPERPARAMETERS ===")
print(json.dumps(bp, indent=2))
print("\n=== UNTUNED VS TUNED MLP COMPARISON ===")
print(tuning_comparison.to_string(index=False))
