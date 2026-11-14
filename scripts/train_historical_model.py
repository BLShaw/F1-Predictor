"""Historical Model Training and Temporal Validation Pipeline.

Trains the production Random Forest regressor with temporal (season-aware) splitting,
statistically valid DNF handling, and atomic artifact replacement.
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, SEASONS_DIR, get_team_for_driver
from src.data_loader import build_weekend_features, load_gp_data
from src.model import F1MLPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")


def extract_classified_race_results(race_session: dict[str, Any] | None) -> dict[str, int]:
    """Extract finishing positions strictly for classified race finishers.

    Excludes mechanical DNFs and crash retirements to prevent target corruption.
    """
    if not race_session or "results" not in race_session:
        return {}

    classified_finishes: dict[str, int] = {}
    for entry in race_session["results"]:
        driver = entry.get("driver")
        pos = entry.get("position")
        status = str(entry.get("status", "")).strip()

        if not driver or pos is None:
            continue

        try:
            pos_int = int(float(pos))
        except (ValueError, TypeError):
            continue

        # Status indicating classified finisher (Finished or completed lap distance)
        is_classified = (
            status == "Finished"
            or status.startswith("+")
            or "Lap" in status
        )

        if is_classified:
            classified_finishes[driver] = pos_int

    return classified_finishes


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract feature vectors and target labels partitioned by temporal seasons.

    Train: 2022-2024 seasons (historical baseline)
    Validation: 2025 season (tuning and calibration)
    Test: 2026 season (unseen out-of-time evaluation)
    """
    predictor = F1MLPredictor(load_historical=False)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    if not SEASONS_DIR.exists():
        logger.error("Seasons data directory does not exist: %s", SEASONS_DIR)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    total_gps = 0
    driver_hist: dict[str, list[float]] = {}
    team_hist: dict[str, list[float]] = {}

    season_dirs = sorted(
        (d for d in SEASONS_DIR.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for year_dir in season_dirs:
        season_year = int(year_dir.name)
        gp_entries: list[tuple[int, Path]] = []
        for gp_dir in year_dir.iterdir():
            if not gp_dir.is_dir() or gp_dir.name.startswith("."):
                continue
            prefix = gp_dir.name.split("_", 1)[0]
            if prefix.isdigit():
                gp_entries.append((int(prefix), gp_dir))

        for _round_num, gp_dir in sorted(gp_entries, key=lambda x: x[0]):
            gp_data = load_gp_data(season_year, gp_dir.name)
            sessions = gp_data.get("sessions", {})
            race_session = sessions.get("race")
            if not race_session:
                continue

            classified_results = extract_classified_race_results(race_session)
            if not classified_results:
                continue

            driver_form_map = {
                drv: float(pd.Series(hist[-5:]).ewm(span=3).mean().iloc[-1])
                for drv, hist in driver_hist.items()
                if hist
            }
            team_form_map = {
                tm: float(pd.Series(hist[-10:]).ewm(span=5).mean().iloc[-1])
                for tm, hist in team_hist.items()
                if hist
            }

            # Construct grid-anchored pre-race features with strictly pre-race rolling form
            weekend_features = build_weekend_features(gp_data, rolling_form=(driver_form_map, team_form_map))
            if weekend_features.empty:
                continue

            # Engineer normalized ML features
            features_df, _X_matrix = predictor._engineer_features(weekend_features)

            for _idx, row in features_df.iterrows():
                driver = row["driver"]
                if driver in classified_results:
                    target_pos = classified_results[driver]
                    sample = {
                        "season": season_year,
                        "gp": gp_dir.name,
                        "driver": driver,
                        "target_position": target_pos,
                    }
                    for f_name in predictor.feature_names:
                        sample[f_name] = row[f_name]

                    if season_year <= 2024:
                        train_rows.append(sample)
                    elif season_year == 2025:
                        val_rows.append(sample)
                    else:
                        test_rows.append(sample)

            # Update rolling history strictly after processing this event
            raw_results = race_session.get("results", [])
            n_cars = max(len(raw_results), max(classified_results.values(), default=1), len(features_df), 1)
            driver_teams = {
                str(entry.get("driver", "")): str(entry.get("team", "")).strip()
                for entry in raw_results
                if entry.get("driver")
            }
            for drv, finish_pos in classified_results.items():
                norm_finish = (finish_pos - 1.0) / max(n_cars - 1, 1)
                driver_hist.setdefault(drv, []).append(norm_finish)
                team_name = driver_teams.get(drv) or get_team_for_driver(drv)
                if team_name:
                    team_hist.setdefault(team_name, []).append(norm_finish)

            total_gps += 1

    logger.info("Extracted features across %d Grand Prix events", total_gps)
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def evaluate_model_partition(
    model: RandomForestRegressor,
    df_partition: pd.DataFrame,
    feature_names: list[str],
    partition_name: str,
) -> dict[str, float]:
    """Compute statistically rigorous evaluation metrics on a dataset partition."""
    if df_partition.empty:
        logger.warning("Partition %s is empty; skipping evaluation.", partition_name)
        return {}

    X = df_partition[feature_names].values
    y_true = df_partition["target_position"].values
    y_pred = model.predict(X)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(model.score(X, y_true))

    max_pos = max(int(y_true.max()), 20)
    y_pred_rounded = np.clip(np.round(y_pred), 1, max_pos)

    grid_col = df_partition["grid_norm"].values * (max_pos - 1) + 1.0
    baseline_grid_mae = float(mean_absolute_error(y_true, grid_col))

    # Top-10 Points Finish Classification Metrics
    y_true_pts = (y_true <= 10).astype(int)
    y_pred_pts = (y_pred_rounded <= 10).astype(int)

    pts_acc = float(accuracy_score(y_true_pts, y_pred_pts))
    pts_f1 = float(f1_score(y_true_pts, y_pred_pts, zero_division=0))
    pts_prec = float(precision_score(y_true_pts, y_pred_pts, zero_division=0))
    pts_rec = float(recall_score(y_true_pts, y_pred_pts, zero_division=0))

    # Top-3 Podium Classification Metrics with Baseline Comparison
    y_true_podium = (y_true <= 3).astype(int)
    y_pred_podium = (y_pred_rounded <= 3).astype(int)

    podium_acc = float(accuracy_score(y_true_podium, y_pred_podium))
    podium_f1 = float(f1_score(y_true_podium, y_pred_podium, zero_division=0))
    podium_prec = float(precision_score(y_true_podium, y_pred_podium, zero_division=0))
    podium_rec = float(recall_score(y_true_podium, y_pred_podium, zero_division=0))

    # Baseline: Dummy classifier predicting 0 (No Podium)
    dummy_no_podium = np.zeros_like(y_true_podium)
    baseline_podium_acc = float(accuracy_score(y_true_podium, dummy_no_podium))

    print(f"\n--- {partition_name.upper()} EVALUATION METRICS (Samples: {len(df_partition)}) ---")
    print(f"R-squared Score:           {r2:.3f}")
    print(f"Mean Absolute Error:        {mae:.2f} positions (Grid-Only Baseline: {baseline_grid_mae:.2f})")
    print(f"Root Mean Squared Error:    {rmse:.2f} positions")
    print(
        f"Points Finish (Top 10) Acc: {pts_acc * 100:.1f}% (F1: {pts_f1:.3f}, Precision: {pts_prec:.3f}, Recall: {pts_rec:.3f})"
    )
    print(f"Podium (Top 3) Accuracy:    {podium_acc * 100:.1f}% (Baseline Dummy: {baseline_podium_acc * 100:.1f}%)")
    print(f"Podium Prediction F1:       {podium_f1:.3f} (Precision: {podium_prec:.3f}, Recall: {podium_rec:.3f})")
    print("-" * 65)

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "baseline_grid_mae": baseline_grid_mae,
        "points_acc": pts_acc,
        "points_f1": pts_f1,
        "podium_acc": podium_acc,
        "podium_f1": podium_f1,
        "baseline_podium_acc": baseline_podium_acc,
    }


def select_best_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Select Random Forest hyperparameters minimizing MAE on the 2025 validation partition."""
    best_params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_split": 8,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    if val_df.empty:
        return best_params

    X_val = val_df[feature_names].values
    y_val = val_df["target_position"].values
    best_mae = float("inf")

    for max_depth in (4, 5, 6):
        for min_samples_leaf in (2, 3, 4):
            for max_features in ("sqrt", 0.8):
                candidate_params: dict[str, Any] = {
                    "n_estimators": 300,
                    "max_depth": max_depth,
                    "min_samples_split": 8,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                candidate = RandomForestRegressor(**candidate_params)
                candidate.fit(X_train, y_train)
                val_mae = float(mean_absolute_error(y_val, candidate.predict(X_val)))
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_params = candidate_params

    logger.info(
        "Selected hyperparameters on 2025 validation set (MAE=%.4f): max_depth=%s, min_samples_leaf=%s, max_features=%s",
        best_mae,
        best_params["max_depth"],
        best_params["min_samples_leaf"],
        best_params["max_features"],
    )
    return best_params


def main() -> bool:
    """Execute historical training and validation pipeline."""
    logger.info("Starting F1 Predictor Model Training Pipeline")

    train_df, val_df, test_df = load_dataset()
    if train_df.empty:
        logger.error("No valid historical training data extracted.")
        return False

    feature_names = F1MLPredictor(load_historical=False).feature_names

    X_train = train_df[feature_names].values
    y_train = train_df["target_position"].values

    logger.info("Training set: %d samples from 2022-2024", len(X_train))
    logger.info("Validation set: %d samples from 2025", len(val_df))
    logger.info("Test set: %d samples from 2026", len(test_df))

    # Tune hyperparameters on 2025 out-of-time validation partition
    best_params = select_best_hyperparameters(X_train, y_train, val_df, feature_names)

    # 1. Fit on 2022-2024 and evaluate In-Sample & 2025 Validation
    val_model = RandomForestRegressor(**best_params)
    val_model.fit(X_train, y_train)
    train_metrics = evaluate_model_partition(val_model, train_df, feature_names, "Training Set (2022-2024 In-Sample)")
    val_metrics = evaluate_model_partition(val_model, val_df, feature_names, "Validation Set (2025 Out-of-Time)")

    # 2. Refit on 2022-2025 and evaluate on 2026 Out-of-Time Test
    train_val_df = pd.concat([df for df in (train_df, val_df) if not df.empty], ignore_index=True)
    test_eval_model = RandomForestRegressor(**best_params)
    test_eval_model.fit(train_val_df[feature_names].values, train_val_df["target_position"].values)
    test_metrics = evaluate_model_partition(test_eval_model, test_df, feature_names, "Test Set (2026 Out-of-Time)")

    # 3. Refit final production artifact on all completed historical seasons (2022-present)
    full_df = pd.concat([df for df in (train_df, val_df, test_df) if not df.empty], ignore_index=True)
    model = RandomForestRegressor(**best_params)
    model.fit(full_df[feature_names].values, full_df["target_position"].values)
    logger.info("Fitted final production model on %d total samples (2022-present)", len(full_df))

    # Model Artifact Validation
    test_sample = np.array(
        [
            [0.0, 0.0, 0.0, 0.1, 0.05, 0.05, 0.50, 0.0, 1.0, 0.0],
            [0.95, 0.90, 0.90, 0.5, 0.85, 0.85, 0.50, 0.0, 1.0, 0.0],
        ]
    )
    test_preds = model.predict(test_sample)

    if np.isnan(test_preds).any() or np.isinf(test_preds).any():
        logger.error("Model verification failed: predictions contain NaN or Inf values.")
        return False

    if test_preds[0] > test_preds[1]:
        logger.warning("Sanity warning: Pole position expected to predict lower rank than P20.")

    # Atomic artifact serialization
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temp_artifact_path = MODELS_DIR / "f1_historical_model.tmp.joblib"
    final_artifact_path = MODELS_DIR / "f1_historical_model.joblib"

    joblib.dump(model, temp_artifact_path)
    os.replace(temp_artifact_path, final_artifact_path)

    logger.info("Model artifact successfully validated and atomically written to %s", final_artifact_path)

    # Atomic metadata export for UI specification cards and auditability
    def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
        return {k: round(float(v), 6) for k, v in metrics.items()}

    final_meta_path = MODELS_DIR / "model_metadata.json"
    core_metadata: dict[str, Any] = {
        "model_type": "RandomForestRegressor",
        "feature_names": feature_names,
        "hyperparameters": {k: v for k, v in best_params.items() if k != "n_jobs"},
        "partitions": {
            "train": {"seasons": "2022-2024", "samples": int(len(train_df)), "metrics": _round_metrics(train_metrics)},
            "validation": {"seasons": "2025", "samples": int(len(val_df)), "metrics": _round_metrics(val_metrics)},
            "test": {"seasons": "2026", "samples": int(len(test_df)), "metrics": _round_metrics(test_metrics)},
            "total_samples": int(len(full_df)),
        },
    }

    trained_at = datetime.now(UTC).isoformat()
    if final_meta_path.exists():
        try:
            existing_meta = json.loads(final_meta_path.read_text(encoding="utf-8"))
            if isinstance(existing_meta, dict) and all(
                existing_meta.get(key) == val for key, val in core_metadata.items()
            ):
                trained_at = str(existing_meta.get("trained_at") or trained_at)
        except Exception as exc:
            logger.debug("Ignoring unreadable existing metadata at %s: %s", final_meta_path, exc)

    metadata_payload: dict[str, Any] = {
        "trained_at": trained_at,
        **core_metadata,
    }
    temp_meta_path = MODELS_DIR / "model_metadata.tmp.json"
    with open(temp_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    os.replace(temp_meta_path, final_meta_path)
    logger.info("Training metadata atomically written to %s", final_meta_path)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
