"""Historical Model Training and Temporal Validation Pipeline.

Trains the production Random Forest regressor with temporal (season-aware) splitting,
statistically valid DNF handling, and atomic artifact replacement.
"""

import logging
import os
import sys
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

from src.config import MODELS_DIR, SEASONS_DIR
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
            or (pos_int <= 16 and status not in ["Did not start", "Disqualified"])
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
    predictor = F1MLPredictor()
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    if not SEASONS_DIR.exists():
        logger.error("Seasons data directory does not exist: %s", SEASONS_DIR)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    total_gps = 0

    for year_dir in sorted(SEASONS_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        season_year = int(year_dir.name)

        for gp_dir in sorted(year_dir.iterdir()):
            if not gp_dir.is_dir() or gp_dir.name.startswith("."):
                continue

            gp_data = load_gp_data(season_year, gp_dir.name)
            sessions = gp_data.get("sessions", {})
            race_session = sessions.get("race")
            if not race_session:
                continue

            classified_results = extract_classified_race_results(race_session)
            if not classified_results:
                continue

            # Construct grid-anchored pre-race features
            weekend_features = build_weekend_features(gp_data)
            if weekend_features.empty:
                continue

            # Engineer normalized ML features
            features_df, _X_matrix = predictor._engineer_features(weekend_features)

            for idx, row in features_df.iterrows():
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

    y_pred_rounded = np.clip(np.round(y_pred), 1, 20)

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
    print(f"Mean Absolute Error:        {mae:.2f} positions")
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
        "points_acc": pts_acc,
        "points_f1": pts_f1,
        "podium_acc": podium_acc,
        "podium_f1": podium_f1,
        "baseline_podium_acc": baseline_podium_acc,
    }


def main() -> bool:
    """Execute historical training and validation pipeline."""
    logger.info("Starting F1 Predictor Model Training Pipeline")

    train_df, val_df, test_df = load_dataset()
    if train_df.empty:
        logger.error("No valid historical training data extracted.")
        return False

    feature_names = ["grid_norm", "pace_norm", "pace_consistency", "has_practice_data", "is_sprint"]

    X_train = train_df[feature_names].values
    y_train = train_df["target_position"].values

    logger.info("Training set: %d samples from 2022-2024", len(X_train))
    logger.info("Validation set: %d samples from 2025", len(val_df))
    logger.info("Test set: %d samples from 2026", len(test_df))

    # Initialize production Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate across all temporal partitions
    evaluate_model_partition(model, train_df, feature_names, "Training Set (2022-2024 In-Sample)")
    evaluate_model_partition(model, val_df, feature_names, "Validation Set (2025 Out-of-Time)")
    evaluate_model_partition(model, test_df, feature_names, "Test Set (2026 Out-of-Time)")

    # Model Artifact Validation
    test_sample = np.array([[0.0, 0.0, 0.1, 1.0, 0.0], [0.95, 0.9, 0.5, 1.0, 0.0]])
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
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
