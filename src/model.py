"""Machine learning and stochastic simulation engine for Formula 1 race outcomes.

Combines supervised Random Forest regression with calibrated Monte Carlo simulations
and TreeSHAP interpretability.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR

# Optional SHAP explainability
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

# Standard F1 FIA Points allocation for positions 1 through 10
POINTS_MAP: dict[int, int] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


def load_saved_ml_model(model_path: Path | None = None) -> RandomForestRegressor | None:
    """Load pre-trained Random Forest model binary from disk."""
    target_path = model_path if model_path is not None else MODELS_DIR / "f1_historical_model.joblib"
    if target_path.exists():
        try:
            model = joblib.load(target_path)
            logger.info("Successfully loaded ML model from %s", target_path)
            return model
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", target_path, exc)
    return None


class AdvancedRacePredictor:
    """Heuristic Monte Carlo simulator for race outcomes based on grid position and pace."""

    def __init__(self, grid_weight: float = 0.55, pace_weight: float = 0.45, base_dnf_prob: float = 0.05) -> None:
        self.grid_weight = grid_weight
        self.pace_weight = pace_weight
        self.base_dnf_prob = base_dnf_prob
        self.pace_variability = 0.5
        self.points_map = POINTS_MAP

    def calculate_base_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite baseline performance index (lower is better)."""
        sim_df = df.copy()
        n_drivers = len(sim_df)

        sim_df["grid"] = pd.to_numeric(sim_df["grid"], errors="coerce").fillna(n_drivers)

        if "best" in sim_df.columns:
            sim_df["best"] = pd.to_numeric(sim_df["best"], errors="coerce")
            max_pace = sim_df["best"].dropna().max() if sim_df["best"].notna().any() else 90.0
            sim_df["best"] = sim_df["best"].fillna(max_pace * 1.02)
            sim_df["pace_rank"] = sim_df["best"].rank(ascending=True)
        else:
            sim_df["pace_rank"] = sim_df["grid"]

        sim_df["base_score"] = (sim_df["grid"] * self.grid_weight) + (sim_df["pace_rank"] * self.pace_weight)
        return sim_df

    def predict(self, features: pd.DataFrame, n_sims: int = 1000) -> pd.DataFrame:
        """Run Monte Carlo simulations with proper DNF points withholding."""
        if features.empty or "driver" not in features.columns:
            return pd.DataFrame()

        sim_df = self.calculate_base_score(features)
        drivers = sim_df["driver"].values
        base_scores = sim_df["base_score"].values
        n_drivers = len(drivers)

        # sim_positions stores finishing rank (1..N_finishers) or np.nan for DNFs
        sim_positions = np.full((n_sims, n_drivers), np.nan)

        for i in range(n_sims):
            noise = np.random.normal(0, self.pace_variability, size=n_drivers)
            iter_scores = base_scores + noise
            dnf_mask = np.random.random(n_drivers) < self.base_dnf_prob

            finisher_indices = np.where(~dnf_mask)[0]
            if len(finisher_indices) > 0:
                finisher_scores = iter_scores[finisher_indices]
                sorted_finisher_order = np.argsort(finisher_scores)
                for rank_pos, f_idx in enumerate(sorted_finisher_order):
                    sim_positions[i, finisher_indices[f_idx]] = rank_pos + 1

        stats: list[dict[str, Any]] = []
        for idx, driver in enumerate(drivers):
            driver_ranks = sim_positions[:, idx]
            classified_mask = ~np.isnan(driver_ranks)
            classified_ranks = driver_ranks[classified_mask]

            win_count = np.sum(driver_ranks == 1)
            podium_count = np.sum((driver_ranks >= 1) & (driver_ranks <= 3))
            points_count = np.sum((driver_ranks >= 1) & (driver_ranks <= 10))

            total_points = sum(self.points_map.get(int(r), 0) for r in classified_ranks)
            exp_points = total_points / n_sims
            finish_rate = float(np.mean(classified_mask))
            avg_finish = float(np.mean(classified_ranks)) if len(classified_ranks) > 0 else float(n_drivers)

            stats.append(
                {
                    "Driver": driver,
                    "Grid": int(sim_df.iloc[idx]["grid"]),
                    "Win %": round(win_count / n_sims, 4),
                    "Podium %": round(podium_count / n_sims, 4),
                    "Points %": round(points_count / n_sims, 4),
                    "Exp. Points": round(exp_points, 2),
                    "Finish Rate %": round(finish_rate * 100, 1),
                    "Avg Finish": round(avg_finish, 1),
                }
            )

        results_df = pd.DataFrame(stats)
        return results_df.sort_values(by=["Win %", "Exp. Points"], ascending=[False, False]).reset_index(drop=True)


class F1MLPredictor:
    """Production ML-based F1 Race Predictor using Random Forest and SHAP explainability."""

    def __init__(self, model: RandomForestRegressor | None = None) -> None:
        self.model = model
        self.is_historical = False
        self.feature_names: list[str] = [
            "grid_norm",
            "pace_norm",
            "pace_consistency",
            "has_practice_data",
            "is_sprint",
        ]
        self.feature_descriptions: dict[str, str] = {
            "grid_norm": "Qualifying starting position (0=Pole, 1=Back of grid)",
            "pace_norm": "Practice pace gap normalized to session benchmark",
            "pace_consistency": "Practice stint lap consistency (lower std = more consistent)",
            "has_practice_data": "Indicator for recorded practice telemetry",
            "is_sprint": "Sprint weekend indicator",
        }

        self.points_map = POINTS_MAP
        self.base_dnf_prob = 0.05
        self.shap_values: np.ndarray | None = None
        self.shap_explainer: Any = None

        if self.model is None:
            self._load_historical_model()

    def _load_historical_model(self) -> None:
        """Load pre-trained model artifact if present on disk."""
        loaded = load_saved_ml_model()
        if loaded is not None:
            self.model = loaded
            self.is_historical = True
        else:
            self.is_historical = False

    def _engineer_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Transform raw weekend pace and grid data into normalized ML features."""
        features_df = df.copy()
        n_drivers = max(len(features_df), 1)

        # 1. Normalized grid position [0.0, 1.0]
        features_df["grid"] = pd.to_numeric(features_df["grid"], errors="coerce").fillna(n_drivers)
        features_df["grid_norm"] = (features_df["grid"] - 1.0) / max(n_drivers - 1, 1)

        # 2. Normalized practice pace [0.0, 1.0] relative to observed session spread
        if "best" in features_df.columns and features_df["best"].notna().any():
            features_df["best"] = pd.to_numeric(features_df["best"], errors="coerce")
            pace_min = float(features_df["best"].dropna().min())
            pace_max = float(features_df["best"].dropna().max())
            pace_range = pace_max - pace_min if pace_max > pace_min else 1.0
            imputed_pace = pace_max * 1.02
            features_df["best"] = features_df["best"].fillna(imputed_pace)
            features_df["pace_norm"] = ((features_df["best"] - pace_min) / pace_range).clip(0.0, 1.5)
        else:
            features_df["pace_norm"] = features_df["grid_norm"]

        # 3. Practice stint consistency
        if "consistency" in features_df.columns and features_df["consistency"].notna().any():
            features_df["consistency"] = pd.to_numeric(features_df["consistency"], errors="coerce")
            med_consistency = float(features_df["consistency"].dropna().median())
            features_df["pace_consistency"] = features_df["consistency"].fillna(med_consistency).clip(0.0, 3.0) / 3.0
        else:
            features_df["pace_consistency"] = 0.35

        # 4. Binary indicator for presence of practice pace telemetry
        if "has_practice_data" not in features_df.columns:
            features_df["has_practice_data"] = 1.0
        else:
            features_df["has_practice_data"] = pd.to_numeric(features_df["has_practice_data"], errors="coerce").fillna(
                1.0
            )

        # 5. Sprint weekend indicator
        if "is_sprint" not in features_df.columns:
            features_df["is_sprint"] = 0.0
        else:
            features_df["is_sprint"] = pd.to_numeric(features_df["is_sprint"], errors="coerce").fillna(0.0)

        feature_matrix = features_df[self.feature_names].values.astype(np.float64)
        return features_df, feature_matrix

    def train_synthetic_fallback(self, n_samples: int = 1500) -> None:
        """Fit a lightweight fallback Random Forest if no historical binary is available."""
        np.random.seed(42)
        X_synth: list[list[float]] = []
        y_synth: list[float] = []

        for _ in range(n_samples):
            grid_norm = float(np.random.uniform(0.0, 1.0))
            pace_noise = float(np.random.normal(0.0, 0.2))
            pace_norm = float(np.clip(grid_norm + pace_noise, 0.0, 1.0))
            consistency = float(np.random.uniform(0.1, 0.6))
            has_practice = 1.0 if np.random.random() > 0.05 else 0.0
            is_sprint = 1.0 if np.random.random() > 0.75 else 0.0

            # Finishing position among classified finishers (supporting grids up to 22 drivers)
            target_finish = 1.0 + (grid_norm * 0.75 + pace_norm * 0.25) * 21.0 + np.random.normal(0.0, 1.2)
            target_finish = float(np.clip(target_finish, 1.0, 22.0))


            X_synth.append([grid_norm, pace_norm, consistency, has_practice, is_sprint])
            y_synth.append(target_finish)

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
        )
        self.model.fit(np.array(X_synth), np.array(y_synth))
        self.is_historical = False

    def predict(self, features: pd.DataFrame, n_sims: int = 2000, dnf_prob: float | None = None) -> dict[str, Any]:
        """Predict race finishing distribution with Monte Carlo simulations and SHAP values."""
        if features.empty or "driver" not in features.columns:
            return {
                "predictions": pd.DataFrame(),
                "feature_importance": None,
                "shap_data": None,
                "shap_error": "No driver features provided",
                "features_used": pd.DataFrame(),
                "model_type": "None",
            }

        n_drivers = len(features)
        features_df, X = self._engineer_features(features)

        if self.model is None:
            self.train_synthetic_fallback()

        # Generate base point predictions for classified finishing rank
        if self.model is not None:
            raw_ml_preds = self.model.predict(X)
        else:
            raw_ml_preds = 1.0 + features_df["grid_norm"].values * (n_drivers - 1)

        predicted_ranks = np.clip(raw_ml_preds, 1.0, float(n_drivers))

        # Compute SHAP feature importance if library is available
        feature_importance: pd.DataFrame | None = None
        shap_data: dict[str, Any] | None = None
        shap_error: str | None = None

        if HAS_SHAP and self.model is not None:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.shap_explainer.shap_values(X, check_additivity=False)
                mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

                feature_importance = (
                    pd.DataFrame(
                        {
                            "Feature": self.feature_names,
                            "Importance": mean_abs_shap,
                            "Description": [self.feature_descriptions[f] for f in self.feature_names],
                        }
                    )
                    .sort_values("Importance", ascending=False)
                    .reset_index(drop=True)
                )

                base_val = self.shap_explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = float(base_val[0])

                shap_data = {
                    "values": self.shap_values,
                    "base_value": base_val,
                    "feature_names": self.feature_names,
                    "X": X,
                }
            except Exception as exc:
                shap_error = str(exc)
                logger.warning("SHAP calculation failed: %s", exc)
        elif not HAS_SHAP:
            shap_error = "SHAP package is not installed."

        # Monte Carlo Simulation with proper non-finisher exclusion
        effective_dnf_prob = dnf_prob if dnf_prob is not None else self.base_dnf_prob
        sim_positions = np.full((n_sims, n_drivers), np.nan)

        for i in range(n_sims):
            # Performance disturbance per race simulation
            noise = np.random.normal(0, 1.4, size=n_drivers)
            sim_scores = predicted_ranks + noise
            dnf_mask = np.random.random(n_drivers) < effective_dnf_prob

            finisher_indices = np.where(~dnf_mask)[0]
            if len(finisher_indices) > 0:
                finisher_scores = sim_scores[finisher_indices]
                sorted_order = np.argsort(finisher_scores)
                for rank_pos, f_idx in enumerate(sorted_order):
                    sim_positions[i, finisher_indices[f_idx]] = rank_pos + 1

        stats: list[dict[str, Any]] = []
        drivers = features_df["driver"].values

        for idx, driver in enumerate(drivers):
            driver_ranks = sim_positions[:, idx]
            classified_mask = ~np.isnan(driver_ranks)
            classified_ranks = driver_ranks[classified_mask]

            win_count = np.sum(driver_ranks == 1)
            podium_count = np.sum((driver_ranks >= 1) & (driver_ranks <= 3))
            points_count = np.sum((driver_ranks >= 1) & (driver_ranks <= 10))

            total_points = sum(self.points_map.get(int(r), 0) for r in classified_ranks)
            exp_points = total_points / n_sims
            finish_rate = float(np.mean(classified_mask))
            avg_finish = float(np.mean(classified_ranks)) if len(classified_ranks) > 0 else float(n_drivers)

            stats.append(
                {
                    "Driver": driver,
                    "Grid": int(features_df.iloc[idx]["grid"]),
                    "Predicted Pos": round(float(predicted_ranks[idx]), 1),
                    "Win %": round(win_count / n_sims, 4),
                    "Podium %": round(podium_count / n_sims, 4),
                    "Points %": round(points_count / n_sims, 4),
                    "Exp. Points": round(exp_points, 2),
                    "Finish Rate %": round(finish_rate * 100, 1),
                    "Avg Finish": round(avg_finish, 1),
                }
            )

        results_df = pd.DataFrame(stats)
        results_df = results_df.sort_values(
            by=["Win %", "Exp. Points"],
            ascending=[False, False],
        ).reset_index(drop=True)

        return {
            "predictions": results_df,
            "feature_importance": feature_importance,
            "shap_data": shap_data,
            "shap_error": shap_error,
            "features_used": features_df[self.feature_names + ["driver"]],
            "model_type": "Random Forest",
        }

    def get_driver_explanation(self, driver_idx: int, features_df: pd.DataFrame) -> dict[str, Any] | None:
        """Get per-driver SHAP contribution breakdown."""
        if self.shap_values is None or self.shap_explainer is None:
            return None

        if driver_idx < 0 or driver_idx >= len(self.shap_values):
            return None

        driver_shap = self.shap_values[driver_idx]
        driver_features = features_df.iloc[driver_idx]

        explanations: list[dict[str, Any]] = []
        for i, feat in enumerate(self.feature_names):
            val = float(driver_features[feat])
            contrib = float(driver_shap[i])
            explanations.append(
                {
                    "feature": feat,
                    "description": self.feature_descriptions[feat],
                    "value": round(val, 3),
                    "shap_contribution": round(contrib, 4),
                    "impact": "Improves Position" if contrib < 0 else "Worsens Position",
                }
            )

        base_val = self.shap_explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])

        return {
            "driver": driver_features.get("driver", f"Driver {driver_idx}"),
            "base_prediction": round(float(base_val), 2),
            "feature_contributions": sorted(explanations, key=lambda x: abs(x["shap_contribution"]), reverse=True),
        }


def get_shap_summary_data(
    shap_values: np.ndarray | None,
    feature_names: Sequence[str],
) -> pd.DataFrame | None:
    """Summarize global SHAP metrics across all drivers."""
    if shap_values is None or len(shap_values) == 0:
        return None

    summary: list[dict[str, Any]] = []
    for i, feature in enumerate(feature_names):
        summary.append(
            {
                "Feature": feature,
                "Mean |SHAP|": float(np.abs(shap_values[:, i]).mean()),
                "Max |SHAP|": float(np.abs(shap_values[:, i]).max()),
                "Std SHAP": float(shap_values[:, i].std()),
            }
        )

    return pd.DataFrame(summary).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
