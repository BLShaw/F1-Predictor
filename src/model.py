"""Machine learning and stochastic simulation engine for Formula 1 race outcomes.

Combines supervised Random Forest regression with calibrated Monte Carlo simulations
and TreeSHAP interpretability.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR, OVERTAKING_DIFFICULTY_MAP, get_constructor_dnf_rate

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

# Context-dependent calibration multipliers for Monte Carlo simulation
WET_VARIANCE_MULTIPLIER: float = 1.35
WET_DNF_MULTIPLIER: float = 1.25


def _is_wet_condition(
    df: pd.DataFrame,
    weather: dict[str, Any] | None = None,
    rainfall: bool | None = None,
) -> bool:
    """Resolve whether race simulation operates under wet-weather conditions."""
    if rainfall is not None:
        return bool(rainfall)
    if isinstance(weather, dict) and "rainfall" in weather:
        return bool(weather.get("rainfall"))
    if "rainfall" in df.columns:
        return bool(df["rainfall"].fillna(False).astype(bool).any())
    return False


def _resolve_simulation_variance(
    df: pd.DataFrame,
    base_std: float,
    weather: dict[str, Any] | None = None,
    rainfall: bool | None = None,
) -> np.ndarray:
    """Compute context-calibrated positional noise std per driver.

    Scales base variance inversely with circuit overtaking_difficulty and increases
    variance under wet-weather conditions.
    """
    n_drivers = len(df)
    if "overtaking_difficulty" in df.columns:
        raw_col = df["overtaking_difficulty"]
        parsed_vals: list[float] = []
        for val in raw_col:
            if pd.isna(val):
                parsed_vals.append(0.50)
            elif isinstance(val, str):
                cleaned = val.strip().lower()
                if cleaned in OVERTAKING_DIFFICULTY_MAP:
                    parsed_vals.append(OVERTAKING_DIFFICULTY_MAP[cleaned])
                else:
                    try:
                        parsed_vals.append(float(cleaned))
                    except ValueError:
                        parsed_vals.append(0.50)
            else:
                try:
                    parsed_vals.append(float(val))
                except (TypeError, ValueError):
                    parsed_vals.append(0.50)
        overtaking_scores = np.clip(np.array(parsed_vals, dtype=np.float64), 0.0, 1.0)
    else:
        overtaking_scores = np.full(n_drivers, 0.50, dtype=np.float64)

    overtaking_factor = 1.20 - 0.40 * overtaking_scores
    wet_factor = WET_VARIANCE_MULTIPLIER if _is_wet_condition(df, weather=weather, rainfall=rainfall) else 1.0
    stds = base_std * overtaking_factor * wet_factor
    return np.clip(stds, 0.05, None)


def _resolve_driver_dnf_probs(
    df: pd.DataFrame,
    base_dnf_prob: float = 0.05,
    dnf_prob_override: float | None = None,
    constructor_dnf_rates: dict[str, float] | None = None,
    weather: dict[str, Any] | None = None,
    rainfall: bool | None = None,
) -> np.ndarray:
    """Resolve per-driver retirement probabilities with constructor and weather calibration."""
    is_wet = _is_wet_condition(df, weather=weather, rainfall=rainfall)
    target_base = dnf_prob_override if dnf_prob_override is not None else base_dnf_prob
    has_dnf_col = "dnf_prob" in df.columns

    probs: list[float] = []
    for _, row in df.iterrows():
        if has_dnf_col and dnf_prob_override is None and pd.notna(row.get("dnf_prob")):
            try:
                p = float(row.get("dnf_prob"))
            except (TypeError, ValueError):
                p = float(target_base)
        else:
            team_val = row.get("team")
            team_str = str(team_val).strip() if pd.notna(team_val) else None
            drv_val = row.get("driver")
            drv_str = str(drv_val).strip() if pd.notna(drv_val) else None
            matched_rate = get_constructor_dnf_rate(
                team_name=team_str,
                driver_code=drv_str,
                default_rate=-1.0,
                custom_rates=constructor_dnf_rates,
            )
            if matched_rate >= 0.0:
                if dnf_prob_override is not None:
                    p = matched_rate * (dnf_prob_override / 0.05)
                elif base_dnf_prob != 0.05 and constructor_dnf_rates is None:
                    p = matched_rate * (base_dnf_prob / 0.05)
                else:
                    p = matched_rate
            else:
                p = float(target_base)

        if is_wet and dnf_prob_override is None:
            p *= WET_DNF_MULTIPLIER

        probs.append(p)

    return np.clip(np.array(probs, dtype=np.float64), 0.0, 0.99)


def _run_vectorized_monte_carlo(
    base_scores: np.ndarray,
    noise_stds: np.ndarray,
    dnf_probs: np.ndarray,
    n_sims: int,
    points_map: dict[int, int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Execute vectorized 2D Monte Carlo race simulations across (n_sims, n_drivers)."""
    n_sims_safe = max(int(n_sims), 1)
    n_drivers = len(base_scores)

    noise = rng.normal(0.0, noise_stds[np.newaxis, :], size=(n_sims_safe, n_drivers))
    sim_scores = base_scores[np.newaxis, :] + noise
    dnf_mask = rng.random((n_sims_safe, n_drivers)) < dnf_probs[np.newaxis, :]

    masked_scores = np.where(dnf_mask, np.inf, sim_scores)
    order = np.argsort(masked_scores, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    sim_positions = np.where(dnf_mask, np.nan, ranks.astype(np.float64))

    points_lookup = np.array([0.0] + [float(points_map.get(r, 0)) for r in range(1, n_drivers + 1)], dtype=np.float64)
    sim_points = points_lookup[np.where(dnf_mask, 0, ranks)]

    classified_mask = ~dnf_mask
    win_rate = np.mean((ranks == 1) & classified_mask, axis=0)
    podium_rate = np.mean((ranks >= 1) & (ranks <= 3) & classified_mask, axis=0)
    points_rate = np.mean((ranks >= 1) & (ranks <= 10) & classified_mask, axis=0)
    exp_points = np.mean(sim_points, axis=0)
    finish_rate = np.mean(classified_mask, axis=0)

    finish_counts = np.sum(classified_mask, axis=0)
    avg_finish = np.where(
        finish_counts > 0,
        np.nansum(sim_positions, axis=0) / np.maximum(finish_counts, 1),
        float(n_drivers),
    )

    return {
        "win_rate": win_rate,
        "podium_rate": podium_rate,
        "points_rate": points_rate,
        "exp_points": exp_points,
        "finish_rate": finish_rate,
        "avg_finish": avg_finish,
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


def load_model_metadata(metadata_path: Path | None = None) -> dict[str, Any]:
    """Load persisted training metadata JSON from disk, returning empty dict if unavailable."""
    target_path = metadata_path if metadata_path is not None else MODELS_DIR / "model_metadata.json"
    if target_path.exists():
        try:
            with open(target_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to load model metadata from %s: %s", target_path, exc)
    return {}


class AdvancedRacePredictor:
    """Heuristic Monte Carlo simulator for race outcomes based on grid position and pace."""

    def __init__(
        self,
        grid_weight: float = 0.55,
        pace_weight: float = 0.45,
        base_dnf_prob: float = 0.05,
        constructor_dnf_rates: dict[str, float] | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self.grid_weight = grid_weight
        self.pace_weight = pace_weight
        self.base_dnf_prob = base_dnf_prob
        self.constructor_dnf_rates = constructor_dnf_rates
        self.random_seed = random_seed
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

    def predict(
        self,
        features: pd.DataFrame,
        n_sims: int = 1000,
        dnf_prob: float | None = None,
        weather: dict[str, Any] | None = None,
        rainfall: bool | None = None,
        constructor_dnf_rates: dict[str, float] | None = None,
        seed: int | None = 42,
    ) -> pd.DataFrame:
        """Run vectorized Monte Carlo simulations with context-calibrated variance and DNF rates."""
        if features.empty or "driver" not in features.columns:
            return pd.DataFrame()

        sim_df = self.calculate_base_score(features)
        drivers = sim_df["driver"].values
        base_scores = sim_df["base_score"].values.astype(np.float64)

        noise_stds = _resolve_simulation_variance(
            sim_df,
            base_std=self.pace_variability,
            weather=weather,
            rainfall=rainfall,
        )
        effective_rates = constructor_dnf_rates if constructor_dnf_rates is not None else self.constructor_dnf_rates
        dnf_probs = _resolve_driver_dnf_probs(
            sim_df,
            base_dnf_prob=self.base_dnf_prob,
            dnf_prob_override=dnf_prob,
            constructor_dnf_rates=effective_rates,
            weather=weather,
            rainfall=rainfall,
        )
        rng = np.random.default_rng(seed)
        mc_stats = _run_vectorized_monte_carlo(
            base_scores=base_scores,
            noise_stds=noise_stds,
            dnf_probs=dnf_probs,
            n_sims=n_sims,
            points_map=self.points_map,
            rng=rng,
        )

        stats: list[dict[str, Any]] = []
        for idx, driver in enumerate(drivers):
            stats.append(
                {
                    "Driver": driver,
                    "Grid": int(sim_df.iloc[idx]["grid"]),
                    "Win %": round(float(mc_stats["win_rate"][idx]), 4),
                    "Podium %": round(float(mc_stats["podium_rate"][idx]), 4),
                    "Points %": round(float(mc_stats["points_rate"][idx]), 4),
                    "Exp. Points": round(float(mc_stats["exp_points"][idx]), 2),
                    "Finish Rate %": round(float(mc_stats["finish_rate"][idx]) * 100, 1),
                    "Avg Finish": round(float(mc_stats["avg_finish"][idx]), 1),
                }
            )

        results_df = pd.DataFrame(stats)
        return results_df.sort_values(by=["Win %", "Exp. Points"], ascending=[False, False]).reset_index(drop=True)


class F1MLPredictor:
    """Production ML-based F1 Race Predictor using Random Forest and SHAP explainability."""

    def __init__(
        self,
        model: RandomForestRegressor | None = None,
        *,
        load_historical: bool = True,
        constructor_dnf_rates: dict[str, float] | None = None,
        random_seed: int | None = 42,
    ) -> None:
        self.model = model
        self.is_historical = False
        self.constructor_dnf_rates = constructor_dnf_rates
        self.random_seed = random_seed
        self.feature_names: list[str] = [
            "grid_norm",
            "quali_gap_norm",
            "pace_norm",
            "pace_consistency",
            "driver_form",
            "constructor_form",
            "overtaking_difficulty",
            "is_street_circuit",
            "has_practice_data",
            "is_sprint",
        ]
        self.feature_descriptions: dict[str, str] = {
            "grid_norm": "Qualifying starting position (0=Pole, 1=Back of grid)",
            "quali_gap_norm": "Qualifying best lap gap to pole normalized to qualifying spread",
            "pace_norm": "Practice pace gap normalized to session benchmark",
            "pace_consistency": "Practice stint lap consistency (lower std = more consistent)",
            "driver_form": "Pre-race EWMA of driver's recent classified finishes (0=P1, 1=Back)",
            "constructor_form": "Pre-race EWMA of constructor's recent classified finishes (0=P1, 1=Back)",
            "overtaking_difficulty": "Circuit overtaking difficulty score (0.25=Low to 1.00=Very High)",
            "is_street_circuit": "Street circuit indicator (1=Street, 0=Permanent)",
            "has_practice_data": "Indicator for recorded practice telemetry",
            "is_sprint": "Sprint weekend indicator",
        }

        self.points_map = POINTS_MAP
        self.base_dnf_prob = 0.05
        self.shap_values: np.ndarray | None = None
        self.shap_explainer: Any = None

        if self.model is None and load_historical:
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

        # 2. Normalized qualifying gap to pole
        if "quali_gap_norm" in features_df.columns:
            features_df["quali_gap_norm"] = (
                pd.to_numeric(features_df["quali_gap_norm"], errors="coerce")
                .fillna(features_df["grid_norm"])
                .clip(0.0, 1.5)
            )
        else:
            features_df["quali_gap_norm"] = features_df["grid_norm"]

        # 3. Normalized practice pace [0.0, 1.5] relative to observed session spread
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

        # 4. Practice stint consistency
        if "consistency" in features_df.columns and features_df["consistency"].notna().any():
            features_df["consistency"] = pd.to_numeric(features_df["consistency"], errors="coerce")
            med_consistency = float(features_df["consistency"].dropna().median())
            features_df["pace_consistency"] = features_df["consistency"].fillna(med_consistency).clip(0.0, 3.0) / 3.0
        else:
            features_df["pace_consistency"] = 0.35

        # 5. Pre-race driver and constructor rolling form
        if "driver_form" in features_df.columns:
            features_df["driver_form"] = (
                pd.to_numeric(features_df["driver_form"], errors="coerce")
                .fillna(features_df["grid_norm"])
                .clip(0.0, 1.0)
            )
        else:
            features_df["driver_form"] = features_df["grid_norm"]

        if "constructor_form" in features_df.columns:
            features_df["constructor_form"] = (
                pd.to_numeric(features_df["constructor_form"], errors="coerce")
                .fillna(features_df["grid_norm"])
                .clip(0.0, 1.0)
            )
        else:
            features_df["constructor_form"] = features_df["grid_norm"]

        # 6. Circuit overtaking difficulty and street circuit indicator
        if "overtaking_difficulty" in features_df.columns:
            features_df["overtaking_difficulty"] = (
                pd.to_numeric(features_df["overtaking_difficulty"], errors="coerce").fillna(0.50).clip(0.0, 1.0)
            )
        else:
            features_df["overtaking_difficulty"] = 0.50

        if "is_street_circuit" in features_df.columns:
            features_df["is_street_circuit"] = (
                pd.to_numeric(features_df["is_street_circuit"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            )
        else:
            features_df["is_street_circuit"] = 0.0

        # 7. Binary indicator for presence of practice pace telemetry
        if "has_practice_data" not in features_df.columns:
            features_df["has_practice_data"] = 1.0
        else:
            features_df["has_practice_data"] = pd.to_numeric(features_df["has_practice_data"], errors="coerce").fillna(
                1.0
            )

        # 8. Sprint weekend indicator
        if "is_sprint" not in features_df.columns:
            features_df["is_sprint"] = 0.0
        else:
            features_df["is_sprint"] = pd.to_numeric(features_df["is_sprint"], errors="coerce").fillna(0.0)

        feature_matrix = features_df[self.feature_names].values.astype(np.float64)
        return features_df, feature_matrix

    def train_synthetic_fallback(self, n_samples: int = 1500) -> None:
        """Fit a lightweight fallback Random Forest if no historical binary is available."""
        rng = np.random.default_rng(42)
        X_synth: list[list[float]] = []
        y_synth: list[float] = []

        for _ in range(n_samples):
            grid_norm = float(rng.uniform(0.0, 1.0))
            quali_gap_norm = float(np.clip(grid_norm + rng.normal(0.0, 0.1), 0.0, 1.5))
            pace_norm = float(np.clip(grid_norm + rng.normal(0.0, 0.2), 0.0, 1.0))
            consistency = float(rng.uniform(0.1, 0.6))
            driver_form = float(np.clip(grid_norm + rng.normal(0.0, 0.15), 0.0, 1.0))
            constructor_form = float(np.clip(grid_norm + rng.normal(0.0, 0.15), 0.0, 1.0))
            overtaking_difficulty = float(rng.choice([0.25, 0.50, 0.75, 1.00]))
            is_street_circuit = 1.0 if rng.random() > 0.65 else 0.0
            has_practice = 1.0 if rng.random() > 0.05 else 0.0
            is_sprint = 1.0 if rng.random() > 0.75 else 0.0

            # Finishing position among classified finishers (supporting grids up to 22 drivers)
            composite = grid_norm * 0.55 + pace_norm * 0.15 + driver_form * 0.20 + constructor_form * 0.10
            target_finish = 1.0 + composite * 21.0 + float(rng.normal(0.0, 1.2))
            target_finish = float(np.clip(target_finish, 1.0, 22.0))

            X_synth.append(
                [
                    grid_norm,
                    quali_gap_norm,
                    pace_norm,
                    consistency,
                    driver_form,
                    constructor_form,
                    overtaking_difficulty,
                    is_street_circuit,
                    has_practice,
                    is_sprint,
                ]
            )
            y_synth.append(target_finish)

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
        )
        self.model.fit(np.array(X_synth), np.array(y_synth))
        self.is_historical = False

    def predict(
        self,
        features: pd.DataFrame,
        n_sims: int = 2000,
        dnf_prob: float | None = None,
        weather: dict[str, Any] | None = None,
        rainfall: bool | None = None,
        constructor_dnf_rates: dict[str, float] | None = None,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        """Predict race finishing distribution with vectorized Monte Carlo simulations and SHAP values."""
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

        # Context-calibrated vectorized Monte Carlo Simulation with non-finisher exclusion
        noise_stds = _resolve_simulation_variance(
            features_df,
            base_std=1.4,
            weather=weather,
            rainfall=rainfall,
        )
        effective_rates = constructor_dnf_rates if constructor_dnf_rates is not None else self.constructor_dnf_rates
        dnf_probs = _resolve_driver_dnf_probs(
            features_df,
            base_dnf_prob=self.base_dnf_prob,
            dnf_prob_override=dnf_prob,
            constructor_dnf_rates=effective_rates,
            weather=weather,
            rainfall=rainfall,
        )
        rng = np.random.default_rng(seed)
        mc_stats = _run_vectorized_monte_carlo(
            base_scores=predicted_ranks.astype(np.float64),
            noise_stds=noise_stds,
            dnf_probs=dnf_probs,
            n_sims=n_sims,
            points_map=self.points_map,
            rng=rng,
        )

        stats: list[dict[str, Any]] = []
        drivers = features_df["driver"].values

        for idx, driver in enumerate(drivers):
            stats.append(
                {
                    "Driver": driver,
                    "Grid": int(features_df.iloc[idx]["grid"]),
                    "Predicted Pos": round(float(predicted_ranks[idx]), 1),
                    "Win %": round(float(mc_stats["win_rate"][idx]), 4),
                    "Podium %": round(float(mc_stats["podium_rate"][idx]), 4),
                    "Points %": round(float(mc_stats["points_rate"][idx]), 4),
                    "Exp. Points": round(float(mc_stats["exp_points"][idx]), 2),
                    "Finish Rate %": round(float(mc_stats["finish_rate"][idx]) * 100, 1),
                    "Avg Finish": round(float(mc_stats["avg_finish"][idx]), 1),
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
