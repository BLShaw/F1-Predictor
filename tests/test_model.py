"""Tests for machine learning model and Monte Carlo simulation engine."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import (
    WET_VARIANCE_MULTIPLIER,
    AdvancedRacePredictor,
    F1MLPredictor,
    _resolve_simulation_variance,
    get_shap_summary_data,
    load_model_metadata,
)
from src.ui.tabs.predictions import evaluate_race_backtest


def _make_dummy_features(n_drivers: int = 20) -> pd.DataFrame:
    """Generate mock driver feature DataFrame."""
    drivers = [f"DRV_{i}" for i in range(1, n_drivers + 1)]
    return pd.DataFrame(
        {
            "driver": drivers,
            "grid": list(range(1, n_drivers + 1)),
            "grid_norm": [(i - 1) / max(n_drivers - 1, 1) for i in range(1, n_drivers + 1)],
            "pace_norm": [(i - 1) / max(n_drivers - 1, 1) for i in range(1, n_drivers + 1)],
            "pace_consistency": [0.2 + 0.05 * i for i in range(n_drivers)],
        }
    )


def test_f1_ml_predictor_output_structure() -> None:
    """Verify F1MLPredictor returns complete output structure and bounded probabilities."""
    features = _make_dummy_features(10)
    predictor = F1MLPredictor()

    results = predictor.predict(features, n_sims=500)

    assert "predictions" in results
    assert "features_used" in results
    assert "model_type" in results
    assert results["model_type"] == "Random Forest"

    preds: pd.DataFrame = results["predictions"]
    assert len(preds) == 10

    required_cols = {"Driver", "Grid", "Avg Finish", "Win %", "Podium %", "Points %", "Exp. Points"}
    assert required_cols.issubset(set(preds.columns))

    # All probabilities must be mathematically valid percentages in [0.0, 1.0]
    for col in ["Win %", "Podium %", "Points %"]:
        assert (preds[col] >= 0.0).all()
        assert (preds[col] <= 1.0).all()

    # Sum of win percentages should be approximately 1.0 (or <= 1.0 allowing for ties/DNFs)
    assert preds["Win %"].sum() <= 1.01


def test_monte_carlo_dnf_points_withholding() -> None:
    """Verify non-finishers are never allocated points even with high DNF counts."""
    # 20 drivers
    features = _make_dummy_features(20)
    predictor = AdvancedRacePredictor()
    # Force high DNF rate (e.g. 70% chance of retirement per driver)
    predictor.base_dnf_prob = 0.7

    preds = predictor.predict(features, n_sims=500)

    assert len(preds) == 20
    # Even under high DNF probability, expected points should not exceed max possible points
    assert (preds["Exp. Points"] >= 0.0).all()
    # P1 max points without fastest lap is 25
    assert (preds["Exp. Points"] <= 26.0).all()
    # Backmarkers should have low expected points
    last_place = preds.iloc[-1]
    assert last_place["Exp. Points"] < 25.0


def test_advanced_race_predictor_predict() -> None:
    """Verify AdvancedRacePredictor fallback engine runs directly from feature frame."""
    features = _make_dummy_features(8)
    predictor = AdvancedRacePredictor()
    preds = predictor.predict(features, n_sims=300)

    assert not preds.empty
    assert len(preds) == 8
    assert "Win %" in preds.columns
    assert "Exp. Points" in preds.columns


def test_2026_cadillac_22_driver_grid_simulation() -> None:
    """Verify ML and Monte Carlo pipelines properly simulate 22-driver grids (2026 regulation)."""
    # 22 drivers representing 11 constructors including Cadillac
    features_22 = _make_dummy_features(22)

    ml_predictor = F1MLPredictor()
    ml_results = ml_predictor.predict(features_22, n_sims=500)
    ml_preds = ml_results["predictions"]

    assert len(ml_preds) == 22
    assert (ml_preds["Win %"] >= 0.0).all()
    assert (ml_preds["Win %"] <= 1.0).all()
    assert ml_preds["Win %"].sum() <= 1.01

    mc_predictor = AdvancedRacePredictor()
    mc_preds = mc_predictor.predict(features_22, n_sims=500)

    assert len(mc_preds) == 22
    assert (mc_preds["Exp. Points"] >= 0.0).all()


def test_f1_ml_predictor_dnf_prob_sensitivity() -> None:
    """Verify that specifying dnf_prob scales finish rates appropriately in F1MLPredictor."""
    features = _make_dummy_features(10)
    predictor = F1MLPredictor()

    results_low_dnf = predictor.predict(features, n_sims=500, dnf_prob=0.01)
    results_high_dnf = predictor.predict(features, n_sims=500, dnf_prob=0.50)

    avg_finish_rate_low = results_low_dnf["predictions"]["Finish Rate %"].mean()
    avg_finish_rate_high = results_high_dnf["predictions"]["Finish Rate %"].mean()

    assert avg_finish_rate_low > avg_finish_rate_high
    assert avg_finish_rate_low >= 95.0
    assert avg_finish_rate_high <= 65.0


def test_get_shap_summary_data_and_driver_explanation() -> None:
    """Verify global SHAP summary computation and driver-level marginal explanation extraction."""
    features = _make_dummy_features(6)
    predictor = F1MLPredictor()
    results = predictor.predict(features, n_sims=100)

    if results.get("shap_data") is not None:
        shap_data = results["shap_data"]
        summary_df = get_shap_summary_data(shap_data["values"], predictor.feature_names)
        assert summary_df is not None
        assert len(summary_df) == len(predictor.feature_names)
        assert "Mean |SHAP|" in summary_df.columns

        driver_expl = predictor.get_driver_explanation(0, results["features_used"])
        assert driver_expl is not None
        assert "feature_contributions" in driver_expl
        assert len(driver_expl["feature_contributions"]) == len(predictor.feature_names)


def test_f1_ml_predictor_skip_historical_load() -> None:
    """Verify F1MLPredictor(load_historical=False) skips disk unpickling on init."""
    predictor = F1MLPredictor(load_historical=False)
    assert predictor.model is None
    assert predictor.is_historical is False

    features = _make_dummy_features(6)
    results = predictor.predict(features, n_sims=100)
    assert len(results["predictions"]) == 6


def test_f1_ml_predictor_10_feature_pipeline() -> None:
    """Verify F1MLPredictor exposes and predicts across the 10-feature matrix."""
    predictor = F1MLPredictor(load_historical=False)
    assert len(predictor.feature_names) == 10
    expected_features = {
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
    }
    assert set(predictor.feature_names) == expected_features

    features = _make_dummy_features(8)
    features["quali_gap_norm"] = features["grid_norm"] * 0.9
    features["driver_form"] = features["grid_norm"] * 0.8
    features["constructor_form"] = features["grid_norm"] * 0.7
    features["overtaking_difficulty"] = 0.75
    features["is_street_circuit"] = 1

    results = predictor.predict(features, n_sims=100)
    assert len(results["predictions"]) == 8
    assert set(predictor.feature_names).issubset(set(results["features_used"].columns))


def test_monte_carlo_seed_reproducibility() -> None:
    """Verify identical seeds reproduce exact simulation outputs and different seeds diverge."""
    features = _make_dummy_features(12)

    ml_predictor = F1MLPredictor(load_historical=False)
    ml_run_1 = ml_predictor.predict(features, n_sims=1000, seed=42)["predictions"]
    ml_run_2 = ml_predictor.predict(features, n_sims=1000, seed=42)["predictions"]
    ml_run_other = ml_predictor.predict(features, n_sims=1000, seed=99)["predictions"]

    pd.testing.assert_frame_equal(ml_run_1, ml_run_2)
    assert not ml_run_1.equals(ml_run_other)

    mc_predictor = AdvancedRacePredictor()
    mc_run_1 = mc_predictor.predict(features, n_sims=1000, seed=42)
    mc_run_2 = mc_predictor.predict(features, n_sims=1000, seed=42)
    mc_run_other = mc_predictor.predict(features, n_sims=1000, seed=99)

    pd.testing.assert_frame_equal(mc_run_1, mc_run_2)
    assert not mc_run_1.equals(mc_run_other)


def test_context_dependent_variance_overtaking_and_rainfall() -> None:
    """Verify overtaking_difficulty and wet-weather conditions scale Monte Carlo noise appropriately."""
    dry_low_diff = _make_dummy_features(10)
    dry_low_diff["overtaking_difficulty"] = 0.25

    dry_high_diff = _make_dummy_features(10)
    dry_high_diff["overtaking_difficulty"] = 1.00

    std_low = _resolve_simulation_variance(dry_low_diff, base_std=1.4)
    std_high = _resolve_simulation_variance(dry_high_diff, base_std=1.4)
    std_wet = _resolve_simulation_variance(dry_high_diff, base_std=1.4, weather={"rainfall": True})

    assert np.allclose(std_low, 1.4 * 1.10)
    assert np.allclose(std_high, 1.4 * 0.80)
    assert np.allclose(std_wet, 1.4 * 0.80 * WET_VARIANCE_MULTIPLIER)
    assert np.all(std_low > std_high)
    assert np.all(std_wet > std_high)

    # Low overtaking difficulty & wet weather produce more upsets (lower pole win dominance)
    mc_predictor = AdvancedRacePredictor(base_dnf_prob=0.0)
    pole_high = mc_predictor.predict(dry_high_diff, n_sims=3000, dnf_prob=0.0, seed=42).set_index("Driver")
    pole_low = mc_predictor.predict(dry_low_diff, n_sims=3000, dnf_prob=0.0, seed=42).set_index("Driver")
    pole_wet = mc_predictor.predict(dry_high_diff, n_sims=3000, dnf_prob=0.0, rainfall=True, seed=42).set_index(
        "Driver"
    )

    assert pole_high.loc["DRV_1", "Win %"] > pole_low.loc["DRV_1", "Win %"]
    assert pole_high.loc["DRV_1", "Win %"] > pole_wet.loc["DRV_1", "Win %"]


def test_constructor_calibrated_dnf_rates() -> None:
    """Verify constructor-calibrated and custom DNF rates differentiate driver Finish Rate %."""
    features = pd.DataFrame(
        {
            "driver": ["NOR", "PIA", "PER", "BOT", "ALB", "SAI"],
            "team": ["McLaren", "McLaren", "Cadillac", "Cadillac", "Williams", "Williams"],
            "grid": [1, 2, 3, 4, 5, 6],
            "grid_norm": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "pace_norm": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "pace_consistency": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        }
    )

    predictor = F1MLPredictor(load_historical=False)
    preds = predictor.predict(features, n_sims=4000, seed=42)["predictions"].set_index("Driver")

    mclaren_finish = float(preds.loc[["NOR", "PIA"], "Finish Rate %"].mean())
    cadillac_finish = float(preds.loc[["PER", "BOT"], "Finish Rate %"].mean())
    williams_finish = float(preds.loc[["ALB", "SAI"], "Finish Rate %"].mean())

    assert mclaren_finish > williams_finish
    assert mclaren_finish > cadillac_finish

    # Verify custom per-constructor rates override built-ins
    custom_preds = predictor.predict(
        features,
        n_sims=3000,
        constructor_dnf_rates={"McLaren": 0.01, "Cadillac": 0.40},
        seed=42,
    )["predictions"].set_index("Driver")
    assert float(custom_preds.loc["NOR", "Finish Rate %"]) >= 98.0
    assert float(custom_preds.loc["PER", "Finish Rate %"]) <= 65.0


def test_evaluate_race_backtest() -> None:
    """Verify Predicted vs. Actual backtest columns and metrics on official race results."""
    predictions = pd.DataFrame(
        [
            {"Driver": "NOR", "Grid": 2, "Predicted Pos": 1.0, "Win %": 0.50, "Exp. Points": 18.0},
            {"Driver": "VER", "Grid": 1, "Predicted Pos": 2.0, "Win %": 0.40, "Exp. Points": 16.0},
            {"Driver": "LEC", "Grid": 3, "Predicted Pos": 3.0, "Win %": 0.08, "Exp. Points": 12.0},
            {"Driver": "HAM", "Grid": 4, "Predicted Pos": 4.0, "Win %": 0.02, "Exp. Points": 10.0},
        ]
    )

    no_race_df, no_metrics = evaluate_race_backtest(predictions, None)
    assert no_metrics is None
    assert "Actual Pos" not in no_race_df.columns

    race_session = {
        "results": [
            {"driver": "VER", "position": 1, "status": "Finished"},
            {"driver": "NOR", "position": 2, "status": "Finished"},
            {"driver": "LEC", "position": 3, "status": "+1 Lap"},
            {"driver": "HAM", "position": 4, "status": "Retired"},
        ]
    }
    enriched_df, metrics = evaluate_race_backtest(predictions, race_session)
    assert metrics is not None
    by_drv = enriched_df.set_index("Driver")

    # Predicted NOR P1 -> Actual P2 (-1), VER P2 -> Actual P1 (+1), LEC P3 -> Actual P3 (0), HAM -> DNF
    assert by_drv.loc["VER", "Actual Pos"] == "P1"
    assert by_drv.loc["VER", "Position Delta"] == "+1"
    assert by_drv.loc["NOR", "Actual Pos"] == "P2"
    assert by_drv.loc["NOR", "Position Delta"] == "-1"
    assert by_drv.loc["LEC", "Actual Pos"] == "P3"
    assert by_drv.loc["LEC", "Position Delta"] == "0"
    assert by_drv.loc["HAM", "Actual Pos"] == "DNF"
    assert by_drv.loc["HAM", "Position Delta"] == "—"

    # Errors across 3 classified finishers: |1-2|=1, |2-1|=1, |3-3|=0 -> MAE = 2/3
    assert abs(metrics["weekend_mae"] - (2.0 / 3.0)) < 1e-6
    assert abs(metrics["top10_accuracy"] - 1.0) < 1e-6
    assert metrics["podium_hits"] == 3
    assert metrics["winner_correct"] is False


def test_load_model_metadata(tmp_path: Path) -> None:
    """Verify load_model_metadata reads persisted JSON metadata and handles missing files."""
    assert load_model_metadata(tmp_path / "missing_metadata.json") == {}

    metadata = load_model_metadata()
    assert isinstance(metadata, dict)
    if metadata:
        assert "hyperparameters" in metadata
        assert "feature_names" in metadata
        assert "partitions" in metadata
        assert len(metadata["feature_names"]) == 10
        assert {"train", "validation", "test"}.issubset(set(metadata["partitions"].keys()))


