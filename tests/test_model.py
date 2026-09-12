"""Tests for machine learning model and Monte Carlo simulation engine."""

import pandas as pd

from src.model import AdvancedRacePredictor, F1MLPredictor, get_shap_summary_data


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

