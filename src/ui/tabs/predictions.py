"""Predictions tab module providing ML predictions, Monte Carlo simulation, and SHAP explainability."""

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import build_weekend_features, get_race_results, resolve_starting_grid
from src.model import AdvancedRacePredictor, F1MLPredictor
from src.ui.charts import _apply_chart_theme, create_shap_waterfall_chart, create_win_probability_chart
from src.ui.components import (
    render_empty_state,
    render_metric_grid,
    render_model_specification,
    render_notice,
    render_section_header,
)
from src.ui.state import get_cached_model, get_cached_model_metadata

logger = logging.getLogger(__name__)


def evaluate_race_backtest(
    predictions: pd.DataFrame,
    race_session: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Compare race predictions against official classified race results when available."""
    if predictions.empty or "Driver" not in predictions.columns:
        return predictions.copy(), None

    race_df = get_race_results(race_session)
    if race_df.empty or "driver" not in race_df.columns:
        return predictions.copy(), None

    actual_map: dict[str, int | None] = {}
    for _, row in race_df.iterrows():
        drv = str(row["driver"]).strip()
        if not drv:
            continue
        status = str(row.get("status", "")).strip()
        is_classified_status = status == "Finished" or status.startswith("+") or "Lap" in status
        pos_val = pd.to_numeric(row.get("position"), errors="coerce")
        if is_classified_status and pd.notna(pos_val) and int(pos_val) > 0:
            actual_map[drv] = int(pos_val)
        else:
            actual_map[drv] = None

    if not actual_map:
        return predictions.copy(), None

    enriched = predictions.copy().reset_index(drop=True)
    has_pred_pos = "Predicted Pos" in enriched.columns
    has_avg_finish = "Avg Finish" in enriched.columns
    has_grid = "Grid" in enriched.columns

    actual_pos_labels: list[str] = []
    pos_deltas: list[str] = []
    abs_errors: list[float] = []
    grid_abs_errors: list[float] = []

    for idx, row in enriched.iterrows():
        drv = str(row["Driver"]).strip()
        if drv not in actual_map:
            actual_pos_labels.append("—")
            pos_deltas.append("—")
            continue

        actual_pos = actual_map[drv]
        if actual_pos is None:
            actual_pos_labels.append("DNF")
            pos_deltas.append("—")
            continue

        actual_pos_labels.append(f"P{actual_pos}")
        if has_pred_pos and pd.notna(row.get("Predicted Pos")):
            pred_pos_val = float(row["Predicted Pos"])
        elif has_avg_finish and pd.notna(row.get("Avg Finish")):
            pred_pos_val = float(row["Avg Finish"])
        else:
            pred_pos_val = float(idx + 1)

        delta = int(round(pred_pos_val)) - actual_pos
        pos_deltas.append(f"{delta:+d}" if delta != 0 else "0")
        abs_errors.append(abs(pred_pos_val - actual_pos))

        if has_grid and pd.notna(row.get("Grid")):
            grid_abs_errors.append(abs(float(row["Grid"]) - actual_pos))

    if not abs_errors:
        return predictions.copy(), None

    enriched["Actual Pos"] = actual_pos_labels
    enriched["Position Delta"] = pos_deltas

    classified_sorted = [
        drv for drv, pos in sorted(((d, p) for d, p in actual_map.items() if p is not None), key=lambda item: item[1])
    ]
    actual_top10 = set(classified_sorted[:10])
    actual_top3 = set(classified_sorted[:3])

    predicted_drivers = [str(d).strip() for d in enriched["Driver"]]
    top10_hits = len(actual_top10 & set(predicted_drivers[:10]))
    podium_hits = len(actual_top3 & set(predicted_drivers[:3]))

    actual_winner = classified_sorted[0]
    predicted_winner = predicted_drivers[0] if predicted_drivers else ""

    metrics: dict[str, Any] = {
        "weekend_mae": float(sum(abs_errors) / len(abs_errors)),
        "grid_baseline_mae": float(sum(grid_abs_errors) / len(grid_abs_errors)) if grid_abs_errors else None,
        "top10_accuracy": float(top10_hits / max(len(actual_top10), 1)),
        "top10_hits": top10_hits,
        "top10_total": len(actual_top10),
        "podium_hits": podium_hits,
        "podium_total": len(actual_top3),
        "winner_correct": bool(predicted_winner == actual_winner),
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
        "classified_count": len(abs_errors),
    }
    return enriched, metrics


def render_predictions_tab(gp_data: dict[str, Any], selected_gp: dict[str, Any]) -> None:
    """Render the Machine Learning Predictions and SHAP Insights tab."""
    render_section_header(
        "RACE OUTCOME PREDICTION PIPELINE",
        "Supervised Random Forest regression combined with calibrated stochastic Monte Carlo simulation",
    )

    sessions = gp_data.get("sessions", {})
    has_pace = bool(sessions.get("fp1") or sessions.get("fp2") or sessions.get("fp3"))

    grid_df = resolve_starting_grid(gp_data)
    has_grid = not grid_df.empty
    ml_ready = has_grid

    status_metrics = [
        {
            "label": "Practice Pace",
            "value": "AVAILABLE" if has_pace else "IMPUTED",
            "delta": "Telemetry Synced" if has_pace else "Grid Rank Fallback",
        },
        {
            "label": "Starting Grid",
            "value": "CONFIRMED" if has_grid else "PENDING",
            "delta": "Official Order" if has_grid else "Qualifying Required",
        },
        {
            "label": "Pipeline Status",
            "value": "OPERATIONAL" if ml_ready else "BLOCKED",
            "delta": "Ready to Predict" if ml_ready else "Awaiting Grid Data",
        },
    ]
    render_metric_grid(status_metrics)

    if not has_grid:
        render_empty_state(
            title="STARTING GRID REQUIRED",
            message="Confirmed grid positions are required to calculate race projections and feature matrices.",
            hint="Download Qualifying or Sprint sessions from Mission Control to proceed.",
        )
        return

    with st.expander("SIMULATION & MODEL PARAMETERS", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_sims = st.slider("Monte Carlo Simulations", 500, 10000, 2000, 500)
        with col2:
            dnf_prob = st.slider("Base DNF Probability (%)", 1, 15, 5) / 100.0
        with col3:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            use_ml = st.checkbox("Enable Machine Learning Regressor", value=True)


    if st.button("EXECUTE PREDICTION ENGINE", type="primary", width="stretch"):
        with st.spinner(
            "Engineering features, computing Random Forest outputs, and running calibrated Monte Carlo simulation..."
        ):
            try:
                features = build_weekend_features(gp_data)
                if features.empty:
                    st.error("Failed to build feature frame from available session telemetry.")
                    return

                if use_ml:
                    cached_model = get_cached_model()
                    predictor = F1MLPredictor(model=cached_model)
                    results = predictor.predict(features, n_sims=n_sims, dnf_prob=dnf_prob)
                    st.session_state["ml_results"] = results
                    st.session_state["predictions"] = results["predictions"]
                    st.session_state["predictor_instance"] = predictor
                else:
                    predictor_mc = AdvancedRacePredictor()
                    predictor_mc.base_dnf_prob = dnf_prob
                    predictions = predictor_mc.predict(features, n_sims=n_sims)
                    st.session_state["ml_results"] = None
                    st.session_state["predictions"] = predictions
                    st.session_state["predictor_instance"] = None

                st.session_state["prediction_params"] = {
                    "n_sims": n_sims,
                    "dnf_prob": dnf_prob,
                    "use_ml": use_ml,
                }
                st.success("Simulation complete.")
            except Exception as e:
                logger.exception("Prediction generation failed.")
                st.error(f"Prediction run encountered an error: {e}")

    if "predictions" not in st.session_state:
        return

    predictions: pd.DataFrame = st.session_state["predictions"]
    ml_results: dict[str, Any] | None = st.session_state.get("ml_results")
    params: dict[str, Any] = st.session_state.get("prediction_params", {})

    model_type = ml_results.get("model_type", "Monte Carlo") if ml_results else "Monte Carlo Simulation"
    sim_count = params.get("n_sims", 2000)

    render_section_header(
        "PREDICTION ANALYSIS & PROBABILISTIC PROJECTIONS",
        f"Engine: {model_type.upper()} | {sim_count:,} Calibrated Iterations",
    )

    pred_tabs = st.tabs(["Win & Podium Projections", "SHAP Feature Sensitivity", "Model Architecture"])

    # Tab 1: Predictions
    with pred_tabs[0]:
        fig = create_win_probability_chart(predictions, team_data=grid_df)
        if fig:
            st.plotly_chart(fig, width="stretch")

        enriched_preds, backtest_metrics = evaluate_race_backtest(predictions, sessions.get("race"))

        if backtest_metrics is not None:
            render_section_header(
                "PREDICTED VS. ACTUAL BACKTEST",
                "Model accuracy evaluation against official classified Grand Prix results",
            )
            grid_mae = backtest_metrics.get("grid_baseline_mae")
            backtest_cards = [
                {
                    "label": "Weekend MAE",
                    "value": f"{backtest_metrics['weekend_mae']:.2f} Pos",
                    "delta": f"Grid MAE: {grid_mae:.2f}" if grid_mae is not None else None,
                },
                {
                    "label": "Top-10 Hit Accuracy",
                    "value": f"{backtest_metrics['top10_accuracy'] * 100:.1f}%",
                    "delta": f"{backtest_metrics['top10_hits']}/{backtest_metrics['top10_total']} Finishers",
                },
                {
                    "label": "Podium Hits",
                    "value": f"{backtest_metrics['podium_hits']}/{backtest_metrics['podium_total']}",
                },
                {
                    "label": "Winner Prediction",
                    "value": "CORRECT" if backtest_metrics["winner_correct"] else "MISSED",
                    "delta": f"Actual P1: {backtest_metrics['actual_winner']} (Pred: {backtest_metrics['predicted_winner']})",
                },
            ]
            render_metric_grid(backtest_cards)

        render_section_header(
            "PROJECTED CLASSIFICATION MATRIX",
            "Ranked expected points, podium likelihood, and average finishing positions",
        )
        display_df = enriched_preds.copy()
        display_df["Win %"] = (display_df["Win %"] * 100).round(1).astype(str) + "%"
        display_df["Podium %"] = (display_df["Podium %"] * 100).round(1).astype(str) + "%"
        display_df["Points %"] = (display_df["Points %"] * 100).round(1).astype(str) + "%"
        display_df["Exp. Points"] = display_df["Exp. Points"].round(2)
        display_df["Avg Finish"] = display_df["Avg Finish"].round(1)

        st.dataframe(display_df, width="stretch", hide_index=True)

        csv = enriched_preds.to_csv(index=False)
        folder = selected_gp.get("folder", "grand_prix")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        st.download_button(
            "Download Prediction Matrix (CSV)",
            data=csv,
            file_name=f"f1_prediction_{folder}_{timestamp}.csv",
            mime="text/csv",
        )

    # Tab 2: Feature Importance (SHAP)
    with pred_tabs[1]:
        if ml_results and ml_results.get("feature_importance") is not None:
            render_notice(
                "SHAP EXPLAINABILITY VALUES",
                "TreeSHAP computes the exact marginal contribution of each feature toward the predicted finishing position relative to expected grid baselines.",
                variant="neutral",
            )

            feat_imp: pd.DataFrame = ml_results["feature_importance"]
            fig_shap = go.Figure()

            fig_shap.add_trace(
                go.Bar(
                    y=feat_imp["Feature"],
                    x=feat_imp["Importance"],
                    orientation="h",
                    marker={"color": "#38BDF8", "line": {"color": "rgba(255,255,255,0.25)", "width": 1}},
                    text=[f"{v:.3f}" for v in feat_imp["Importance"]],
                    textposition="outside",
                    textfont={"family": "Orbitron, monospace", "size": 10, "color": "#F8FAFC"},
                    hovertemplate="<b>%{y}</b><br>Mean |SHAP| Sensitivity: %{x:.4f}<extra></extra>",
                )
            )

            fig_shap.update_xaxes(title="Mean |SHAP| Value")
            fig_shap.update_yaxes(autorange="reversed")
            _apply_chart_theme(fig_shap, "MEAN |SHAP| FEATURE IMPACT ON POSITION", height=320)
            st.plotly_chart(fig_shap, width="stretch")

            render_section_header(
                "FEATURE SENSITIVITY DEFINITIONS", "Descriptions of model inputs and performance implications"
            )
            feature_desc = pd.DataFrame(
                [
                    {
                        "Feature Identifier": "grid_norm",
                        "Input Description": "Starting grid position normalized to [0, 1] interval",
                        "Performance Interpretation": "Lower is better (0.0 = Pole Position)",
                    },
                    {
                        "Feature Identifier": "quali_gap_norm",
                        "Input Description": "Best qualifying lap delta to pole normalized to qualifying spread",
                        "Performance Interpretation": "Lower is better (0.0 = Pole Qualifying Lap)",
                    },
                    {
                        "Feature Identifier": "pace_norm",
                        "Input Description": "Practice flying lap pace delta relative to session fastest lap",
                        "Performance Interpretation": "Lower is better (0.0 = Fastest Practice Lap)",
                    },
                    {
                        "Feature Identifier": "pace_consistency",
                        "Input Description": "Standard deviation of representative clean flying laps",
                        "Performance Interpretation": "Lower is better (tighter stint variance)",
                    },
                    {
                        "Feature Identifier": "driver_form",
                        "Input Description": "Pre-race EWMA of driver's recent classified finishes (last 5 races)",
                        "Performance Interpretation": "Lower is better (0.0 = P1 form)",
                    },
                    {
                        "Feature Identifier": "constructor_form",
                        "Input Description": "Pre-race EWMA of constructor's recent classified finishes",
                        "Performance Interpretation": "Lower is better (0.0 = Top team form)",
                    },
                    {
                        "Feature Identifier": "overtaking_difficulty",
                        "Input Description": "Circuit overtaking difficulty score from tracks.json (0.25–1.00)",
                        "Performance Interpretation": "Higher increases starting grid importance",
                    },
                    {
                        "Feature Identifier": "is_street_circuit",
                        "Input Description": "Binary indicator for street circuits (1=Street, 0=Permanent)",
                        "Performance Interpretation": "Captures street-track attrition and passing constraints",
                    },
                    {
                        "Feature Identifier": "has_practice_data",
                        "Input Description": "Indicator for recorded practice telemetry (1=Available, 0=Imputed)",
                        "Performance Interpretation": "Flags fallback imputation on missed practice",
                    },
                    {
                        "Feature Identifier": "is_sprint",
                        "Input Description": "Binary indicator for Sprint weekend format",
                        "Performance Interpretation": "Adjusts for reduced practice / parc ferme setup constraints",
                    },
                ]
            )
            st.dataframe(feature_desc, width="stretch", hide_index=True)

            # Driver-level marginal contribution breakdown
            predictor_inst = st.session_state.get("predictor_instance")
            features_used = ml_results.get("features_used")
            if predictor_inst is not None and features_used is not None and "driver" in features_used.columns:
                driver_list = list(features_used["driver"])
                render_section_header(
                    "DRIVER MARGINAL CONTRIBUTION",
                    "Driver-level TreeSHAP marginal impact breakdown relative to expected grid baseline",
                )
                selected_driver = st.selectbox(
                    "Select Driver for Marginal SHAP Breakdown",
                    options=driver_list,
                    key="shap_driver_select",
                )
                if selected_driver:
                    drv_idx = driver_list.index(selected_driver)
                    drv_expl = predictor_inst.get_driver_explanation(drv_idx, features_used)
                    if drv_expl and drv_expl.get("feature_contributions"):
                        fig_waterfall = create_shap_waterfall_chart(drv_expl)
                        if fig_waterfall is not None:
                            st.plotly_chart(fig_waterfall, width="stretch")

                        expl_rows = [
                            {
                                "Feature": item["feature"],
                                "Normalized Value": item["value"],
                                "SHAP Impact": f"{item['shap_contribution']:+.3f}",
                                "Effect": item["impact"],
                                "Description": item["description"],
                            }
                            for item in drv_expl["feature_contributions"]
                        ]
                        st.dataframe(pd.DataFrame(expl_rows), width="stretch", hide_index=True)
        else:
            if ml_results and ml_results.get("shap_error"):
                st.error(f"SHAP calculation encountered an issue: {ml_results['shap_error']}")
            else:
                render_empty_state(
                    title="SHAP EXPLAINABILITY INACTIVE",
                    message="SHAP diagnostics require the Machine Learning Regressor to be enabled during simulation.",
                )

    # Tab 3: Model Architecture
    with pred_tabs[2]:
        render_section_header("TECHNICAL SPECIFICATION", "Hyperparameters, training splits, and simulation calibration")
        render_model_specification(
            model_type="RandomForestRegressor",
            simulations=sim_count,
            dnf_rate=params.get("dnf_prob", 0.05),
            train_seasons="2022-2024",
            test_season="2026",
            metadata=get_cached_model_metadata(),
        )

        if ml_results and ml_results.get("features_used") is not None:
            render_section_header(
                "ENGINEERED WEEKEND FEATURES", "Current event feature vector fed into prediction regressor"
            )
            features_df = ml_results["features_used"]
            display_feat = features_df.copy()
            for col in display_feat.columns:
                if col != "driver" and display_feat[col].dtype in ["float64", "float32"]:
                    display_feat[col] = display_feat[col].round(4)
            st.dataframe(display_feat, width="stretch", hide_index=True)
