"""Predictions tab module providing ML predictions, Monte Carlo simulation, and SHAP explainability."""

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import build_weekend_features, resolve_starting_grid
from src.model import AdvancedRacePredictor, F1MLPredictor
from src.ui.charts import _apply_chart_theme, create_win_probability_chart
from src.ui.components import (
    render_empty_state,
    render_metric_grid,
    render_model_specification,
    render_notice,
    render_section_header,
)
from src.ui.state import get_cached_model

logger = logging.getLogger(__name__)


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

        render_section_header(
            "PROJECTED CLASSIFICATION MATRIX",
            "Ranked expected points, podium likelihood, and average finishing positions",
        )
        display_df = predictions.copy()
        display_df["Win %"] = (display_df["Win %"] * 100).round(1).astype(str) + "%"
        display_df["Podium %"] = (display_df["Podium %"] * 100).round(1).astype(str) + "%"
        display_df["Points %"] = (display_df["Points %"] * 100).round(1).astype(str) + "%"
        display_df["Exp. Points"] = display_df["Exp. Points"].round(2)
        display_df["Avg Finish"] = display_df["Avg Finish"].round(1)

        st.dataframe(display_df, width="stretch", hide_index=True)

        csv = predictions.to_csv(index=False)
        folder = selected_gp.get("folder", "grand_prix")
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M")
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
                        "Feature Identifier": "pace_norm",
                        "Input Description": "Practice flying lap pace delta relative to session fastest lap",
                        "Performance Interpretation": "Lower is better (0.0 = Fastest Practice Lap)",
                    },
                    {
                        "Feature Identifier": "pace_consistency",
                        "Input Description": "Standard deviation of representative clean flying laps",
                        "Performance Interpretation": "Lower is better (tighter stint variance)",
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
