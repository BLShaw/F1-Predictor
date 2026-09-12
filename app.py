"""Formula 1 Predictor - Interactive Dashboard.

Modular Streamlit application providing session pace analysis, starting grid
resolutions, historical models, and calibrated Monte Carlo race simulations.
"""

from pathlib import Path

import streamlit as st

from src.ui.components import render_app_header, render_empty_state, render_gp_banner
from src.ui.sidebar import render_sidebar
from src.ui.state import get_cached_gp_data
from src.ui.tabs import (
    render_overview_tab,
    render_practice_tab,
    render_predictions_tab,
    render_qualifying_tab,
    render_race_tab,
    render_sprint_tab,
)

st.set_page_config(
    page_title="Formula 1 Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject centralized design system styling
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path, encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def main() -> None:
    """Render main application layout, navigation, and active session tabs."""
    render_app_header()

    selected_season, selected_gp = render_sidebar()

    if not selected_season or not selected_gp:
        render_empty_state(
            title="AWAITING TELEMETRY SELECTION",
            message="Select a championship season and Grand Prix event from Mission Control in the sidebar to begin analysis.",
            hint="To fetch telemetry from terminal: python scripts/download_historical_data.py --years 2025 2026",
        )
        return

    gp_data = get_cached_gp_data(selected_season, selected_gp["folder"])
    sessions = gp_data.get("sessions", {})

    is_sprint_weekend = bool(
        sessions.get("sprint") or sessions.get("sprint_qualifying") or sessions.get("sprint_shootout")
    )

    render_gp_banner(
        name=selected_gp.get("name", "Grand Prix"),
        round_num=selected_gp.get("round", "—"),
        season=selected_season,
        is_sprint=is_sprint_weekend,
    )

    if is_sprint_weekend:
        tab_names = ["Overview", "Practice", "Sprint Shootout", "Sprint Race", "Race", "Predict"]
    else:
        tab_names = ["Overview", "Practice", "Qualifying", "Race", "Predict"]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_overview_tab(gp_data, selected_gp)

    with tabs[1]:
        render_practice_tab(gp_data, is_sprint_weekend)

    with tabs[2]:
        render_qualifying_tab(gp_data, is_sprint=is_sprint_weekend)

    if is_sprint_weekend:
        with tabs[3]:
            render_sprint_tab(gp_data, selected_season)
        with tabs[4]:
            render_race_tab(gp_data)
        with tabs[5]:
            render_predictions_tab(gp_data, selected_gp)
    else:
        with tabs[3]:
            render_race_tab(gp_data)
        with tabs[4]:
            render_predictions_tab(gp_data, selected_gp)


if __name__ == "__main__":
    main()
