"""Race tab module displaying Grand Prix classification, podium cards, and points."""

from typing import Any

import streamlit as st

from src.data_loader import get_race_results
from src.ui.components import render_empty_state, render_podium, render_section_header


def render_race_tab(gp_data: dict[str, Any]) -> None:
    """Render the Grand Prix Race Results tab."""
    render_section_header(
        "GRAND PRIX CLASSIFICATION", "Official race results, finishing intervals, and points allocation"
    )

    sessions = gp_data.get("sessions", {})
    if "race" not in sessions or not sessions["race"]:
        render_empty_state(
            title="RACE TELEMETRY UNAVAILABLE",
            message="No Grand Prix race session telemetry is stored for this event.",
            hint="Download the Race session from Mission Control once the event concludes.",
        )
        return

    race_df = get_race_results(sessions["race"])
    if race_df.empty:
        render_empty_state(
            title="RACE RESULTS EMPTY",
            message="Race classification rows could not be parsed from stored session JSON.",
        )
        return

    render_section_header("PODIUM CELEBRATION", "Top 3 classified finishers on the podium")
    render_podium(race_df, points_key="points")

    render_section_header("OFFICIAL CLASSIFICATION", "Complete finishing order, starting grid delta, and points earned")
    display_df = race_df.rename(
        columns={
            "position": "Pos",
            "driver": "Driver",
            "team": "Team",
            "grid_position": "Grid",
            "status": "Status",
            "points": "Points",
        }
    )

    cols_to_show = [c for c in ["Pos", "Driver", "Team", "Grid", "Status", "Points"] if c in display_df.columns]
    st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)
