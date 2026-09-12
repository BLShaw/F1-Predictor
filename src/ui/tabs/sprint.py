"""Sprint tab module displaying sprint race classification and championship points."""

from typing import Any

import pandas as pd
import streamlit as st

from src.data_loader import get_race_results
from src.ui.components import (
    render_empty_state,
    render_notice,
    render_podium,
    render_section_header,
)

SPRINT_POINTS: dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def render_sprint_tab(gp_data: dict[str, Any], selected_season: int | str) -> None:
    """Render the Sprint Race classification and points tab."""
    render_section_header(
        "SPRINT RACE CLASSIFICATION", "Official finishing order and sprint championship points (8-7-6-5-4-3-2-1)"
    )

    sessions = gp_data.get("sessions", {})
    sprint_session = sessions.get("sprint")

    try:
        year_val = int(selected_season)
    except (ValueError, TypeError):
        year_val = int(gp_data.get("year", 0))

    if year_val and year_val <= 2022:
        render_notice(
            "2021–2022 SPRINT FORMAT",
            "Under 2021–2022 sporting regulations, the Saturday Sprint race finishing order determined the Sunday Grand Prix starting grid.",
            variant="info",
        )

    if sprint_session:
        sprint_df = get_race_results(sprint_session)

        if not sprint_df.empty:
            sprint_df["sprint_points"] = sprint_df["position"].apply(
                lambda x: SPRINT_POINTS.get(int(x), 0) if pd.notna(x) else 0
            )

            render_section_header(
                "SPRINT PODIUM FINISHERS", "Top 3 classified finishers awarding 8-7-6 championship points"
            )
            render_podium(sprint_df, points_key="sprint_points")

            render_section_header(
                "FULL SPRINT CLASSIFICATION", "Official finishing order and sprint points awarded"
            )
            display_df = sprint_df.rename(
                columns={
                    "position": "Pos",
                    "driver": "Driver",
                    "team": "Team",
                    "grid_position": "Grid",
                    "status": "Status",
                    "sprint_points": "Points",
                }
            )

            cols_to_show = [
                c for c in ["Pos", "Driver", "Team", "Grid", "Status", "Points"] if c in display_df.columns
            ]
            st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)
        else:
            render_empty_state("SPRINT RESULTS EMPTY", "Sprint classification could not be extracted.")
    else:
        render_empty_state(
            title="SPRINT TELEMETRY MISSING",
            message="No Sprint Race data is stored for this Grand Prix event.",
            hint="Download the Sprint session via scripts/download_historical_data.py or Mission Control.",
        )

