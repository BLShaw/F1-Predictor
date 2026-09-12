"""Qualifying tab module displaying qualifying and sprint shootout classification and gap charts."""

from typing import Any

import streamlit as st

from src.data_loader import get_qualifying_results
from src.ui.charts import create_qualifying_chart
from src.ui.components import render_empty_state, render_metric_grid, render_section_header
from src.utils.helpers import format_lap_time


def _render_session_classification(
    session_data: dict[str, Any] | None,
    session_label: str = "QUALIFYING",
    is_sprint: bool = False,
) -> None:
    """Render metrics, gap chart, and classification table for a single session."""
    if not session_data:
        render_empty_state(
            title=f"{session_label} TELEMETRY UNAVAILABLE",
            message=f"No {session_label.lower()} session data is currently stored for this Grand Prix event.",
            hint="Download session data using the Data Ingestion Engine in Mission Control.",
        )
        return

    quali_df = get_qualifying_results(session_data)
    if quali_df.empty:
        render_empty_state(
            title=f"{session_label} DATA UNPARSED",
            message=f"{session_label.title()} classification rows could not be parsed from stored session JSON.",
        )
        return

    pole_row = quali_df.iloc[0] if not quali_df.empty else {}
    best_time_val = (
        pole_row.get("q3")
        or pole_row.get("sq3")
        or pole_row.get("q2")
        or pole_row.get("sq2")
        or pole_row.get("q1")
        or pole_row.get("sq1")
    )
    pole_time = format_lap_time(best_time_val)

    weather = session_data.get("weather", {})
    track_temp = weather.get("track_temp", weather.get("air_temp")) if weather else None
    temp_val = f"{float(track_temp):.1f}°C" if track_temp is not None else "—"

    leader_label = "Shootout Leader" if is_sprint else "Pole Position"
    time_label = "Shootout Lap Time" if is_sprint else "Pole Lap Time"
    cars_label = "Shootout Cars" if is_sprint else "Qualifying Cars"

    metrics = [
        {"label": leader_label, "value": str(pole_row.get("driver", "—"))},
        {"label": time_label, "value": pole_time},
        {"label": cars_label, "value": str(len(quali_df))},
        {"label": "Track Temperature", "value": temp_val},
    ]
    render_metric_grid(metrics)

    fig = create_qualifying_chart(quali_df)
    if fig:
        st.plotly_chart(fig, width="stretch")

    grid_title = "SPRINT STARTING GRID" if is_sprint else "STARTING GRID ORDER"
    grid_subtitle = (
        "Official shootout classification with phase intervals"
        if is_sprint
        else "Official classification with Q1, Q2, and Q3 phase intervals"
    )
    render_section_header(grid_title, grid_subtitle)

    display_df = quali_df.copy()

    # Consolidate shootout vs standard qualifying phase columns to prevent collision
    if is_sprint:
        for i in [1, 2, 3]:
            sq_col = f"sq{i}"
            q_col = f"q{i}"
            if sq_col not in display_df.columns and q_col in display_df.columns:
                display_df[sq_col] = display_df[q_col]
            if q_col in display_df.columns and sq_col in display_df.columns and q_col != sq_col:
                display_df = display_df.drop(columns=[q_col])

        rename_map = {
            "position": "Pos",
            "driver": "Driver",
            "team": "Team",
            "sq1": "SQ1",
            "sq2": "SQ2",
            "sq3": "SQ3",
        }
        preferred_cols = ["Pos", "Driver", "Team", "SQ1", "SQ2", "SQ3"]
    else:
        for i in [1, 2, 3]:
            sq_col = f"sq{i}"
            q_col = f"q{i}"
            if q_col not in display_df.columns and sq_col in display_df.columns:
                display_df[q_col] = display_df[sq_col]
            if sq_col in display_df.columns and q_col in display_df.columns and sq_col != q_col:
                display_df = display_df.drop(columns=[sq_col])

        rename_map = {
            "position": "Pos",
            "driver": "Driver",
            "team": "Team",
            "q1": "Q1",
            "q2": "Q2",
            "q3": "Q3",
        }
        preferred_cols = ["Pos", "Driver", "Team", "Q1", "Q2", "Q3"]

    for col in ["q1", "q2", "q3", "sq1", "sq2", "sq3"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_lap_time)

    display_df = display_df.rename(columns=rename_map)
    display_df = display_df.loc[:, ~display_df.columns.duplicated()]

    cols_to_show = [col for col in preferred_cols if col in display_df.columns]
    st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)


def render_qualifying_tab(gp_data: dict[str, Any], is_sprint: bool = False) -> None:
    """Render the Qualifying or Sprint Shootout Results tab."""
    sessions = gp_data.get("sessions", {})
    shootout_session = sessions.get("sprint_qualifying") or sessions.get("sprint_shootout")
    quali_session = sessions.get("qualifying")

    if is_sprint:
        render_section_header(
            "SPRINT SHOOTOUT CLASSIFICATION",
            "Official shootout classification and grid positions for the sprint competition",
        )

        if shootout_session and quali_session:
            sub_tabs = st.tabs(["Sprint Shootout", "Grand Prix Qualifying"])
            with sub_tabs[0]:
                _render_session_classification(
                    shootout_session, session_label="SPRINT SHOOTOUT", is_sprint=True
                )
            with sub_tabs[1]:
                _render_session_classification(
                    quali_session, session_label="QUALIFYING", is_sprint=False
                )
        elif shootout_session:
            _render_session_classification(
                shootout_session, session_label="SPRINT SHOOTOUT", is_sprint=True
            )
        elif quali_session:
            _render_session_classification(
                quali_session, session_label="SPRINT SHOOTOUT", is_sprint=True
            )
        else:
            _render_session_classification(
                None, session_label="SPRINT SHOOTOUT", is_sprint=True
            )
    else:
        render_section_header(
            "QUALIFYING CLASSIFICATION",
            "Official grid positions and knockout elimination times",
        )
        _render_session_classification(
            quali_session, session_label="QUALIFYING", is_sprint=False
        )
