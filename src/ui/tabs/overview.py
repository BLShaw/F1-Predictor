"""Overview tab module displaying Grand Prix metadata, weather, and session summaries."""

from typing import Any

import pandas as pd
import streamlit as st

from src.data_loader import get_drivers_from_session
from src.ui.components import render_empty_state, render_metric_grid, render_section_header
from src.utils.helpers import format_lap_time


def render_overview_tab(gp_data: dict[str, Any], selected_gp: dict[str, Any]) -> None:
    """Render the Grand Prix Overview tab."""
    render_section_header("SESSION OVERVIEW", "Telemetry status, track metrics, and weekend conditions")

    sessions = gp_data.get("sessions", {})
    available_sessions = sum(1 for v in selected_gp.get("sessions", {}).values() if v)

    driver_count = 0
    for session_type in ["qualifying", "race", "fp1", "fp2"]:
        if session_type in sessions:
            drivers = get_drivers_from_session(sessions[session_type])
            if not drivers.empty:
                driver_count = len(drivers)
                break

    weather: dict[str, Any] = {}
    for s in sessions.values():
        if s and "weather" in s and s["weather"]:
            weather = s["weather"]
            break

    track_temp = weather.get("track_temp", weather.get("air_temp")) if weather else None
    temp_val = f"{float(track_temp):.1f}°C" if track_temp is not None else "—"

    if weather:
        is_rain = weather.get("rainfall", False)
        condition_val = "Wet" if is_rain else "Dry"
    else:
        condition_val = "—"

    metrics = [
        {
            "label": "Sessions Stored",
            "value": f"{available_sessions}/8",
            "delta": "Complete" if available_sessions >= 5 else "Incomplete",
        },
        {
            "label": "Confirmed Grid",
            "value": f"{driver_count} Drivers" if driver_count else "—",
        },
        {
            "label": "Track Temperature",
            "value": temp_val,
        },
        {
            "label": "Track Conditions",
            "value": condition_val,
        },
    ]
    render_metric_grid(metrics)

    render_section_header("WEEKEND SESSION SUMMARY", "Pacesetters and best recorded lap times by session")

    summary_data: list[dict[str, str]] = []
    for session_type, session in sessions.items():
        if session:
            best_times = session.get("best_times", {})
            fastest_driver = min(best_times, key=best_times.get) if best_times else "—"
            fastest_time = format_lap_time(min(best_times.values())) if best_times else "—"

            summary_data.append(
                {
                    "Session": session_type.upper().replace("_", " "),
                    "Status": "Complete",
                    "Pacesetter": str(fastest_driver),
                    "Best Lap": fastest_time,
                    "Session Date": str(session.get("date", "—")),
                }
            )

    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)
    else:
        render_empty_state(
            title="NO SESSIONS RECORDED",
            message="No session telemetry is currently stored for this Grand Prix event.",
            hint="Download sessions using the Data Ingestion Engine in Mission Control.",
        )
