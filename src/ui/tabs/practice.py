"""Practice tab module displaying practice session pace analysis and charts."""

from typing import Any

import pandas as pd
import streamlit as st

from src.ui.charts import create_pace_chart
from src.ui.components import render_empty_state, render_metric_grid, render_notice, render_section_header
from src.utils.helpers import format_gap, format_lap_time


def render_practice_tab(gp_data: dict[str, Any], is_sprint_weekend: bool) -> None:
    """Render the Practice Analysis tab."""
    render_section_header("PRACTICE PACE ANALYSIS", "Flying lap deltas, fastest stints, and track metrics")

    sessions = gp_data.get("sessions", {})
    available_fp: list[tuple[str, str, dict[str, Any]]] = []

    if sessions.get("fp1"):
        available_fp.append(("Practice 1", "fp1", sessions["fp1"]))
    if not is_sprint_weekend:
        if sessions.get("fp2"):
            available_fp.append(("Practice 2", "fp2", sessions["fp2"]))
        if sessions.get("fp3"):
            available_fp.append(("Practice 3", "fp3", sessions["fp3"]))

    if not available_fp:
        render_empty_state(
            title="PRACTICE TELEMETRY UNAVAILABLE",
            message="No Free Practice session telemetry is currently stored for this Grand Prix.",
            hint="Download practice sessions using the Data Ingestion Engine in Mission Control.",
        )
        if is_sprint_weekend:
            render_notice(
                "SPRINT WEEKEND FORMAT",
                "Sprint format events feature only a single practice session (FP1) before competitive sessions.",
                variant="warning",
            )
        return

    if is_sprint_weekend:
        render_notice(
            "SPRINT FORMAT SCHEDULE",
            "Under sprint weekend regulations, FP1 is the sole practice session before competitive sessions.",
            variant="warning",
        )

    fp_tab_names = [name for name, _, _ in available_fp]
    fp_tabs = st.tabs(fp_tab_names)

    for fp_tab, (tab_name, _session_key, session_data) in zip(fp_tabs, available_fp):
        with fp_tab:
            best_times = session_data.get("best_times", {})
            weather = session_data.get("weather", {})
            session_date = str(session_data.get("date", "—"))

            fastest_driver = min(best_times, key=best_times.get) if best_times else "—"
            fastest_time = format_lap_time(min(best_times.values())) if best_times else "—"
            temp = weather.get("track_temp", weather.get("air_temp")) if weather else None
            temp_display = f"{float(temp):.1f}°C" if temp is not None else "—"

            metrics = [
                {"label": "Pacesetter", "value": fastest_driver},
                {"label": "Fastest Lap", "value": fastest_time},
                {"label": "Track Temp", "value": temp_display},
                {"label": "Session Date", "value": session_date},
            ]
            render_metric_grid(metrics)

            if best_times:
                pace_df = pd.DataFrame([{"driver": d, "best": t} for d, t in best_times.items()])
                fig = create_pace_chart(pace_df, session_title=f"{tab_name.upper()} PACE SPREAD")
                if fig:
                    st.plotly_chart(fig, width="stretch")

                render_section_header(
                    f"{tab_name.upper()} CLASSIFICATION", "Ranked representative lap times and intervals to leader"
                )
                display_df = pace_df.sort_values("best").reset_index(drop=True)
                fastest_lap = display_df["best"].min()
                display_df["gap"] = display_df["best"] - fastest_lap
                display_df["position"] = range(1, len(display_df) + 1)
                display_df["Best Lap"] = display_df["best"].apply(format_lap_time)
                display_df["Interval"] = display_df["gap"].apply(lambda g: format_gap(g) if g > 0 else "LEADER")

                renamed = display_df[["position", "driver", "Best Lap", "Interval"]].rename(
                    columns={
                        "position": "Pos",
                        "driver": "Driver",
                    }
                )
                st.dataframe(renamed, width="stretch", hide_index=True)
            else:
                render_empty_state(
                    title="NO LAP TIMES RECORDED",
                    message=f"No valid lap times could be parsed for {tab_name}.",
                )
