"""Sprint tab module displaying sprint qualifying/shootout and sprint race results."""

from typing import Any

import pandas as pd
import streamlit as st

from src.data_loader import get_qualifying_results, get_race_results
from src.ui.components import (
    render_empty_state,
    render_metric_grid,
    render_notice,
    render_podium,
    render_section_header,
)
from src.utils.helpers import format_lap_time

SPRINT_POINTS: dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def render_sprint_tab(gp_data: dict[str, Any], selected_season: int | str) -> None:
    """Render the Sprint tab with shootout and race sub-tabs."""
    render_section_header("SPRINT WEEKEND COMPETITION", "Sprint Shootout grid and Saturday sprint race classification")

    sessions = gp_data.get("sessions", {})
    sprint_sub_tabs = st.tabs(["Sprint Shootout", "Sprint Race"])

    # 1. Sprint Qualifying / Shootout
    with sprint_sub_tabs[0]:
        sprint_quali_data = sessions.get("sprint_qualifying") or sessions.get("sprint_shootout")

        if sprint_quali_data:
            sq_results = get_qualifying_results(sprint_quali_data)
            has_quali_times = not sq_results.empty and any(
                col in sq_results.columns for col in ["q1", "q2", "q3", "sq1", "sq2", "sq3"]
            )
            has_laps_only = not has_quali_times and bool(
                sprint_quali_data.get("best_times") or sprint_quali_data.get("laps")
            )

            if has_quali_times:
                session_date = str(sprint_quali_data.get("date", "—"))
                weather = sprint_quali_data.get("weather", {})
                temp = weather.get("track_temp", weather.get("air_temp")) if weather else None
                temp_val = f"{float(temp):.1f}°C" if temp is not None else "—"
                is_rain = weather.get("rainfall", False) if weather else False

                metrics = [
                    {"label": "Shootout Date", "value": session_date},
                    {"label": "Track Temp", "value": temp_val},
                    {"label": "Weather Condition", "value": "Wet" if is_rain else "Dry" if weather else "—"},
                ]
                render_metric_grid(metrics)

                render_section_header("SPRINT STARTING GRID", "Shootout classification and qualifying intervals")
                display_df = sq_results.copy()
                for i in [1, 2, 3]:
                    sq_col = f"sq{i}"
                    q_col = f"q{i}"
                    if sq_col not in display_df.columns and q_col in display_df.columns:
                        display_df[sq_col] = display_df[q_col]
                    if q_col in display_df.columns and sq_col in display_df.columns and q_col != sq_col:
                        display_df = display_df.drop(columns=[q_col])

                for col in ["q1", "q2", "q3", "sq1", "sq2", "sq3"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(format_lap_time)

                display_df = display_df.rename(
                    columns={
                        "position": "Pos",
                        "driver": "Driver",
                        "team": "Team",
                        "sq1": "SQ1",
                        "sq2": "SQ2",
                        "sq3": "SQ3",
                    }
                )
                display_df = display_df.loc[:, ~display_df.columns.duplicated()]

                cols_to_show = [c for c in ["Pos", "Driver", "Team", "SQ1", "SQ2", "SQ3"] if c in display_df.columns]
                st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)

            elif has_laps_only:
                render_notice(
                    "2021–2022 SPRINT FORMAT",
                    "Under legacy sprint rules, Friday Qualifying set the Sprint grid, and the Saturday Sprint decided the Sunday Grand Prix grid.",
                    variant="info",
                )

                best_times = sprint_quali_data.get("best_times", {})
                if best_times:
                    render_section_header("SESSION FASTEST LAPS", "Ranked representative lap times")
                    times_df = pd.DataFrame(
                        [
                            {"Driver": driver, "Best Lap": format_lap_time(time)}
                            for driver, time in sorted(best_times.items(), key=lambda x: x[1])
                        ]
                    )
                    times_df.insert(0, "Pos", range(1, len(times_df) + 1))
                    st.dataframe(times_df, width="stretch", hide_index=True)
            else:
                render_empty_state("SHOOTOUT DATA EMPTY", "Sprint Qualifying results could not be parsed.")
        else:
            try:
                year_val = int(selected_season)
            except (ValueError, TypeError):
                year_val = int(gp_data.get("year", 0))

            if year_val and year_val <= 2022:
                render_notice(
                    "2021–2022 SPRINT FORMAT",
                    "This event used the original format where standard Qualifying set the Sprint grid. See the Qualifying tab.",
                    variant="info",
                )
            else:
                render_empty_state(
                    title="SHOOTOUT TELEMETRY MISSING",
                    message="No Sprint Shootout session telemetry is stored for this Grand Prix.",
                    hint="Download the Sprint Qualifying session in Mission Control.",
                )

    # 2. Sprint Race
    with sprint_sub_tabs[1]:
        if sessions.get("sprint"):
            sprint_df = get_race_results(sessions["sprint"])

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
                hint="Download the Sprint session in Mission Control.",
            )
