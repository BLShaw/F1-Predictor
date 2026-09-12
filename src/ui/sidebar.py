"""Mission Control sidebar module for F1 Predictor."""

import logging
from datetime import datetime
from typing import Any

import fastf1
import streamlit as st

from src.data_fetcher import fetch_gp, fetch_session, save_session_json
from src.ui.state import get_cached_gps, get_cached_seasons
from src.utils.paths import clear_fastf1_cache

logger = logging.getLogger(__name__)


def render_sidebar() -> tuple[int | None, dict[str, Any] | None]:
    """Render the Mission Control sidebar.

    Returns:
        tuple of (selected_season: int | None, selected_gp: dict | None)
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="sidebar-title">MISSION CONTROL</div>
                <div class="sidebar-caption">Telemetry Management & Control</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        with st.expander("DATA INGESTION ENGINE", expanded=False):
            st.markdown(
                """
                <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.75rem;">
                    Download telemetry sessions directly from FastF1 into the local repository database.
                </div>
                """,
                unsafe_allow_html=True,
            )

            download_mode = st.radio(
                "Ingestion Scope",
                options=["Full Season", "Single GP", "Single Session"],
                horizontal=True,
                label_visibility="collapsed",
            )

            current_year = datetime.now().year

            if download_mode == "Full Season":
                fetch_year = st.number_input(
                    "Season Year",
                    min_value=2018,
                    max_value=current_year + 1,
                    value=current_year,
                    step=1,
                    help="Select the F1 season year to download",
                )

                season_sessions: list[str] = []
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("FP1", value=True, key="s_fp1"):
                        season_sessions.append("FP1")
                    if st.checkbox("FP2", value=True, key="s_fp2"):
                        season_sessions.append("FP2")
                    if st.checkbox("FP3", value=True, key="s_fp3"):
                        season_sessions.append("FP3")
                    if st.checkbox("Qualifying", value=True, key="s_q"):
                        season_sessions.append("Q")
                with col2:
                    if st.checkbox("Sprint Quali", value=False, key="s_sq"):
                        season_sessions.append("SQ")
                    if st.checkbox("Sprint", value=False, key="s_s"):
                        season_sessions.append("S")
                    if st.checkbox("Race", value=True, key="s_r"):
                        season_sessions.append("R")

                if st.button("DOWNLOAD SEASON", width="stretch", type="primary", key="btn_season"):
                    if not season_sessions:
                        st.warning("Select at least one session type.")
                    else:
                        try:
                            schedule = fastf1.get_event_schedule(fetch_year)
                            races = schedule[schedule["EventFormat"] != "testing"]
                            total_gps = len(races)
                            progress_bar = st.progress(0, text="Initializing...")
                            status_text = st.empty()
                            success_count = 0

                            for idx, (_, event) in enumerate(races.iterrows()):
                                round_num = int(event["RoundNumber"])
                                gp_name = str(event["EventName"])
                                progress = (idx + 1) / max(total_gps, 1)
                                progress_bar.progress(progress, text=f"Fetching {gp_name}...")
                                status_text.markdown(f"**Round {round_num}**: {gp_name}")

                                try:
                                    results = fetch_gp(fetch_year, round_num, season_sessions)
                                    success_count += sum(1 for v in results.values() if v)
                                except Exception as exc:
                                    logger.warning("Error downloading %s: %s", gp_name, exc)

                            progress_bar.progress(1.0, text="Complete!")
                            st.success(f"Downloaded {success_count} sessions across {total_gps} Grand Prix events.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to fetch calendar schedule: {exc}")

            elif download_mode == "Single GP":
                gp_year = st.number_input(
                    "Year",
                    min_value=2018,
                    max_value=current_year + 1,
                    value=current_year,
                    step=1,
                    key="gp_year",
                )

                try:
                    schedule = fastf1.get_event_schedule(gp_year)
                    races = schedule[schedule["EventFormat"] != "testing"]
                    gp_names = [f"R{int(row['RoundNumber']):02d} - {row['EventName']}" for _, row in races.iterrows()]
                except Exception:
                    gp_names = []

                if gp_names:
                    selected_gp_name = st.selectbox("Select Grand Prix", options=gp_names)
                    round_num = int(selected_gp_name.split(" - ")[0][1:])
                else:
                    round_num = st.number_input("Round Number", min_value=1, max_value=25, value=1)

                gp_sessions: list[str] = []
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("FP1", value=True, key="gp_fp1"):
                        gp_sessions.append("FP1")
                    if st.checkbox("FP2", value=True, key="gp_fp2"):
                        gp_sessions.append("FP2")
                    if st.checkbox("FP3", value=True, key="gp_fp3"):
                        gp_sessions.append("FP3")
                    if st.checkbox("Qualifying", value=True, key="gp_q"):
                        gp_sessions.append("Q")
                with col2:
                    if st.checkbox("Sprint Quali", value=False, key="gp_sq"):
                        gp_sessions.append("SQ")
                    if st.checkbox("Sprint", value=False, key="gp_s"):
                        gp_sessions.append("S")
                    if st.checkbox("Race", value=True, key="gp_r"):
                        gp_sessions.append("R")

                if st.button("DOWNLOAD GP", width="stretch", type="primary", key="btn_gp"):
                    if not gp_sessions:
                        st.warning("Select at least one session type.")
                    else:
                        with st.spinner("Downloading Grand Prix telemetry..."):
                            try:
                                results = fetch_gp(gp_year, round_num, gp_sessions)
                                success_count = sum(1 for v in results.values() if v)
                                st.success(f"Downloaded {success_count}/{len(gp_sessions)} sessions.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Download failed: {exc}")

            elif download_mode == "Single Session":
                col1, col2 = st.columns(2)
                with col1:
                    session_year = st.number_input(
                        "Year",
                        min_value=2018,
                        max_value=current_year + 1,
                        value=current_year,
                        step=1,
                        key="s_year",
                    )
                with col2:
                    session_type = st.selectbox(
                        "Session",
                        options=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"],
                        format_func=lambda x: {
                            "FP1": "Practice 1",
                            "FP2": "Practice 2",
                            "FP3": "Practice 3",
                            "Q": "Qualifying",
                            "SQ": "Sprint Shootout",
                            "S": "Sprint Race",
                            "R": "Grand Prix Race",
                        }.get(x, x),
                    )

                session_gp_id = st.text_input("Round # or GP Name", value="1", help="Enter round number or name")

                if st.button("DOWNLOAD SESSION", width="stretch", type="primary", key="btn_session"):
                    if not session_gp_id:
                        st.warning("Enter a valid GP round or name.")
                    else:
                        with st.spinner(f"Fetching {session_type}..."):
                            try:
                                result = fetch_session(session_year, session_gp_id, session_type)
                                if result:
                                    s_data, r_num, name = result
                                    save_session_json(s_data, session_year, r_num, name, session_type)
                                    st.success(f"Successfully downloaded {session_type} for {name}.")
                                    st.rerun()
                                else:
                                    st.error("Session not available from telemetry source.")
                            except Exception as exc:
                                st.error(f"Download failed: {exc}")

        st.markdown("---")

        st.markdown('<div class="sidebar-section-label">Data Selection</div>', unsafe_allow_html=True)

        seasons = get_cached_seasons()
        if not seasons:
            st.info("No local data found. Use Data Ingestion above to fetch F1 telemetry.")
            return None, None

        selected_season = st.selectbox(
            "CHAMPIONSHIP SEASON",
            options=seasons,
            format_func=lambda x: f"{x} Season",
        )

        gps = get_cached_gps(selected_season) if selected_season else []
        if not gps:
            st.warning(f"No Grand Prix data available for {selected_season}.")
            return selected_season, None

        gp_options = {gp["folder"]: gp for gp in gps}
        selected_gp_folder = st.selectbox(
            "GRAND PRIX EVENT",
            options=list(gp_options.keys()),
            format_func=lambda x: f"R{gp_options[x]['round']:02d} | {gp_options[x]['name']}",
        )
        selected_gp = gp_options.get(selected_gp_folder)

        st.markdown("---")

        st.markdown('<div class="sidebar-section-label">Quick Actions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("REFRESH GP", width="stretch", type="primary"):
                if selected_season and selected_gp:
                    with st.spinner("Refreshing Grand Prix data..."):
                        try:
                            results = fetch_gp(selected_season, selected_gp["round"])
                            success_count = sum(1 for v in results.values() if v)
                            st.success(f"{success_count} sessions updated.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Refresh failed: {exc}")
                else:
                    st.warning("Select a Grand Prix first.")

        with col2:
            if st.button("CLEAR CACHE", width="stretch"):
                success, msg = clear_fastf1_cache()
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)

        st.markdown("---")

        if selected_gp:
            st.markdown('<div class="sidebar-section-label">Session Telemetry</div>', unsafe_allow_html=True)

            sessions = selected_gp.get("sessions", {})
            session_display = [
                ("FP1", "FP1", sessions.get("fp1", False)),
                ("FP2", "FP2", sessions.get("fp2", False)),
                ("FP3", "FP3", sessions.get("fp3", False)),
                ("Qualifying", "Q", sessions.get("qualifying", False)),
                ("Shootout", "SQ", sessions.get("sprint_qualifying", False)),
                ("Sprint", "S", sessions.get("sprint", False)),
                ("Race", "R", sessions.get("race", False)),
            ]

            for row_idx in range(0, len(session_display), 2):
                col_left, col_right = st.columns(2)
                row_items = session_display[row_idx : row_idx + 2]
                cols = [col_left, col_right]
                for col, (disp_name, session_code, is_avail) in zip(cols, row_items):
                    with col:
                        if is_avail:
                            st.markdown(
                                f"""
                                <div class="telemetry-badge-available">
                                    <span>{disp_name}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            if st.button(
                                f"Fetch {disp_name}",
                                key=f"fetch_btn_{session_code}_{row_idx}",
                                width="stretch",
                            ):
                                with st.spinner(f"Fetching {disp_name}..."):
                                    try:
                                        res = fetch_session(selected_season, selected_gp["round"], session_code)
                                        if res:
                                            s_data, r_num, gp_name = res
                                            save_session_json(s_data, selected_season, r_num, gp_name, session_code)
                                            st.success(f"{disp_name} downloaded.")
                                            st.rerun()
                                        else:
                                            st.warning(f"{disp_name} not available.")
                                    except Exception as exc:
                                        st.warning(f"Fetch failed: {exc}")


            st.markdown("---")

        st.markdown(
            """
            <div class="sidebar-footer">
                <div class="sidebar-footer-title">F1 PREDICTOR v2.2.0</div>
                <div class="sidebar-footer-sub">Engineered with Random Forest & Monte Carlo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_season, selected_gp
