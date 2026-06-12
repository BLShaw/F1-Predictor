"""
Formula 1 Predictor 
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime

# Import project modules
from src.data_loader import (
    get_available_seasons,
    get_available_gps,
    load_gp_data,
    aggregate_practice_pace,
    get_qualifying_results,
    get_race_results,
    get_drivers_from_session,
    load_static_data
)
from src.data_fetcher import fetch_gp, fetch_session, save_session_json
from src.model import AdvancedRacePredictor, F1MLPredictor

# STYLING & CONFIGURATION

from src.utils.helpers import get_team_color, format_lap_time, format_gap
from src.ui.charts import create_pace_chart, create_qualifying_chart, create_prediction_chart

# Page configuration
st.set_page_config(
    page_title="Formula 1 Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS for F1-grade premium styling
try:
    with open("assets/style.css", "r") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


# Helper and Charting functions extracted to src/utils/helpers.py and src/ui/charts.py


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="f1-header">
        <h1>Formula 1 Predictor</h1>
        <p class="subtitle">Advanced Race Prediction • Monte Carlo Simulation • Real-Time Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR - MISSION CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-family: 'Orbitron', monospace; font-size: 1.2rem; color: #E10600; letter-spacing: 3px;">
                MISSION CONTROL
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # DATA DOWNLOADER SECTION
        # ═══════════════════════════════════════════════════════════════════
        with st.expander("**DATA DOWNLOADER**", expanded=False):
            st.markdown("""
            <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-bottom: 1rem;">
                Fetch F1 data directly from FastF1. Choose to download entire seasons, specific GPs, or individual sessions.
            </div>
            """, unsafe_allow_html=True)
            
            # Download Mode Selection
            download_mode = st.radio(
                "Download Mode",
                options=["Full Season", "Single GP", "Single Session"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown("")
            
            # ─────────────────────────────────────────────────────────────────
            # MODE 1: FULL SEASON DOWNLOAD
            # ─────────────────────────────────────────────────────────────────
            if download_mode == "Full Season":
                st.markdown("##### Download Full Season")
                
                current_year = datetime.now().year
                fetch_year = st.number_input(
                    "Season Year",
                    min_value=2018,
                    max_value=current_year,
                    value=current_year,
                    step=1,
                    help="Select the F1 season year to download (2018-present)"
                )
                
                # Session selection for season
                st.markdown("**Sessions to fetch:**")
                season_sessions = []
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("FP1", value=True, key="season_fp1"): season_sessions.append("FP1")
                    if st.checkbox("FP2", value=True, key="season_fp2"): season_sessions.append("FP2")
                    if st.checkbox("FP3", value=True, key="season_fp3"): season_sessions.append("FP3")
                    if st.checkbox("Qualifying", value=True, key="season_q"): season_sessions.append("Q")
                with col2:
                    if st.checkbox("Sprint Quali", value=False, key="season_sq"): season_sessions.append("SQ")
                    if st.checkbox("Sprint", value=False, key="season_s"): season_sessions.append("S")
                    if st.checkbox("Race", value=True, key="season_r"): season_sessions.append("R")
                
                if st.button("DOWNLOAD SEASON", width="stretch", type="primary", key="btn_season"):
                    if not season_sessions:
                        st.warning("Select at least one session type")
                    else:
                        try:
                            import fastf1
                            schedule = fastf1.get_event_schedule(fetch_year)
                            
                            # Filter out testing events
                            races = schedule[schedule["EventFormat"] != "testing"]
                            total_gps = len(races)
                            
                            progress_bar = st.progress(0, text="Initializing...")
                            status_text = st.empty()
                            
                            success_count = 0
                            fail_count = 0
                            
                            for idx, (_, event) in enumerate(races.iterrows()):
                                round_num = int(event["RoundNumber"])
                                gp_name = event["EventName"]
                                
                                progress = (idx + 1) / total_gps
                                progress_bar.progress(progress, text=f"Fetching {gp_name}...")
                                status_text.markdown(f"**Round {round_num}**: {gp_name}")
                                
                                try:
                                    results = fetch_gp(fetch_year, round_num, season_sessions)
                                    session_success = sum(1 for v in results.values() if v)
                                    success_count += session_success
                                except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                                    fail_count += 1
                                    st.warning(f"{gp_name}: {str(e)[:50]}")
                            
                            progress_bar.progress(1.0, text="Complete!")
                            st.success(f"Downloaded {success_count} sessions from {total_gps} GPs")
                            if fail_count > 0:
                                st.warning(f"{fail_count} GPs had errors")
                            st.rerun()
                            
                        except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                            st.error(f"Failed to fetch schedule: {str(e)}")
            
            # ─────────────────────────────────────────────────────────────────
            # MODE 2: SINGLE GP DOWNLOAD
            # ─────────────────────────────────────────────────────────────────
            elif download_mode == "Single GP":
                st.markdown("##### Download Single Grand Prix")
                
                current_year = datetime.now().year
                gp_year = st.number_input(
                    "Season Year",
                    min_value=2018,
                    max_value=current_year,
                    value=current_year,
                    step=1,
                    key="gp_year"
                )
                
                # Try to load schedule for GP selection
                gp_identifier = None
                try:
                    import fastf1
                    schedule = fastf1.get_event_schedule(gp_year)
                    races = schedule[schedule["EventFormat"] != "testing"]
                    
                    gp_names = [f"R{int(row['RoundNumber']):02d} - {row['EventName']}" 
                               for _, row in races.iterrows()]
                    
                    selected_gp_name = st.selectbox(
                        "Select Grand Prix",
                        options=gp_names,
                        key="gp_select"
                    )
                    
                    # Extract round number
                    gp_identifier = int(selected_gp_name.split(" - ")[0].replace("R", ""))
                    
                except (ImportError, ValueError, ConnectionError, TypeError):
                    # Fallback to manual input
                    gp_input_method = st.radio(
                        "Identify GP by:",
                        options=["Round Number", "GP Name"],
                        horizontal=True,
                        key="gp_input_method"
                    )
                    
                    if gp_input_method == "Round Number":
                        gp_identifier = st.number_input(
                            "Round Number",
                            min_value=1,
                            max_value=24,
                            value=1,
                            key="gp_round"
                        )
                    else:
                        gp_identifier = st.text_input(
                            "GP Name (e.g., Monaco, Silverstone)",
                            value="",
                            key="gp_name_input"
                        )
                
                # Session selection for GP
                st.markdown("**Sessions to fetch:**")
                gp_sessions = []
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("FP1", value=True, key="gp_fp1"): gp_sessions.append("FP1")
                    if st.checkbox("FP2", value=True, key="gp_fp2"): gp_sessions.append("FP2")
                    if st.checkbox("FP3", value=True, key="gp_fp3"): gp_sessions.append("FP3")
                    if st.checkbox("Qualifying", value=True, key="gp_q"): gp_sessions.append("Q")
                with col2:
                    if st.checkbox("Sprint Quali", value=False, key="gp_sq"): gp_sessions.append("SQ")
                    if st.checkbox("Sprint", value=False, key="gp_s"): gp_sessions.append("S")
                    if st.checkbox("Race", value=True, key="gp_r"): gp_sessions.append("R")
                
                if st.button("DOWNLOAD GP", width="stretch", type="primary", key="btn_gp"):
                    if not gp_identifier:
                        st.warning("Please select or enter a GP")
                    elif not gp_sessions:
                        st.warning("Select at least one session type")
                    else:
                        with st.spinner(f"Fetching GP data from FastF1..."):
                            try:
                                results = fetch_gp(gp_year, gp_identifier, gp_sessions)
                                success_count = sum(1 for v in results.values() if v)
                                st.success(f"Downloaded {success_count}/{len(gp_sessions)} sessions")
                                st.rerun()
                            except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                                st.error(f"Error: {str(e)}")
            
            # ─────────────────────────────────────────────────────────────────
            # MODE 3: SINGLE SESSION DOWNLOAD
            # ─────────────────────────────────────────────────────────────────
            else:  # Single Session
                st.markdown("##### Download Single Session")
                
                current_year = datetime.now().year
                session_year = st.number_input(
                    "Season Year",
                    min_value=2018,
                    max_value=current_year,
                    value=current_year,
                    step=1,
                    key="session_year"
                )
                
                # GP identification
                session_gp = st.text_input(
                    "GP Name or Round Number",
                    value="",
                    placeholder="e.g., Monaco, 7, Silverstone",
                    key="session_gp"
                )
                
                # Try to convert to int if it's a number
                try:
                    session_gp_id = int(session_gp)
                except ValueError:
                    session_gp_id = session_gp if session_gp else None
                
                # Session type selection
                session_type = st.selectbox(
                    "Session Type",
                    options=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"],
                    format_func=lambda x: {
                        "FP1": "Practice 1",
                        "FP2": "Practice 2",
                        "FP3": "Practice 3",
                        "Q": "Qualifying",
                        "SQ": "Sprint Qualifying",
                        "S": "Sprint Race",
                        "R": "Race"
                    }.get(x, x),
                    key="session_type"
                )
                
                if st.button("DOWNLOAD SESSION", width="stretch", type="primary", key="btn_session"):
                    if not session_gp_id:
                        st.warning("Please enter a GP name or round number")
                    else:
                        with st.spinner(f"Fetching {session_type} from FastF1..."):
                            try:
                                result = fetch_session(session_year, session_gp_id, session_type)
                                if result:
                                    session_data, round_num, gp_name = result
                                    save_session_json(session_data, session_year, round_num, gp_name, session_type)
                                    st.success(f"Downloaded {session_type} for {gp_name}")
                                    st.rerun()
                                else:
                                    st.error("Session not available")
                            except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # DATA SELECTION SECTION
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem; color: rgba(255,255,255,0.5); 
                    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">
            Data Selection
        </div>
        """, unsafe_allow_html=True)
        
        # Season Selection
        seasons = get_available_seasons()
        
        if not seasons:
            st.info("No local data. Use Data Downloader above to fetch F1 data.")
            selected_season = None
            selected_gp = None
        else:
            selected_season = st.selectbox(
                "SELECT SEASON",
                options=seasons,
                format_func=lambda x: f"{x} Season"
            )
            
            # GP Selection
            gps = get_available_gps(selected_season) if selected_season else []
            
            if not gps:
                st.warning(f"No GP data for {selected_season}")
                selected_gp = None
            else:
                gp_options = {gp["folder"]: gp for gp in gps}
                selected_gp_folder = st.selectbox(
                    "SELECT GRAND PRIX",
                    options=list(gp_options.keys()),
                    format_func=lambda x: f"R{gp_options[x]['round']:02d} • {gp_options[x]['name']}"
                )
                selected_gp = gp_options.get(selected_gp_folder)
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # QUICK ACTIONS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("""
        <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem; color: rgba(255,255,255,0.5); 
                    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">
            Quick Actions
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("REFRESH GP", width="stretch", type="primary"):
                if selected_season and selected_gp:
                    with st.spinner("Refreshing GP data..."):
                        try:
                            results = fetch_gp(selected_season, selected_gp["round"])
                            success_count = sum(1 for v in results.values() if v)
                            st.success(f"{success_count} sessions updated")
                            st.rerun()
                        except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Select a GP first")
        
        with col2:
            if st.button("CLEAR CACHE", width="stretch"):
                cache_dir = Path("f1_cache")
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(exist_ok=True)
                    st.success("Cache cleared")
                else:
                    st.info("Cache is empty")
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # SESSION STATUS
        # ═══════════════════════════════════════════════════════════════════
        if selected_gp:
            st.markdown("""
            <div style="font-family: 'Orbitron', monospace; font-size: 0.75rem; color: rgba(255,255,255,0.5); 
                        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">
                Session Status
            </div>
            """, unsafe_allow_html=True)
            
            sessions = selected_gp.get("sessions", {})
            
            # Create a grid layout for session badges
            session_display = [
                ("FP1", "fp1", sessions.get("fp1", False)),
                ("FP2", "fp2", sessions.get("fp2", False)),
                ("FP3", "fp3", sessions.get("fp3", False)),
                ("QUALI", "Q", sessions.get("qualifying", False)),
                ("SQ", "SQ", sessions.get("sprint_qualifying", False)),
                ("SPRINT", "S", sessions.get("sprint", False)),
                ("RACE", "R", sessions.get("race", False)),
            ]
            
            col1, col2 = st.columns(2)
            for i, (name, session_code, available) in enumerate(session_display):
                with col1 if i % 2 == 0 else col2:
                    if available:
                        st.markdown(f'<span class="status-badge status-available">✓ {name}</span>', 
                                   unsafe_allow_html=True)
                    else:
                        # Add fetch button for missing sessions
                        if st.button(f"{name}", key=f"fetch_{session_code}"):
                            with st.spinner(f"Fetching {name}..."):
                                try:
                                    result = fetch_session(selected_season, selected_gp["round"], session_code)
                                    if result:
                                        session_data, round_num, gp_name = result
                                        save_session_json(session_data, selected_season, round_num, gp_name, session_code)
                                        st.success(f"{name} downloaded")
                                        st.rerun()
                                    else:
                                        st.warning(f"{name} not available")
                                except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                                    st.warning(f"{str(e)[:30]}")
            
            st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: rgba(255,255,255,0.3); font-size: 0.7rem;">
            <div style="font-family: 'Orbitron', monospace;">v2.1.0</div>
            <div style="margin-top: 0.5rem;">Powered by FastF1 & Monte Carlo</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN CONTENT AREA
    # ─────────────────────────────────────────────────────────────────────────
    
    if not selected_season or not selected_gp:
        # Welcome / Getting Started
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-family: 'Orbitron', monospace; font-size: 1.5rem; color: rgba(255,255,255,0.3); 
                        letter-spacing: 5px; margin-bottom: 1rem;">
                AWAITING DATA
            </div>
            <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.5); max-width: 500px; margin: 0 auto;">
                Select a season and Grand Prix from the sidebar to begin analysis.<br><br>
                No data? Use the terminal to fetch race data:
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.code("""
# Fetch entire 2024 season
python -m src.data_fetcher --year 2024 --season

# Or fetch a specific GP
python -m src.data_fetcher --year 2024 --gp Monaco
        """, language="bash")
        
        return
    
    # Load GP Data
    gp_data = load_gp_data(selected_season, selected_gp["folder"])
    sessions = gp_data.get("sessions", {})
    
    # GP Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
        <div style="font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 800; color: white;">
            {selected_gp['name']}
        </div>
        <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.4);">
            Round {selected_gp['round']} • {selected_season}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TABS - Dynamic based on sprint weekend
    # ─────────────────────────────────────────────────────────────────────────
    
    # Detect sprint weekend
    is_sprint_weekend = (
        sessions.get("sprint") is not None or 
        sessions.get("sprint_qualifying") is not None or
        sessions.get("sprint_shootout") is not None
    )
    
    # Build tab list dynamically
    if is_sprint_weekend:
        tab_names = ["Overview", "Practice", "Qualifying", "Sprint", "Race", "Predict"]
    else:
        tab_names = ["Overview", "Practice", "Qualifying", "Race", "Predict"]
    
    tabs = st.tabs(tab_names)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Session Overview")
        
        # Metrics Row
        available_sessions = sum(1 for v in selected_gp.get("sessions", {}).values() if v)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sessions Available", f"{available_sessions}/8", 
                     delta="Complete" if available_sessions >= 5 else "Incomplete")
        
        with col2:
            # Get driver count from any available session
            driver_count = 0
            for session_type in ["qualifying", "race", "fp1"]:
                if session_type in sessions:
                    drivers = get_drivers_from_session(sessions[session_type])
                    if not drivers.empty:
                        driver_count = len(drivers)
                        break
            st.metric("Drivers", driver_count)
        
        with col3:
            # Weather from any session
            weather = {}
            for s in sessions.values():
                if s and "weather" in s and s["weather"]:
                    weather = s["weather"]
                    break
            
            if weather:
                temp = weather.get("track_temp", weather.get("air_temp", "—"))
                st.metric("Track Temp", f"{temp:.1f}°C" if isinstance(temp, (int, float)) else "—")
            else:
                st.metric("Track Temp", "—")
        
        with col4:
            if weather:
                rain = "🌧️ Wet" if weather.get("rainfall", False) else "☀️ Dry"
                st.metric("Conditions", rain)
            else:
                st.metric("Conditions", "—")
        
        st.markdown("---")
        
        # Quick Session Summary
        st.markdown("### Session Summary")
        
        summary_data = []
        for session_type, session in sessions.items():
            if session:
                best_times = session.get("best_times", {})
                fastest_driver = min(best_times, key=best_times.get) if best_times else "—"
                fastest_time = format_lap_time(min(best_times.values())) if best_times else "—"
                
                summary_data.append({
                    "Session": session_type.upper().replace("_", " "),
                    "Status": "Complete",
                    "Fastest": fastest_driver,
                    "Best Lap": fastest_time,
                    "Date": session.get("date", "—")
                })
        
        if summary_data:
            st.dataframe(
                pd.DataFrame(summary_data),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("No session data available. Use INIT DOWNLOAD to fetch data.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: PRACTICE
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Practice Analysis")
        
        # Determine if this is a sprint weekend (has sprint session but no FP2/FP3)
        is_sprint_weekend = (
            sessions.get("sprint") is not None or 
            sessions.get("sprint_qualifying") is not None or
            (not sessions.get("fp2") and not sessions.get("fp3") and sessions.get("fp1"))
        )
        
        # Determine available practice sessions
        available_fp_sessions = []
        if sessions.get("fp1"):
            available_fp_sessions.append(("FP1", "fp1", sessions["fp1"]))
        if not is_sprint_weekend:
            if sessions.get("fp2"):
                available_fp_sessions.append(("FP2", "fp2", sessions["fp2"]))
            if sessions.get("fp3"):
                available_fp_sessions.append(("FP3", "fp3", sessions["fp3"]))
        
        if not available_fp_sessions:
            st.info("No practice data available. Fetch FP1/FP2/FP3 sessions first.")
            
            # Show what's expected
            if is_sprint_weekend:
                st.markdown("""
                <div style="background: rgba(255,184,0,0.1); border: 1px solid rgba(255,184,0,0.3); 
                            border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <div style="font-family: 'Orbitron', monospace; color: #FFB800; font-size: 0.9rem;">
                        Sprint Weekend Detected
                    </div>
                    <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
                        Sprint weekends only have FP1. Use the Data Downloader to fetch FP1.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(100,196,255,0.1); border: 1px solid rgba(100,196,255,0.3); 
                            border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <div style="font-family: 'Orbitron', monospace; color: #64C4FF; font-size: 0.9rem;">
                        Standard Weekend
                    </div>
                    <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
                        Expected sessions: FP1, FP2, FP3. Use the Data Downloader to fetch practice sessions.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show sprint weekend indicator
            if is_sprint_weekend:
                st.markdown("""
                <div style="display: inline-block; background: rgba(255,184,0,0.2); border: 1px solid rgba(255,184,0,0.4); 
                            border-radius: 20px; padding: 0.35rem 0.75rem; margin-bottom: 1rem;">
                    <span style="font-family: 'Orbitron', monospace; color: #FFB800; font-size: 0.75rem; letter-spacing: 1px;">
                        SPRINT WEEKEND
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Create sub-tabs for each practice session
            fp_tab_names = [name for name, _, _ in available_fp_sessions]
            fp_tabs = st.tabs(fp_tab_names)
            
            for fp_tab, (tab_name, session_key, session_data) in zip(fp_tabs, available_fp_sessions):
                with fp_tab:
                    if session_data:
                        # Session info header
                        session_date = session_data.get("date", "—")
                        weather = session_data.get("weather", {})
                        
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            best_times = session_data.get("best_times", {})
                            if best_times:
                                fastest_driver = min(best_times, key=best_times.get)
                                st.metric("Fastest", fastest_driver)
                            else:
                                st.metric("Fastest", "—")
                        
                        with col2:
                            if best_times:
                                fastest_time = format_lap_time(min(best_times.values()))
                                st.metric("Best Lap", fastest_time)
                            else:
                                st.metric("Best Lap", "—")
                        
                        with col3:
                            if weather:
                                temp = weather.get("track_temp", weather.get("air_temp", "—"))
                                st.metric("Track Temp", f"{temp:.1f}°C" if isinstance(temp, (int, float)) else "—")
                            else:
                                st.metric("Track Temp", "—")
                        
                        with col4:
                            st.metric("Date", session_date)
                        
                        st.markdown("---")
                        
                        # Pace Chart for this session
                        best_times = session_data.get("best_times", {})
                        if best_times:
                            # Create DataFrame for this session
                            session_pace_df = pd.DataFrame([
                                {"driver": driver, "best": time}
                                for driver, time in best_times.items()
                            ])
                            
                            if not session_pace_df.empty:
                                # Sort by best time
                                session_pace_df = session_pace_df.sort_values("best").head(20)
                                
                                # Calculate gap to fastest
                                fastest = session_pace_df["best"].min()
                                session_pace_df["gap"] = session_pace_df["best"] - fastest
                                
                                # Color scale: green (fast) to red (slow)
                                max_gap = session_pace_df["gap"].max() if session_pace_df["gap"].max() > 0 else 1
                                colors = [f"rgb({min(255, int(150 + (g/max_gap)*105))}, {max(50, int(200 - (g/max_gap)*150))}, 50)" 
                                          for g in session_pace_df["gap"]]
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Bar(
                                    y=session_pace_df["driver"],
                                    x=session_pace_df["gap"],
                                    orientation='h',
                                    marker=dict(
                                        color=colors,
                                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                                    ),
                                    text=[format_gap(g) for g in session_pace_df["gap"]],
                                    textposition='outside',
                                    textfont=dict(family="Orbitron", size=11, color="white"),
                                    hovertemplate="<b>%{y}</b><br>Gap: +%{x:.3f}s<br>Time: %{customdata}<extra></extra>",
                                    customdata=[format_lap_time(t) for t in session_pace_df["best"]]
                                ))
                                
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="Inter", color="white"),
                                    title=dict(
                                        text=f"{session_key.upper()} PACE ANALYSIS",
                                        font=dict(family="Orbitron", size=16, color="white"),
                                        x=0.5
                                    ),
                                    xaxis=dict(
                                        title="Gap to Fastest (seconds)",
                                        gridcolor="rgba(255,255,255,0.1)",
                                        zerolinecolor="rgba(255,255,255,0.3)"
                                    ),
                                    yaxis=dict(
                                        title="",
                                        autorange="reversed",
                                        tickfont=dict(family="Orbitron", size=11)
                                    ),
                                    height=450,
                                    margin=dict(l=80, r=100, t=60, b=40),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, width="stretch")
                                
                                # Detailed Table
                                st.markdown(f"### {session_key.upper()} Lap Times")
                                
                                display_df = session_pace_df.copy()
                                display_df["position"] = range(1, len(display_df) + 1)
                                display_df["best"] = display_df["best"].apply(format_lap_time)
                                display_df["gap"] = display_df["gap"].apply(lambda x: format_gap(x) if x > 0 else "—")
                                
                                display_df = display_df[["position", "driver", "best", "gap"]]
                                display_df = display_df.rename(columns={
                                    "position": "Pos",
                                    "driver": "Driver",
                                    "best": "Best Lap",
                                    "gap": "Gap"
                                })
                                
                                st.dataframe(display_df, width="stretch", hide_index=True)
                        else:
                            st.info(f"No lap time data available for {session_key.upper()}.")
                    else:
                        st.info(f"{session_key.upper()} data not available.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: QUALIFYING
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Qualifying Results")
        
        if "qualifying" in sessions:
            quali_df = get_qualifying_results(sessions["qualifying"])
            
            if not quali_df.empty:
                # Gap Chart
                fig = create_qualifying_chart(quali_df)
                if fig:
                    st.plotly_chart(fig, width="stretch")
                
                # Results Table
                st.markdown("### Grid Positions")
                
                display_df = quali_df.copy()
                
                # Format times
                for col in ["q1", "q2", "q3"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(format_lap_time)
                
                display_df = display_df.rename(columns={
                    "position": "Pos",
                    "driver": "Driver",
                    "team": "Team",
                    "q1": "Q1",
                    "q2": "Q2",
                    "q3": "Q3"
                })
                
                cols_to_show = ["Pos", "Driver", "Team", "Q1", "Q2", "Q3"]
                cols_to_show = [c for c in cols_to_show if c in display_df.columns]
                
                st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)
            else:
                st.info("Qualifying results not available.")
        else:
            st.info("No qualifying data available. Fetch the Qualifying session first.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: SPRINT (Only for Sprint Weekends)
    # ═══════════════════════════════════════════════════════════════════════
    if is_sprint_weekend:
        with tabs[3]:  # Sprint tab
            st.markdown("### Sprint Weekend")
            
            # Sprint Weekend Badge
            st.markdown("""
            <div style="display: inline-block; background: rgba(255,184,0,0.2); border: 1px solid rgba(255,184,0,0.4); 
                        border-radius: 20px; padding: 0.5rem 1rem; margin-bottom: 1.5rem;">
                <span style="font-family: 'Orbitron', monospace; color: #FFB800; font-size: 0.85rem; letter-spacing: 2px;">
                    SPRINT FORMAT
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Sub-tabs for Sprint Qualifying/Shootout and Sprint Race
            sprint_sub_tabs = st.tabs(["Sprint Qualifying", "Sprint Race"])
            
            # ─────────────────────────────────────────────────────────────────
            # SPRINT QUALIFYING / SHOOTOUT
            # ─────────────────────────────────────────────────────────────────
            with sprint_sub_tabs[0]:
                st.markdown("### Sprint Qualifying")
                
                # Check for sprint qualifying or sprint shootout
                sprint_quali_data = sessions.get("sprint_qualifying") or sessions.get("sprint_shootout")
                
                if sprint_quali_data:
                    # Detect format: 2023+ has qualifying results (q1/q2/q3), 2021-2022 uses regular Qualifying
                    session_type = sprint_quali_data.get("session_type", "")
                    has_results = "results" in sprint_quali_data and sprint_quali_data["results"]
                    has_laps_only = ("laps" in sprint_quali_data or "best_times" in sprint_quali_data) and not has_results
                    
                    sq_results = get_qualifying_results(sprint_quali_data)
                    
                    # Check if we got actual qualifying times (SQ1/SQ2/SQ3 or Q1/Q2/Q3)
                    has_quali_times = not sq_results.empty and any(
                        col in sq_results.columns for col in ["q1", "q2", "q3", "sq1", "sq2", "sq3"]
                    )
                    
                    if has_quali_times:
                        # Modern format (2023+): Sprint Shootout with SQ1/SQ2/SQ3
                        session_date = sprint_quali_data.get("date", "—")
                        weather = sprint_quali_data.get("weather", {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Date", session_date)
                        with col2:
                            if weather:
                                temp = weather.get("track_temp", weather.get("air_temp", "—"))
                                st.metric("Track Temp", f"{temp:.1f}°C" if isinstance(temp, (int, float)) else "—")
                            else:
                                st.metric("Track Temp", "—")
                        with col3:
                            if weather:
                                rain = "🌧️ Wet" if weather.get("rainfall", False) else "☀️ Dry"
                                st.metric("Conditions", rain)
                            else:
                                st.metric("Conditions", "—")
                        
                        st.markdown("---")
                        
                        # Sprint Qualifying results table
                        st.markdown("### Sprint Grid")
                        
                        display_df = sq_results.copy()
                        
                        # Format times if available
                        for col in ["q1", "q2", "q3", "sq1", "sq2", "sq3"]:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(format_lap_time)
                        
                        display_df = display_df.rename(columns={
                            "position": "Pos",
                            "driver": "Driver",
                            "team": "Team",
                            "q1": "SQ1",
                            "q2": "SQ2",
                            "q3": "SQ3",
                            "sq1": "SQ1",
                            "sq2": "SQ2",
                            "sq3": "SQ3"
                        })
                        
                        cols_to_show = ["Pos", "Driver", "Team", "SQ1", "SQ2", "SQ3"]
                        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
                        
                        st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)
                    
                    elif has_laps_only:
                        # Legacy format (2021-2022): Sprint Qualifying was determined by regular Qualifying
                        st.markdown("""
                        <div style="background: rgba(100,196,255,0.1); border: 1px solid rgba(100,196,255,0.3); 
                                    border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                            <div style="font-family: 'Orbitron', monospace; color: #64C4FF; font-size: 0.9rem;">
                                2021-2022 Sprint Format Detected
                            </div>
                            <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
                                In the original sprint format, there was no separate Sprint Qualifying/Shootout session.
                                <br><br>
                                <strong>How it worked:</strong>
                                <ul style="margin: 0.5rem 0 0 1rem; color: rgba(255,255,255,0.5);">
                                    <li>Friday: FP1 → Qualifying (set Sprint grid)</li>
                                    <li>Saturday: FP2 → Sprint Race (set Race grid)</li>
                                    <li>Sunday: Race</li>
                                </ul>
                                <br>
                                Check the <strong>Qualifying</strong> tab for the Sprint grid positions.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show best lap times from the "sprint qualifying" session (which is lap data)
                        best_times = sprint_quali_data.get("best_times", {})
                        if best_times:
                            st.markdown("### Session Best Laps")
                            
                            times_df = pd.DataFrame([
                                {"Driver": driver, "Best Lap": format_lap_time(time)}
                                for driver, time in sorted(best_times.items(), key=lambda x: x[1])
                            ])
                            times_df.insert(0, "Pos", range(1, len(times_df) + 1))
                            
                            st.dataframe(times_df, width="stretch", hide_index=True)
                    else:
                        st.info("Sprint Qualifying results not available.")
                else:
                    # Check if this is a legacy format by looking at the year
                    gp_year = gp_data.get("year", 0) or (selected_season if isinstance(selected_season, int) else 0)
                    
                    if gp_year and gp_year <= 2022:
                        st.markdown("""
                        <div style="background: rgba(100,196,255,0.1); border: 1px solid rgba(100,196,255,0.3); 
                                    border-radius: 8px; padding: 1rem;">
                            <div style="font-family: 'Orbitron', monospace; color: #64C4FF; font-size: 0.9rem;">
                                2021-2022 Sprint Format
                            </div>
                            <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
                                This event used the original sprint format where regular <strong>Qualifying</strong> 
                                determined the Sprint grid. Check the <strong>Qualifying</strong> tab.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No Sprint Qualifying data available. Fetch the Sprint Qualifying session first.")
            
            # ─────────────────────────────────────────────────────────────────
            # SPRINT RACE
            # ─────────────────────────────────────────────────────────────────
            with sprint_sub_tabs[1]:
                st.markdown("### Sprint Race Results")
                
                if sessions.get("sprint"):
                    sprint_df = get_race_results(sessions["sprint"])
                    
                    if not sprint_df.empty:
                        # Sprint Podium
                        st.markdown("### Sprint Podium")
                        
                        podium = sprint_df.head(3)
                        cols = st.columns(3)
                        
                        # Sprint points (8-7-6-5-4-3-2-1 for P1-P8)
                        sprint_points = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
                        positions = ["P1", "P2", "P3"]
                        
                        for i, (col, (_, row)) in enumerate(zip(cols, podium.iterrows())):
                            with col:
                                team_color = get_team_color(row.get("team", ""))
                                pos = int(row.get("position", i+1))
                                pts = sprint_points.get(pos, 0)
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, {team_color}33 0%, rgba(30,30,45,0.9) 100%);
                                            border: 1px solid {team_color}66; border-radius: 12px; padding: 1.5rem;
                                            text-align: center;">
                                    <div style="font-size: 1.5rem;">{positions[i]}</div>
                                    <div style="font-family: 'Orbitron', monospace; font-size: 1.25rem; font-weight: 700; 
                                                color: white; margin: 0.5rem 0;">{row.get('driver', '—')}</div>
                                    <div style="color: {team_color}; font-size: 0.85rem;">{row.get('team', '—')}</div>
                                    <div style="font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #FFB800; 
                                                margin-top: 0.5rem;">{pts} PTS</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Full Sprint Results
                        st.markdown("### Sprint Classification")
                        
                        display_df = sprint_df.copy()
                        display_df["sprint_points"] = display_df["position"].apply(
                            lambda x: sprint_points.get(int(x), 0) if pd.notna(x) else 0
                        )
                        
                        display_df = display_df.rename(columns={
                            "position": "Pos",
                            "driver": "Driver",
                            "team": "Team",
                            "grid_position": "Grid",
                            "status": "Status",
                            "sprint_points": "Points"
                        })
                        
                        cols_to_show = ["Pos", "Driver", "Team", "Grid", "Status", "Points"]
                        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
                        
                        st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)
                    else:
                        st.info("Sprint Race results not available.")
                else:
                    st.info("No Sprint Race data available. Fetch the Sprint session first.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: RACE (Index varies based on sprint weekend)
    # ═══════════════════════════════════════════════════════════════════════
    race_tab_index = 4 if is_sprint_weekend else 3
    with tabs[race_tab_index]:
        st.markdown("### Race Results")
        
        if "race" in sessions:
            race_df = get_race_results(sessions["race"])
            
            if not race_df.empty:
                # Podium Display
                st.markdown("### Podium")
                
                podium = race_df.head(3)
                cols = st.columns(3)
                
                positions = ["P1", "P2", "P3"]
                
                for i, (col, (_, row)) in enumerate(zip(cols, podium.iterrows())):
                    with col:
                        team_color = get_team_color(row.get("team", ""))
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {team_color}33 0%, rgba(30,30,45,0.9) 100%);
                                    border: 1px solid {team_color}66; border-radius: 12px; padding: 1.5rem;
                                    text-align: center;">
                            <div style="font-size: 1.5rem;">{positions[i]}</div>
                            <div style="font-family: 'Orbitron', monospace; font-size: 1.25rem; font-weight: 700; 
                                        color: white; margin: 0.5rem 0;">{row.get('driver', '—')}</div>
                            <div style="color: {team_color}; font-size: 0.85rem;">{row.get('team', '—')}</div>
                            <div style="font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #FFB800; 
                                        margin-top: 0.5rem;">{int(row.get('points', 0))} PTS</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Full Results
                st.markdown("### Classification")
                
                display_df = race_df.rename(columns={
                    "position": "Pos",
                    "driver": "Driver",
                    "team": "Team",
                    "grid_position": "Grid",
                    "status": "Status",
                    "points": "Points"
                })
                
                cols_to_show = ["Pos", "Driver", "Team", "Grid", "Status", "Points"]
                cols_to_show = [c for c in cols_to_show if c in display_df.columns]
                
                st.dataframe(display_df[cols_to_show], width="stretch", hide_index=True)
            else:
                st.info("Race results not available.")
        else:
            st.info("No race data available. Fetch the Race session first.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB: PREDICT - ML-POWERED PREDICTIONS WITH SHAP
    # ═══════════════════════════════════════════════════════════════════════
    predict_tab_index = 5 if is_sprint_weekend else 4
    with tabs[predict_tab_index]:
        st.markdown("### 🤖 ML-Powered Race Prediction")
        
        # ML Methodology Banner
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(138,43,226,0.2) 0%, rgba(225,6,0,0.2) 100%); 
                    border: 1px solid rgba(138,43,226,0.4); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-family: 'Orbitron', monospace; color: #9B59B6; font-size: 1rem; margin-bottom: 0.75rem;">
                MACHINE LEARNING PIPELINE
            </div>
            <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                    <div>
                        <div style="color: #64C4FF; font-weight: 600;">Feature Engineering</div>
                        <ul style="margin: 0.25rem 0 0 1rem; color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                            <li>Grid Position (normalized)</li>
                            <li>Practice Pace (normalized)</li>
                            <li>Pace Consistency</li>
                            <li>Grid-Pace Delta</li>
                            <li>Position Strength Score</li>
                        </ul>
                    </div>
                    <div>
                        <div style="color: #FFB800; font-weight: 600;">Prediction Model</div>
                        <ul style="margin: 0.25rem 0 0 1rem; color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                            <li>Random Forest Ensemble</li>
                            <li>Monte Carlo Calibration</li>
                        </ul>
                    </div>
                    <div>
                        <div style="color: #E10600; font-weight: 600;">SHAP Explainability</div>
                        <ul style="margin: 0.25rem 0 0 1rem; color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                            <li>Feature Importance</li>
                            <li>Per-Driver Contributions</li>
                            <li>Model Interpretability</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for required data
        pace_df = aggregate_practice_pace(gp_data)
        quali_df = get_qualifying_results(sessions.get("qualifying", {}))
        
        has_pace = not pace_df.empty
        has_quali = not quali_df.empty
        
        # Data availability indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: {'rgba(0,255,0,0.1)' if has_pace else 'rgba(255,0,0,0.1)'}; 
                        border: 1px solid {'rgba(0,255,0,0.3)' if has_pace else 'rgba(255,0,0,0.3)'}; 
                        border-radius: 8px; padding: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem;">{'✅' if has_pace else '❌'}</div>
                <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem; color: rgba(255,255,255,0.7);">Practice Data</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: {'rgba(0,255,0,0.1)' if has_quali else 'rgba(255,0,0,0.1)'}; 
                        border: 1px solid {'rgba(0,255,0,0.3)' if has_quali else 'rgba(255,0,0,0.3)'}; 
                        border-radius: 8px; padding: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem;">{'✅' if has_quali else '❌'}</div>
                <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem; color: rgba(255,255,255,0.7);">Qualifying Data</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            ml_ready = has_quali
            st.markdown(f"""
            <div style="background: {'rgba(138,43,226,0.2)' if ml_ready else 'rgba(100,100,100,0.1)'}; 
                        border: 1px solid {'rgba(138,43,226,0.4)' if ml_ready else 'rgba(100,100,100,0.3)'}; 
                        border-radius: 8px; padding: 0.75rem; text-align: center;">
                <div style="font-size: 1.5rem;">{'🤖' if ml_ready else '⏳'}</div>
                <div style="font-family: 'Orbitron', monospace; font-size: 0.8rem; color: rgba(255,255,255,0.7);">ML Ready</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if has_quali:
            # Simulation Parameters
            with st.expander("Model Configuration", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    n_sims = st.slider("Monte Carlo Simulations", 500, 10000, 2000, 500)
                with col2:
                    dnf_prob = st.slider("DNF Probability (%)", 1, 15, 5) / 100
                with col3:
                    use_ml = st.checkbox("Use ML Model", value=True)
            
            # Generate Prediction Button
            if st.button("RUN ML PREDICTION", type="primary", width="stretch"):
                with st.spinner("Running ML predictions and SHAP analysis..."):
                    try:
                        # Prepare features
                        if has_pace:
                            # Merge pace and quali - handle column names
                            if "position" in quali_df.columns:
                                features = pace_df.merge(
                                    quali_df[["driver", "position"]].rename(columns={"position": "grid"}),
                                    on="driver",
                                    how="inner"
                                )
                            else:
                                # Grid might already be the column name
                                features = pace_df.merge(
                                    quali_df[["driver"]],
                                    on="driver",
                                    how="inner"
                                )
                                features["grid"] = range(1, len(features) + 1)
                        else:
                            # Use quali only
                            if "position" in quali_df.columns:
                                features = quali_df[["driver", "position"]].copy()
                                features = features.rename(columns={"position": "grid"})
                            else:
                                features = quali_df[["driver"]].copy()
                                features["grid"] = range(1, len(features) + 1)
                            features["best"] = features["grid"]  # Fallback pace
                        
                        if features.empty:
                            st.error("Could not merge pace and qualifying data.")
                        else:
                            if use_ml:
                                predictor = F1MLPredictor()
                                results = predictor.predict(features, n_sims=n_sims)
                                
                                # Store full results
                                st.session_state["ml_results"] = results
                                st.session_state["predictions"] = results["predictions"]
                            else:
                                # Use Monte Carlo only
                                predictor = AdvancedRacePredictor()
                                predictor.base_dnf_prob = dnf_prob
                                
                                predictions = predictor.predict(features, n_sims=n_sims)
                                
                                st.session_state["ml_results"] = None
                                st.session_state["predictions"] = predictions
                            
                            st.session_state["prediction_params"] = {
                                "n_sims": n_sims,
                                "dnf_prob": dnf_prob,
                                "use_ml": use_ml
                            }
                            
                            st.success(f"{'XGBoost + SHAP' if use_ml else 'Monte Carlo'} analysis complete!")
                    
                    except (ValueError, ConnectionError, KeyError, TypeError, AttributeError) as e:
                        st.error(f"Prediction failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display Results
            if "predictions" in st.session_state:
                predictions = st.session_state["predictions"]
                ml_results = st.session_state.get("ml_results")
                params = st.session_state.get("prediction_params", {})
                
                st.markdown("---")
                
                # Model Info Header
                model_type = ml_results.get("model_type", "Monte Carlo") if ml_results else "Monte Carlo"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-family: 'Orbitron', monospace; color: rgba(255,255,255,0.6); 
                                font-size: 0.8rem; letter-spacing: 2px;">
                        MODEL: {model_type.upper()} • {params.get('n_sims', 2000):,} SIMULATIONS
                    </div>
                    <div style="background: rgba(138,43,226,0.3); border-radius: 15px; padding: 0.3rem 0.75rem;">
                        <span style="font-family: 'Orbitron', monospace; color: #9B59B6; font-size: 0.75rem;">
                            {'ML + SHAP' if ml_results else 'MONTE CARLO'}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Sub-tabs for different views
                pred_tabs = st.tabs(["Predictions", "Feature Importance", "Model Insights"])
                
                # Tab 1: Predictions
                with pred_tabs[0]:
                    # Win Probability Chart with team colors
                    st.markdown("#### Win Probability Distribution")
                    
                    pred_sorted = predictions.sort_values("Win %", ascending=False).head(15)
                    
                    # Get team info for colors
                    if has_quali and "team" in quali_df.columns:
                        team_map = dict(zip(quali_df["driver"], quali_df["team"]))
                        colors = [get_team_color(team_map.get(d, "")) for d in pred_sorted["Driver"]]
                    else:
                        colors = ["#E10600"] * len(pred_sorted)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=pred_sorted["Driver"],
                        x=pred_sorted["Win %"] * 100,
                        orientation='h',
                        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.4)', width=1)),
                        text=[f"{p*100:.1f}%" for p in pred_sorted["Win %"]],
                        textposition='outside',
                        textfont=dict(family="Orbitron", size=11, color="white"),
                        hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="white"),
                        xaxis=dict(title="Win Probability (%)", gridcolor="rgba(255,255,255,0.1)"),
                        yaxis=dict(title="", autorange="reversed", tickfont=dict(family="Orbitron", size=11)),
                        height=450,
                        margin=dict(l=100, r=80, t=20, b=40),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                    
                    # Full Results Table
                    st.markdown("#### Complete Prediction Results")
                    
                    display_df = predictions.copy()
                    display_df["Win %"] = (display_df["Win %"] * 100).round(1).astype(str) + "%"
                    display_df["Podium %"] = (display_df["Podium %"] * 100).round(1).astype(str) + "%"
                    display_df["Points %"] = (display_df["Points %"] * 100).round(1).astype(str) + "%"
                    display_df["Exp. Points"] = display_df["Exp. Points"].round(2)
                    display_df["Avg Finish"] = display_df["Avg Finish"].round(1)
                    
                    st.dataframe(display_df, width="stretch", hide_index=True)
                    
                    # Download
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        "Download Predictions (CSV)",
                        data=csv,
                        file_name=f"f1_ml_prediction_{selected_gp['folder']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                # Tab 2: Feature Importance (SHAP)
                with pred_tabs[1]:
                    if ml_results and ml_results.get("feature_importance") is not None:
                        st.markdown("#### SHAP Feature Importance")
                        
                        st.markdown("""
                        <div style="background: rgba(138,43,226,0.1); border: 1px solid rgba(138,43,226,0.3); 
                                    border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                            <div style="font-family: 'Inter', sans-serif; color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                                <strong>SHAP (SHapley Additive exPlanations)</strong> values show how each feature 
                                contributes to the model's predictions. Higher values = more impact on predicted position.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        feat_imp = ml_results["feature_importance"]
                        
                        # Feature importance bar chart
                        fig = go.Figure()
                        
                        colors = ["#9B59B6", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
                        
                        fig.add_trace(go.Bar(
                            y=feat_imp["Feature"],
                            x=feat_imp["Importance"],
                            orientation='h',
                            marker=dict(color=colors[:len(feat_imp)], line=dict(color='white', width=1)),
                            text=[f"{v:.3f}" for v in feat_imp["Importance"]],
                            textposition='outside',
                            textfont=dict(family="Orbitron", size=11, color="white"),
                            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", color="white"),
                            title=dict(
                                text="Mean |SHAP| Value (Impact on Predicted Position)",
                                font=dict(family="Orbitron", size=14),
                                x=0.5
                            ),
                            xaxis=dict(title="Mean |SHAP| Value", gridcolor="rgba(255,255,255,0.1)"),
                            yaxis=dict(title="", autorange="reversed"),
                            height=350,
                            margin=dict(l=150, r=60, t=60, b=40)
                        )
                        
                        st.plotly_chart(fig, width="stretch")
                        
                        # Feature descriptions table
                        st.markdown("#### Feature Definitions")
                        
                        feature_desc = pd.DataFrame([
                            {"Feature": "grid_norm", "Description": "Qualifying position normalized (0=pole, 1=last)", "Impact": "Lower is better"},
                            {"Feature": "pace_norm", "Description": "Practice pace normalized (0=fastest, 1=slowest)", "Impact": "Lower is better"},
                            {"Feature": "pace_consistency", "Description": "Variation between average and best lap times", "Impact": "Lower = more consistent"},
                            {"Feature": "grid_pace_delta", "Description": "Difference between grid and pace rankings", "Impact": "Negative = faster than grid suggests"},
                            {"Feature": "position_strength", "Description": "Combined weighted performance score", "Impact": "Lower = stronger performer"}
                        ])
                        
                        st.dataframe(feature_desc, width="stretch", hide_index=True)
                    else:
                        if ml_results and ml_results.get("shap_error"):
                            st.error(f"SHAP calculation failed: {ml_results['shap_error']}")
                        else:
                            st.info("Run prediction with 'Use XGBoost ML Model' enabled to see SHAP analysis.")
                
                # Tab 3: Model Insights
                with pred_tabs[2]:
                    st.markdown("#### Model Architecture & Methodology")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div style="background: rgba(54,113,198,0.1); border: 1px solid rgba(54,113,198,0.3); 
                                    border-radius: 12px; padding: 1rem;">
                            <div style="font-family: 'Orbitron', monospace; color: #3671C6; font-size: 0.9rem; margin-bottom: 0.75rem;">
                                XGBoost Configuration
                            </div>
                            <table style="width: 100%; font-size: 0.85rem; color: rgba(255,255,255,0.7);">
                                <tr><td>Algorithm</td><td style="text-align: right;">XGBRegressor</td></tr>
                                <tr><td>Estimators</td><td style="text-align: right;">100</td></tr>
                                <tr><td>Max Depth</td><td style="text-align: right;">4</td></tr>
                                <tr><td>Learning Rate</td><td style="text-align: right;">0.1</td></tr>
                                <tr><td>Training Data</td><td style="text-align: right;">1000 synthetic races</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="background: rgba(225,6,0,0.1); border: 1px solid rgba(225,6,0,0.3); 
                                    border-radius: 12px; padding: 1rem;">
                            <div style="font-family: 'Orbitron', monospace; color: #E10600; font-size: 0.9rem; margin-bottom: 0.75rem;">
                                Monte Carlo Simulation
                            </div>
                            <table style="width: 100%; font-size: 0.85rem; color: rgba(255,255,255,0.7);">
                                <tr><td>Iterations</td><td style="text-align: right;">{:,}</td></tr>
                                <tr><td>DNF Probability</td><td style="text-align: right;">5%</td></tr>
                                <tr><td>Noise Distribution</td><td style="text-align: right;">Normal(0, 1.5)</td></tr>
                                <tr><td>Position Assignment</td><td style="text-align: right;">Rank-based</td></tr>
                            </table>
                        </div>
                        """.format(params.get('n_sims', 2000)), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Model Pipeline Diagram
                    st.markdown("#### Prediction Pipeline")
                    
                    st.markdown("""
                    <div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; 
                                flex-wrap: wrap; padding: 1rem 0;">
                        <div style="background: rgba(100,196,255,0.2); border: 1px solid rgba(100,196,255,0.4); 
                                    border-radius: 8px; padding: 0.75rem 1rem; text-align: center;">
                            <div style="font-size: 1.25rem;"></div>
                            <div style="font-size: 0.75rem; color: #64C4FF;">Raw Data</div>
                        </div>
                        <div style="color: rgba(255,255,255,0.4);">→</div>
                        <div style="background: rgba(255,184,0,0.2); border: 1px solid rgba(255,184,0,0.4); 
                                    border-radius: 8px; padding: 0.75rem 1rem; text-align: center;">
                            <div style="font-size: 1.25rem;"></div>
                            <div style="font-size: 0.75rem; color: #FFB800;">Feature Eng.</div>
                        </div>
                        <div style="color: rgba(255,255,255,0.4);">→</div>
                        <div style="background: rgba(138,43,226,0.2); border: 1px solid rgba(138,43,226,0.4); 
                                    border-radius: 8px; padding: 0.75rem 1rem; text-align: center;">
                            <div style="font-size: 1.25rem;"></div>
                            <div style="font-size: 0.75rem; color: #9B59B6;">XGBoost</div>
                        </div>
                        <div style="color: rgba(255,255,255,0.4);">→</div>
                        <div style="background: rgba(225,6,0,0.2); border: 1px solid rgba(225,6,0,0.4); 
                                    border-radius: 8px; padding: 0.75rem 1rem; text-align: center;">
                            <div style="font-size: 1.25rem;"></div>
                            <div style="font-size: 0.75rem; color: #E10600;">Monte Carlo</div>
                        </div>
                        <div style="color: rgba(255,255,255,0.4);">→</div>
                        <div style="background: rgba(46,204,113,0.2); border: 1px solid rgba(46,204,113,0.4); 
                                    border-radius: 8px; padding: 0.75rem 1rem; text-align: center;">
                            <div style="font-size: 1.25rem;"></div>
                            <div style="font-size: 0.75rem; color: #2ECC71;">Predictions</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Features used (if available)
                    if ml_results and ml_results.get("features_used") is not None:
                        st.markdown("#### Engineered Features (Current Race)")
                        
                        features_df = ml_results["features_used"]
                        display_feat = features_df.copy()
                        
                        for col in display_feat.columns:
                            if col != "driver" and display_feat[col].dtype in ['float64', 'float32']:
                                display_feat[col] = display_feat[col].round(4)
                        
                        st.dataframe(display_feat, width="stretch", hide_index=True)
        
        else:
            st.warning("Qualifying data is required for predictions. Please fetch the Qualifying session first.")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
