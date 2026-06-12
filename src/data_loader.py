"""
This module provides functions to load session data from JSON files
organized by season and Grand Prix.
"""

import json
import logging
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
SEASONS_DIR = DATA_DIR / "seasons"


def get_available_seasons() -> List[int]:
    """Get list of seasons with data available."""
    if not SEASONS_DIR.exists():
        return []
    
    seasons = []
    for folder in SEASONS_DIR.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            seasons.append(int(folder.name))
    
    return sorted(seasons, reverse=True)


@st.cache_data
def get_season_schedule(year: int) -> List[Dict]:
    """Load season schedule from JSON."""
    schedule_path = SEASONS_DIR / str(year) / "schedule.json"
    
    if schedule_path.exists():
        with open(schedule_path, 'r') as f:
            return json.load(f)
    
    return []


def get_available_gps(year: int) -> List[Dict]:
    """
    Get list of GPs with data for a season.
    
    Returns:
        List of dicts with round, name, folder, and available sessions
    """
    season_path = SEASONS_DIR / str(year)
    
    if not season_path.exists():
        return []
    
    gps = []
    for folder in sorted(season_path.iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            # Parse folder name (e.g., "01_Bahrain_GP")
            parts = folder.name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                round_num = int(parts[0])
                gp_name = parts[1].replace("_", " ")
                
                # Check which sessions are available
                sessions = get_available_sessions(year, folder.name)
                
                gps.append({
                    "round": round_num,
                    "name": gp_name,
                    "folder": folder.name,
                    "sessions": sessions
                })
    
    return sorted(gps, key=lambda x: x["round"])


def get_available_sessions(year: int, gp_folder: str) -> Dict[str, bool]:
    """Check which session JSON files exist for a GP."""
    gp_path = SEASONS_DIR / str(year) / gp_folder
    
    session_files = {
        "fp1": "fp1.json",
        "fp2": "fp2.json",
        "fp3": "fp3.json",
        "qualifying": "qualifying.json",
        "sprint_qualifying": "sprint_qualifying.json",
        "sprint_shootout": "sprint_shootout.json",
        "sprint": "sprint.json",
        "race": "race.json"
    }
    
    available = {}
    for session_name, filename in session_files.items():
        available[session_name] = (gp_path / filename).exists()
    
    return available


def load_session(year: int, gp_folder: str, session_type: str) -> Optional[Dict]:
    """
    Load a single session from JSON file.
    
    Args:
        year: Season year
        gp_folder: GP folder name (e.g., "01_Bahrain_GP")
        session_type: One of fp1, fp2, fp3, qualifying, sprint, race
    
    Returns:
        Session data dictionary or None
    """
    gp_path = SEASONS_DIR / str(year) / gp_folder
    session_path = gp_path / f"{session_type}.json"
    
    if session_path.exists():
        with open(session_path, 'r') as f:
            return json.load(f)
    
    return None


@st.cache_data
def load_gp_data(year: int, gp_folder: str) -> Dict[str, Any]:
    """
    Load all available session data for a GP.
    
    Returns:
        Dict with session_type: session_data for all available sessions
    """
    sessions = get_available_sessions(year, gp_folder)
    
    gp_data = {
        "year": year,
        "gp_folder": gp_folder,
        "sessions": {}
    }
    
    for session_type, is_available in sessions.items():
        if is_available:
            gp_data["sessions"][session_type] = load_session(year, gp_folder, session_type)
    
    # Load metadata if available
    metadata_path = SEASONS_DIR / str(year) / gp_folder / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            gp_data["metadata"] = json.load(f)
    
    return gp_data


def get_driver_best_times(session_data: Dict) -> Dict[str, float]:
    """Extract best lap times per driver from session data."""
    if not session_data or "best_times" not in session_data:
        return {}
    return session_data.get("best_times", {})


def get_qualifying_results(session_data: Dict) -> pd.DataFrame:
    """Convert qualifying session to DataFrame with Q1/Q2/Q3 times."""
    if not session_data or "results" not in session_data:
        return pd.DataFrame()
    
    return pd.DataFrame(session_data["results"])


def get_race_results(session_data: Dict) -> pd.DataFrame:
    """Convert race session to DataFrame with results."""
    if not session_data or "results" not in session_data:
        return pd.DataFrame()
    
    return pd.DataFrame(session_data["results"])


def get_session_laps(session_data: Dict) -> pd.DataFrame:
    """Convert session laps to DataFrame."""
    if not session_data or "laps" not in session_data:
        return pd.DataFrame()
    
    return pd.DataFrame(session_data["laps"])


def aggregate_practice_pace(gp_data: Dict) -> pd.DataFrame:
    """
    Aggregate pace data from all practice sessions.
    
    Returns DataFrame with driver best times from FP1, FP2, FP3.
    """
    practice_sessions = ["fp1", "fp2", "fp3"]
    pace_data = {}
    
    for session_type in practice_sessions:
        if session_type in gp_data.get("sessions", {}):
            session = gp_data["sessions"][session_type]
            best_times = get_driver_best_times(session)
            
            for driver, time in best_times.items():
                if driver not in pace_data:
                    pace_data[driver] = {}
                pace_data[driver][session_type] = time
    
    # Convert to DataFrame
    if pace_data:
        df = pd.DataFrame.from_dict(pace_data, orient="index")
        df.index.name = "driver"
        
        # Get available practice columns
        available_cols = [c for c in practice_sessions if c in df.columns]
        
        # Calculate overall best and average (only from available sessions)
        if available_cols:
            df["best"] = df[available_cols].min(axis=1)
            df["avg"] = df[available_cols].mean(axis=1)
        else:
            # No valid columns, return empty
            return pd.DataFrame()
        
        return df.reset_index()
    
    return pd.DataFrame()


def get_historical_gp_data(gp_name: str, years: List[int] = None) -> List[Dict]:
    """
    Load historical data for a specific GP across multiple years.
    Useful for predictions based on track-specific history.
    
    Args:
        gp_name: Name to match (partial match supported)
        years: List of years to search (default: all available)
    
    Returns:
        List of GP data dicts from matching events
    """
    if years is None:
        years = get_available_seasons()
    
    historical_data = []
    
    for year in years:
        gps = get_available_gps(year)
        for gp in gps:
            if gp_name.lower() in gp["name"].lower():
                gp_data = load_gp_data(year, gp["folder"])
                gp_data["match_year"] = year
                gp_data["match_name"] = gp["name"]
                historical_data.append(gp_data)
    
    return historical_data


@st.cache_data
def load_static_data() -> Dict[str, Any]:
    """Load static data files (only tracks.json remains)."""
    static_data = {}
    
    # Only tracks.json is needed - driver/team info comes from FastF1
    tracks_path = DATA_DIR / "tracks.json"
    if tracks_path.exists():
        with open(tracks_path, 'r') as f:
            static_data["tracks"] = json.load(f)
    
    return static_data


def get_drivers_from_session(session_data: Dict) -> pd.DataFrame:
    """
    Extract driver information from session results.
    This replaces the need for static drivers.json.
    """
    if not session_data or "results" not in session_data:
        return pd.DataFrame()
    
    results = pd.DataFrame(session_data["results"])
    
    if results.empty:
        return pd.DataFrame()
    
    # Extract driver columns
    driver_cols = ["driver", "driver_number", "team"]
    available_cols = [c for c in driver_cols if c in results.columns]
    
    return results[available_cols].drop_duplicates()


def get_teams_from_session(session_data: Dict) -> pd.DataFrame:
    """
    Extract team information from session results.
    This replaces the need for static teams.json.
    """
    if not session_data or "results" not in session_data:
        return pd.DataFrame()
    
    results = pd.DataFrame(session_data["results"])
    
    if results.empty or "team" not in results.columns:
        return pd.DataFrame()
    
    return results[["team"]].drop_duplicates()


# Backward compatibility functions
def prepare_features_from_gp(gp_data: Dict, target_session: str = "race") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features and target for ML model from GP data.
    
    Uses practice and qualifying data to predict race results.
    """
    # Get practice pace
    practice_pace = aggregate_practice_pace(gp_data)
    
    # Get qualifying results
    quali_data = None
    if "qualifying" in gp_data.get("sessions", {}):
        quali_data = get_qualifying_results(gp_data["sessions"]["qualifying"])
    
    # Get target (race results)
    target = None
    if target_session in gp_data.get("sessions", {}):
        target = get_race_results(gp_data["sessions"][target_session])
    
    if practice_pace.empty or quali_data is None or quali_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Merge features - check which columns exist
    quali_cols = ["driver"]
    if "position" in quali_data.columns:
        quali_cols.append("position")
    for qcol in ["q1", "q2", "q3"]:
        if qcol in quali_data.columns:
            quali_cols.append(qcol)
    
    rename_map = {"position": "quali_pos"} if "position" in quali_cols else {}
    
    features = practice_pace.merge(
        quali_data[quali_cols].rename(columns=rename_map),
        on="driver",
        how="inner"
    )
    
    # Prepare target if race data available
    if target is not None and not target.empty:
        features = features.merge(
            target[["driver", "position"]].rename(columns={"position": "race_pos"}),
            on="driver",
            how="inner"
        )
    
    return features, target