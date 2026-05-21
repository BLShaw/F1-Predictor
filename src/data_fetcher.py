"""
F1 Data Fetcher - FastF1 Integration Module
"""

import fastf1
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure FastF1 cache
CACHE_DIR = Path("f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Data directory
DATA_DIR = Path("data/seasons")

# Session type mappings
SESSION_TYPES = {
    "FP1": "Practice 1",
    "FP2": "Practice 2", 
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "SQ": "Sprint Qualifying",
    "SS": "Sprint Shootout",
    "S": "Sprint",
    "R": "Race"
}


def get_gp_folder_name(round_num: int, gp_name: str) -> str:
    """Create standardized GP folder name."""
    # Clean GP name for folder
    clean_name = gp_name.replace(" ", "_").replace("Grand_Prix", "GP")
    return f"{round_num:02d}_{clean_name}"


def ensure_gp_folder(year: int, gp_name: str, round_num: int) -> Path:
    """Ensure GP folder exists and return path."""
    folder_name = get_gp_folder_name(round_num, gp_name)
    gp_path = DATA_DIR / str(year) / folder_name
    gp_path.mkdir(parents=True, exist_ok=True)
    return gp_path


def extract_lap_data(laps: pd.DataFrame) -> List[Dict]:
    """Extract lap data from FastF1 laps DataFrame."""
    lap_data = []
    for _, lap in laps.iterrows():
        lap_entry = {
            "driver": lap.get("Driver", ""),
            "lap_number": int(lap.get("LapNumber", 0)),
            "lap_time": lap.get("LapTime").total_seconds() if pd.notna(lap.get("LapTime")) else None,
            "sector1": lap.get("Sector1Time").total_seconds() if pd.notna(lap.get("Sector1Time")) else None,
            "sector2": lap.get("Sector2Time").total_seconds() if pd.notna(lap.get("Sector2Time")) else None,
            "sector3": lap.get("Sector3Time").total_seconds() if pd.notna(lap.get("Sector3Time")) else None,
            "compound": lap.get("Compound", ""),
            "tyre_life": int(lap.get("TyreLife", 0)) if pd.notna(lap.get("TyreLife")) else None,
            "is_personal_best": bool(lap.get("IsPersonalBest", False)),
            "stint": int(lap.get("Stint", 1)) if pd.notna(lap.get("Stint")) else 1,
        }
        lap_data.append(lap_entry)
    return lap_data


def extract_results(results: pd.DataFrame) -> List[Dict]:
    """Extract results from FastF1 results DataFrame."""
    result_data = []
    for _, row in results.iterrows():
        entry = {
            "position": int(row.get("Position", 0)) if pd.notna(row.get("Position")) else None,
            "driver": row.get("Abbreviation", ""),
            "driver_number": int(row.get("DriverNumber", 0)) if pd.notna(row.get("DriverNumber")) else None,
            "team": row.get("TeamName", ""),
            "grid_position": int(row.get("GridPosition", 0)) if pd.notna(row.get("GridPosition")) else None,
            "status": row.get("Status", ""),
            "points": float(row.get("Points", 0)) if pd.notna(row.get("Points")) else 0,
        }
        
        # Add qualifying times if available
        for q in ["Q1", "Q2", "Q3"]:
            if q in row and pd.notna(row.get(q)):
                entry[q.lower()] = row.get(q).total_seconds()
        
        # Add race time if available
        if "Time" in row and pd.notna(row.get("Time")):
            try:
                entry["time"] = row.get("Time").total_seconds()
            except (AttributeError, TypeError):
                entry["time"] = str(row.get("Time"))
        
        result_data.append(entry)
    return result_data


def extract_weather(session) -> Dict:
    """Extract weather data from session."""
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            return {
                "air_temp": float(weather["AirTemp"].mean()) if "AirTemp" in weather else None,
                "track_temp": float(weather["TrackTemp"].mean()) if "TrackTemp" in weather else None,
                "humidity": float(weather["Humidity"].mean()) if "Humidity" in weather else None,
                "rainfall": bool(weather["Rainfall"].any()) if "Rainfall" in weather else False,
                "wind_speed": float(weather["WindSpeed"].mean()) if "WindSpeed" in weather else None,
            }
    except Exception as e:
        logger.warning(f"Could not extract weather data: {e}")
    return {}


def fetch_session(year: int, gp: str, session_type: str) -> Optional[Dict]:
    """
    Fetch a single session from FastF1 and return as dictionary.
    
    Args:
        year: Season year (e.g., 2024)
        gp: GP name or round number
        session_type: One of FP1, FP2, FP3, Q, SQ, SS, S, R
    
    Returns:
        Dictionary with session data or None if unavailable
    """
    try:
        logger.info(f"Fetching {session_type} for {gp} {year}...")
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=False, messages=False)
        
        # Get event info
        event = session.event
        round_num = int(event["RoundNumber"])
        gp_name = event["EventName"]
        
        # Build session data
        session_data = {
            "session_type": session_type,
            "session_name": SESSION_TYPES.get(session_type, session_type),
            "year": year,
            "round": round_num,
            "gp_name": gp_name,
            "date": str(session.date.date()) if session.date else None,
            "track": event.get("Location", ""),
            "country": event.get("Country", ""),
        }
        
        # Extract best times from laps data (individual lap data is not saved as the app only uses best_times)
        if hasattr(session, 'laps') and session.laps is not None and not session.laps.empty:
            
            # Calculate best times per driver
            best_times = {}
            for driver in session.laps["Driver"].unique():
                driver_laps = session.laps[session.laps["Driver"] == driver]
                valid_times = driver_laps["LapTime"].dropna()
                if not valid_times.empty:
                    best_times[driver] = valid_times.min().total_seconds()
            session_data["best_times"] = best_times
        
        # Add results
        if hasattr(session, 'results') and session.results is not None and not session.results.empty:
            session_data["results"] = extract_results(session.results)
        
        # Add weather
        session_data["weather"] = extract_weather(session)
        
        # Add fetch timestamp
        session_data["fetched_at"] = datetime.now().isoformat()
        
        logger.info(f"Successfully fetched {session_type} for {gp_name} {year}")
        return session_data, round_num, gp_name
        
    except Exception as e:
        logger.error(f"Error fetching {session_type} for {gp} {year}: {e}")
        return None


def save_session_json(session_data: Dict, year: int, round_num: int, gp_name: str, session_type: str):
    """Save session data to JSON file."""
    gp_path = ensure_gp_folder(year, gp_name, round_num)
    
    # Map session type to filename
    filename_map = {
        "FP1": "fp1.json",
        "FP2": "fp2.json",
        "FP3": "fp3.json",
        "Q": "qualifying.json",
        "SQ": "sprint_qualifying.json",
        "SS": "sprint_shootout.json",
        "S": "sprint.json",
        "R": "race.json"
    }
    
    filename = filename_map.get(session_type, f"{session_type.lower()}.json")
    filepath = gp_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2, default=str)
    
    logger.info(f"Saved {filepath}")
    
    # Update metadata
    update_metadata(gp_path, session_type)


def update_metadata(gp_path: Path, session_type: str):
    """Update GP metadata file with session status."""
    metadata_path = gp_path / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"sessions": {}}
    
    metadata["sessions"][session_type] = {
        "status": "complete",
        "fetched_at": datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def fetch_gp(year: int, gp: str, sessions: List[str] = None) -> Dict[str, bool]:
    """
    Fetch all sessions for a Grand Prix.
    
    Args:
        year: Season year
        gp: GP name or round number
        sessions: List of session types to fetch (default: all)
    
    Returns:
        Dict of session_type: success status
    """
    if sessions is None:
        sessions = ["FP1", "FP2", "FP3", "Q", "R"]
    
    results = {}
    for session_type in sessions:
        result = fetch_session(year, gp, session_type)
        if result:
            session_data, round_num, gp_name = result
            save_session_json(session_data, year, round_num, gp_name, session_type)
            results[session_type] = True
        else:
            results[session_type] = False
    
    return results


def fetch_season(year: int, sessions: List[str] = None):
    """
    Fetch all GPs for a season.
    
    Args:
        year: Season year
        sessions: List of session types to fetch per GP
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        
        # Save schedule
        schedule_path = DATA_DIR / str(year) / "schedule.json"
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        
        schedule_data = []
        for _, event in schedule.iterrows():
            if event.get("EventFormat", "") not in ["testing"]:
                schedule_data.append({
                    "round": int(event["RoundNumber"]),
                    "name": event["EventName"],
                    "location": event.get("Location", ""),
                    "country": event.get("Country", ""),
                    "date": str(event.get("EventDate", "")),
                    "format": event.get("EventFormat", "conventional")
                })
        
        with open(schedule_path, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        logger.info(f"Saved schedule for {year} ({len(schedule_data)} events)")
        
        # Fetch each GP
        for event in schedule_data:
            if event["round"] > 0:  # Skip testing events
                logger.info(f"\n{'='*50}")
                logger.info(f"Fetching {event['name']} (Round {event['round']})")
                logger.info(f"{'='*50}")
                
                # Determine sessions based on format
                gp_sessions = sessions or ["FP1", "FP2", "FP3", "Q", "R"]
                if event["format"] == "sprint_shootout":
                    gp_sessions = ["FP1", "Q", "SS", "S", "R"]
                elif event["format"] == "sprint":
                    gp_sessions = ["FP1", "SQ", "S", "FP2", "Q", "R"]
                
                fetch_gp(year, event["round"], gp_sessions)
                
    except Exception as e:
        logger.error(f"Error fetching season {year}: {e}")


def get_available_sessions(year: int, gp_folder: str) -> Dict[str, bool]:
    """Check which sessions are available for a GP."""
    gp_path = DATA_DIR / str(year) / gp_folder
    
    sessions = {}
    for session_file in ["fp1.json", "fp2.json", "fp3.json", "qualifying.json", 
                         "sprint_qualifying.json", "sprint_shootout.json", "sprint.json", "race.json"]:
        sessions[session_file.replace(".json", "")] = (gp_path / session_file).exists()
    
    return sessions


def update_latest_session(year: int = None) -> Optional[str]:
    """
    Fetch the most recent completed session.
    Used for real-time updates during race weekends.
    """
    if year is None:
        year = datetime.now().year
    
    try:
        schedule = fastf1.get_event_schedule(year)
        now = datetime.now()
        
        # Find the most recent event
        for _, event in schedule.iterrows():
            event_date = pd.to_datetime(event.get("Session5Date"))  # Race date
            if event_date and event_date < now:
                round_num = int(event["RoundNumber"])
                
                # Try to fetch each session
                for session_type in ["R", "Q", "FP3", "FP2", "FP1"]:
                    result = fetch_session(year, round_num, session_type)
                    if result:
                        session_data, round_num, gp_name = result
                        save_session_json(session_data, year, round_num, gp_name, session_type)
                        return f"Updated {session_type} for {gp_name}"
        
        return "No new sessions available"
        
    except Exception as e:
        logger.error(f"Error updating latest session: {e}")
        return None


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="F1 Data Fetcher")
    parser.add_argument("--year", type=int, help="Season year")
    parser.add_argument("--gp", type=str, help="GP name or round number")
    parser.add_argument("--session", type=str, help="Session type (FP1, FP2, FP3, Q, R, etc.)")
    parser.add_argument("--season", action="store_true", help="Fetch entire season")
    parser.add_argument("--update", action="store_true", help="Update latest session")
    
    args = parser.parse_args()
    
    if args.update:
        result = update_latest_session(args.year)
        print(result)
    elif args.season and args.year:
        fetch_season(args.year)
    elif args.year and args.gp:
        if args.session:
            result = fetch_session(args.year, args.gp, args.session)
            if result:
                session_data, round_num, gp_name = result
                save_session_json(session_data, args.year, round_num, gp_name, args.session)
        else:
            fetch_gp(args.year, args.gp)
    else:
        parser.print_help()
