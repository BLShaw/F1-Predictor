import fastf1
import pandas as pd
import requests
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
from .config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(CACHE_DIR)

def load_session_data(year: int = 2024, race_name: Optional[str] = None, race_number: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load F1 session data and process lap times"""
    if race_name:
        session = fastf1.get_session(year, race_name, "R")
    elif race_number:
        session = fastf1.get_session(year, race_number, "R")
    else:
        # Default to Monaco if no race is specified
        session = fastf1.get_session(year, "Monaco", "R")
    
    session.load()
    
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    laps.dropna(inplace=True)
    
    # Convert times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
        
    # Get race results for ground truth
    results = session.results[["Abbreviation", "Position"]].copy()
    results.rename(columns={"Abbreviation": "Driver"}, inplace=True)
    
    return laps, results

def aggregate_sector_times(laps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sector times by driver"""
    sector_times = laps.groupby("Driver").agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }).reset_index()
    
    sector_times["TotalSectorTime (s)"] = (
        sector_times["Sector1Time (s)"] +
        sector_times["Sector2Time (s)"] +
        sector_times["Sector3Time (s)"]
    )
    
    return sector_times

def get_weather_data(track_name: str = "Monaco", target_date: Optional[str] = None, api_key: Optional[str] = None) -> Tuple[float, float]:
    """Fetch weather data for a specific track"""
    if api_key is None:
        api_key = WEATHER_API_KEY
    
    # Check if API key is valid (not default placeholder)
    if not api_key or api_key == "YOURAPIKEY":
        logger.warning(f"No valid API key provided for {track_name}, using historical averages.")
        return 0.0, 20.0
    
    if track_name not in TRACK_COORDINATES:
        logger.warning(f"Track {track_name} not found in coordinates, defaulting to Monaco.")
        track_name = "Monaco"
        
    lat = TRACK_COORDINATES[track_name]["lat"]
    lon = TRACK_COORDINATES[track_name]["lon"]
    
    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(weather_url, timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Weather API returned status {response.status_code}: {response.text}")
            return 0.0, 20.0
            
        weather_data = response.json()
        
        # Find forecast closest to target date/time
        # If no target_date, use default FORECAST_TIME
        target = target_date if target_date else FORECAST_TIME
        
        # Simple match: look for the date string in the forecast timestamp
        # In a real app, we'd use datetime objects for closer matching
        forecast_data = next((f for f in weather_data["list"] if target.split(' ')[0] in f["dt_txt"]), None)
        
        # Fallback: if specific date not found (e.g. 2025 date vs current 5-day forecast), 
        # just take the first item as a "current conditions" proxy if we are doing a live check,
        # OR return defaults if we really wanted that specific future date.
        # For this simulator, if we can't find the future date, let's default to benign conditions 
        # so the user can use the sliders to override.
        
        if forecast_data:
            rain_probability = float(forecast_data.get("pop", 0))
            temperature = float(forecast_data["main"]["temp"])
            return rain_probability, temperature
        else:
            logger.warning(f"No forecast found for target {target}, returning defaults.")
            return 0.0, 20.0
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API connection failed: {e}")
        return 0.0, 20.0
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Weather data parsing error: {e}")
        return 0.0, 20.0
    except Exception as e:
        logger.error(f"Unexpected error fetching weather: {e}")
        return 0.0, 20.0

def prepare_qualifying_data(
    track_name: str = "Monaco", 
    qualifying_override: Optional[Dict[str, List[Any]]] = None, 
    race_pace_override: Optional[Dict[str, float]] = None, 
    weather_override: Optional[Tuple[float, float]] = None,
    target_date: Optional[str] = None,
    tyre_strategy: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, float, float]:
    """Prepare qualifying data with team performance scores, adjusted for track"""
    # Use override if provided, else copy default to avoid mutating original
    if qualifying_override:
        qualifying_2025 = pd.DataFrame(qualifying_override)
    else:
        qualifying_2025 = pd.DataFrame(QUALIFYING_2025_DATA)
        
    pace_data = race_pace_override if race_pace_override else CLEAN_AIR_RACE_PACE
    qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(pace_data)
    
    # Apply Tyre Strategy Adjustments
    if tyre_strategy:
        # Heuristic: Pace adjustments relative to Medium
        # Soft: Faster initial pace (-0.5s)
        # Medium: Baseline (0.0s)
        # Hard: Slower initial pace (+0.5s)
        tyre_pace_deltas = {
            "Soft": -0.5,
            "Medium": 0.0,
            "Hard": 0.5
        }
        
        for driver, tyre in tyre_strategy.items():
            if driver in qualifying_2025["Driver"].values:
                delta = tyre_pace_deltas.get(tyre, 0.0)
                # Apply delta to CleanAirRacePace
                mask = qualifying_2025["Driver"] == driver
                qualifying_2025.loc[mask, "CleanAirRacePace (s)"] += delta
                logger.info(f"Applied {tyre} tyre strategy for {driver}: {delta:+.1f}s pace adjustment")

    max_points = max(TEAM_POINTS.values())
    team_performance_score = {team: points / max_points for team, points in TEAM_POINTS.items()}
    
    qualifying_2025["Team"] = qualifying_2025["Driver"].map(DRIVER_TO_TEAM)
    qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)
    
    # Use track-specific position change if available, otherwise use default
    if track_name in TRACK_SPECIFIC_POSITION_CHANGE:
        position_changes = TRACK_SPECIFIC_POSITION_CHANGE[track_name]
    else:
        position_changes = AVERAGE_POSITION_CHANGE
    
    qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(position_changes)
    
    track_stats = TRACK_CHARACTERISTICS.get(track_name, {"Downforce": 3, "PitLoss": 22.0})
    qualifying_2025["TrackDownforce"] = track_stats["Downforce"]
    qualifying_2025["PitLossTime"] = track_stats["PitLoss"]
    
    if weather_override:
        rain_probability, temperature = weather_override
    else:
        rain_probability, temperature = get_weather_data(track_name, target_date=target_date)
    
    # Wet Weather Logic
    # 15% pace penalty if rain probability is high (> 50%)
    WET_PACE_PENALTY = 1.15
    
    if rain_probability > 0.50:
        logger.info(f"Wet weather detected ({rain_probability:.0%}), applying {WET_PACE_PENALTY}x pace penalty.")
        qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["CleanAirRacePace (s)"] * WET_PACE_PENALTY
        
        # Wet Qualifying: Times are slower
        # Simple heuristic: add 10% to qualifying times
        qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * 1.10
    else:
        qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]
    
    return qualifying_2025, rain_probability, temperature