import fastf1
import pandas as pd
import requests
from datetime import datetime
from .config import *

fastf1.Cache.enable_cache(CACHE_DIR)

def load_session_data(year=2024, race_name=None, race_number=None):
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
    
    return laps

def aggregate_sector_times(laps):
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

def get_weather_data(track_name="Monaco", api_key=None):
    """Fetch weather data for a specific track"""
    if api_key is None:
        api_key = WEATHER_API_KEY
    
    if track_name not in TRACK_COORDINATES:
        # Default to Monaco if track not found
        track_name = "Monaco"
        
    lat = TRACK_COORDINATES[track_name]["lat"]
    lon = TRACK_COORDINATES[track_name]["lon"]
    
    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(weather_url)
        weather_data = response.json()
        forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == FORECAST_TIME), None)
        
        rain_probability = forecast_data["pop"] if forecast_data else 0
        temperature = forecast_data["main"]["temp"] if forecast_data else 20
    except:
        rain_probability = 0
        temperature = 20
    
    return rain_probability, temperature

def prepare_qualifying_data(track_name="Monaco"):
    """Prepare qualifying data with team performance scores, adjusted for track"""
    qualifying_2025 = pd.DataFrame(QUALIFYING_2025_DATA)
    qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(CLEAN_AIR_RACE_PACE)
    
    # Add team data
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
    
    # Handle weather adjustments
    rain_probability, temperature = get_weather_data(track_name)
    
    if rain_probability >= 0.75:
        # Note: WetPerformanceFactor would need to be defined
        qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]
    else:
        qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]
    
    return qualifying_2025, rain_probability, temperature