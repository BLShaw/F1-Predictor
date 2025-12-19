import os
import json

# Configuration and constants
CACHE_DIR = "f1_cache"
WEATHER_API_KEY = "YOURAPIKEY"  # Replace with actual API key
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_json_data(filename):
    """Load data from a JSON file in the data directory"""
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found in {DATA_DIR}. Returning empty dict.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: {filename} is not valid JSON. Returning empty dict.")
        return {}

# Load external data
_drivers_data = load_json_data('drivers.json')
_teams_data = load_json_data('teams.json')
_tracks_data = load_json_data('tracks.json')
_season_data = load_json_data('season.json')
_tyres_data = load_json_data('tyres.json')

# Track coordinates for weather data
TRACK_COORDINATES = _tracks_data.get('TRACK_COORDINATES', {})

# Default forecast time (will be updated based on race)
FORECAST_TIME = "2025-05-25 13:00:00"

# Clean air race pace data (general data, can be track-specific if needed)
CLEAN_AIR_RACE_PACE = _drivers_data.get('CLEAN_AIR_RACE_PACE', {})

# Team points data (2024 season final points)
TEAM_POINTS = _teams_data.get('TEAM_POINTS', {})

# Driver to team mapping
DRIVER_TO_TEAM = _drivers_data.get('DRIVER_TO_TEAM', {})

# Average position change at tracks (general data, can be track-specific if needed)
AVERAGE_POSITION_CHANGE = _drivers_data.get('AVERAGE_POSITION_CHANGE', {})

# Track-specific average position change (if available)
TRACK_SPECIFIC_POSITION_CHANGE = _tracks_data.get('TRACK_SPECIFIC_POSITION_CHANGE', {})

# Track characteristics: Downforce (1=Low, 5=High), PitLoss (seconds, estimated)
TRACK_CHARACTERISTICS = _tracks_data.get('TRACK_CHARACTERISTICS', {})

# Tyre degradation factors (seconds lost per lap)
TYRE_DEGRADATION = _tyres_data.get('TYRE_DEGRADATION', {})

# Default 2025 Qualifying data (placeholder - actual data should be loaded based on race)
QUALIFYING_2025_DATA = _season_data.get('QUALIFYING_2025_DATA', {})

# F1 race schedule information
RACE_SCHEDULE = _season_data.get('RACE_SCHEDULE', [])