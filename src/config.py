"""
F1 Configuration Module
"""

import os
import json
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SEASONS_DIR = DATA_DIR / "seasons"
CACHE_DIR = BASE_DIR / "f1_cache"

# Weather API
WEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "YOURAPIKEY")


def load_json_data(filename: str) -> dict:
    """Load data from a JSON file in the data directory."""
    file_path = DATA_DIR / filename
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


# Load static data
_tracks_data = load_json_data('tracks.json')

# Track data (required for weather API and track characteristics)
TRACK_COORDINATES = _tracks_data.get('TRACK_COORDINATES', {})
TRACK_CHARACTERISTICS = _tracks_data.get('TRACK_CHARACTERISTICS', {})

# F1 Team Colors (2024 Season)
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "Ferrari": "#E8002D",
    "Scuderia Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "Mercedes-AMG Petronas F1 Team": "#27F4D2",
    "McLaren": "#FF8000",
    "McLaren F1 Team": "#FF8000",
    "Aston Martin": "#229971",
    "Aston Martin Aramco F1 Team": "#229971",
    "Alpine": "#FF87BC",
    "Alpine F1 Team": "#FF87BC",
    "Williams": "#64C4FF",
    "Williams Racing": "#64C4FF",
    "AlphaTauri": "#5E8FAA",
    "Visa Cash App RB F1 Team": "#6692FF",
    "RB": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Kick Sauber": "#52E252",
    "Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD",
}