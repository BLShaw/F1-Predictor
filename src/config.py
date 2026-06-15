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

# F1 Team Colors
TEAM_COLORS = {
    "Red Bull Racing": "#00162b",
    "Red Bull": "#00162b",
    "Ferrari": "#da291c",
    "Scuderia Ferrari": "#da291c",
    "Mercedes": "#00f5d0",
    "Mercedes-AMG Petronas F1 Team": "#00f5d0",
    "McLaren": "#ff8000",
    "McLaren F1 Team": "#ff8000",
    "Aston Martin": "#00665e",
    "Aston Martin Aramco F1 Team": "#00665e",
    "Alpine": "#061a4d",
    "Alpine F1 Team": "#061a4d",
    "Williams": "#000a20",
    "Williams Racing": "#000a20",
    "AlphaTauri": "#070b36",
    "Visa Cash App RB F1 Team": "#070b36",
    "RB": "#070b36",
    "Racing Bulls": "#070b36",
    "Alfa Romeo": "#101319",
    "Kick Sauber": "#101319",
    "Sauber": "#101319",
    "Audi": "#101319",
    "Haas F1 Team": "#e6002d",
    "Haas": "#e6002d",
    "Cadillac": "#ffffff",
    "Cadillac F1 Team": "#ffffff",
}
