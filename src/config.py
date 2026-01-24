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