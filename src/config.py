"""F1 Predictor Configuration Module."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Absolute filesystem directories anchored to repository root
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
SEASONS_DIR: Path = DATA_DIR / "seasons"
MODELS_DIR: Path = DATA_DIR / "models"
CACHE_DIR: Path = BASE_DIR / "f1_cache"


def load_json_data(filename: str) -> dict[str, Any]:
    """Load data from a JSON file in the data directory."""
    file_path = DATA_DIR / filename
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug("Configuration file not found: %s", file_path)
        return {}
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON file %s: %s", file_path, exc)
        return {}


# Load static track data if available
_tracks_data: dict[str, Any] = load_json_data("tracks.json")
TRACK_COORDINATES: dict[str, Any] = _tracks_data.get("TRACK_COORDINATES", {})
TRACK_CHARACTERISTICS: dict[str, Any] = _tracks_data.get("TRACK_CHARACTERISTICS", {})

# Official F1 Team Livery Colors (Vibrant, high-contrast palette for telemetry dark UI)
DEFAULT_TEAM_COLOR: str = "#FFFFFF"
TEAM_COLORS: dict[str, str] = {
    # Red Bull Racing
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "Oracle Red Bull Racing": "#3671C6",

    # Ferrari
    "Ferrari": "#E80020",
    "Scuderia Ferrari": "#E80020",
    "Scuderia Ferrari HP": "#E80020",

    # Mercedes
    "Mercedes": "#27F4D2",
    "Mercedes-AMG Petronas": "#27F4D2",
    "Mercedes-AMG Petronas F1 Team": "#27F4D2",

    # McLaren
    "McLaren": "#FF8000",
    "McLaren F1 Team": "#FF8000",

    # Aston Martin
    "Aston Martin": "#229971",
    "Aston Martin Aramco": "#229971",
    "Aston Martin Aramco F1 Team": "#229971",

    # Alpine
    "Alpine": "#0093CC",
    "Alpine F1 Team": "#0093CC",
    "BWT Alpine F1 Team": "#0093CC",

    # Williams
    "Williams": "#00A0DE",
    "Williams Racing": "#00A0DE",

    # RB / Racing Bulls / AlphaTauri / Toro Rosso
    "RB": "#6692FF",
    "Visa Cash App RB": "#6692FF",
    "Visa Cash App RB F1 Team": "#6692FF",
    "Racing Bulls": "#6692FF",
    "AlphaTauri": "#6692FF",
    "Scuderia AlphaTauri": "#6692FF",
    "Toro Rosso": "#6692FF",

    # Kick Sauber / Stake / Sauber / Alfa Romeo
    "Kick Sauber": "#52E252",
    "Stake F1 Team Kick Sauber": "#52E252",
    "Stake F1 Team": "#52E252",
    "Sauber": "#52E252",
    "Stake": "#52E252",
    "Kick": "#52E252",
    "Alfa Romeo": "#C92D4B",
    "Alfa Romeo Racing": "#C92D4B",

    # Haas
    "Haas": "#B6BABD",
    "Haas F1 Team": "#B6BABD",
    "MoneyGram Haas F1 Team": "#B6BABD",

    # Expansion & Historical Teams (including 2026 Cadillac 11th constructor)
    "Cadillac": "#C5A059",
    "Cadillac F1 Team": "#C5A059",
    "Cadillac Racing": "#C5A059",
    "Caddilac": "#C5A059",
    "Caddilac F1 Team": "#C5A059",
    "Caddilac Racing": "#C5A059",
    "Audi": "#E20613",
    "Audi F1 Team": "#E20613",
    "Renault": "#FFF500",
    "Racing Point": "#F596C8",
    "Force India": "#F596C8",
}

# 2026 Season Grid Specification (11 Constructors, 22 Drivers)
GRID_TEAMS_2026: dict[str, str] = {
    "Mercedes": "#27F4D2",
    "Ferrari": "#E80020",
    "Red Bull Racing": "#3671C6",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#00A0DE",
    "Racing Bulls": "#6692FF",
    "Audi": "#E20613",
    "Haas F1 Team": "#B6BABD",
    "Cadillac": "#C5A059",
}

# 2026 Driver lineup (22 drivers across 11 teams)
DRIVER_TEAMS_2026: dict[str, str] = {
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "COL": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",
    "BOR": "Audi",
    "HUL": "Audi",
    "BEA": "Haas F1 Team",
    "OCO": "Haas F1 Team",
    "PER": "Cadillac",
    "BOT": "Cadillac",
}

# Comprehensive Driver 3-letter abbreviation to constructor mapping fallback
DRIVER_TEAMS: dict[str, str] = {
    # 2026 Active Drivers (22 drivers)
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "COL": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",
    "BOR": "Audi",
    "HUL": "Audi",
    "BEA": "Haas F1 Team",
    "OCO": "Haas F1 Team",
    "PER": "Cadillac",
    "BOT": "Cadillac",
    # Historical recent drivers
    "TSU": "RB",
    "RIC": "RB",
    "ZHO": "Kick Sauber",
    "SAR": "Williams",
    "MAG": "Haas F1 Team",
    "DOO": "Alpine",
}


def get_team_for_driver(driver_code: str | None) -> str:
    """Return team constructor name for a given driver abbreviation."""
    if not driver_code:
        return ""
    return DRIVER_TEAMS.get(str(driver_code).strip().upper(), "")
