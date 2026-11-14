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

OVERTAKING_DIFFICULTY_MAP: dict[str, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "very_high": 1.00,
}

TRACK_ALIASES: dict[str, str] = {
    "sakhir": "Bahrain",
    "kuala lumpur": "Bahrain",
    "bahrain": "Bahrain",
    "jeddah": "Jeddah",
    "saudi": "Jeddah",
    "melbourne": "Melbourne",
    "australian": "Melbourne",
    "suzuka": "Suzuka",
    "japanese": "Suzuka",
    "shanghai": "Shanghai",
    "chinese": "Shanghai",
    "miami gardens": "Miami",
    "miami": "Miami",
    "imola": "Imola",
    "emilia": "Imola",
    "monte carlo": "Monaco",
    "monaco": "Monaco",
    "montréal": "Montreal",
    "montreal": "Montreal",
    "canadian": "Montreal",
    "barcelona": "Barcelona",
    "madrid": "Madrid",
    "spanish": "Barcelona",
    "spielberg": "Spielberg",
    "austrian": "Spielberg",
    "silverstone": "Silverstone",
    "british": "Silverstone",
    "budapest": "Budapest",
    "hungarian": "Budapest",
    "spa-francorchamps": "Spa",
    "spa": "Spa",
    "belgian": "Spa",
    "zandvoort": "Zandvoort",
    "dutch": "Zandvoort",
    "monza": "Monza",
    "italian": "Monza",
    "baku": "Baku",
    "azerbaijan": "Baku",
    "marina bay": "Singapore",
    "singapore": "Singapore",
    "austin": "Austin",
    "united states": "Austin",
    "mexico city": "Mexico City",
    "mexico": "Mexico City",
    "são paulo": "Sao Paulo",
    "sao paulo": "Sao Paulo",
    "las vegas": "Las Vegas",
    "lusail": "Lusail",
    "qatar": "Lusail",
    "yas island": "Abu Dhabi",
    "abu dhabi": "Abu Dhabi",
    "le castellet": "Le Castellet",
    "french": "Le Castellet",
}

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


# Constructor-calibrated baseline mechanical & attrition DNF rates centered around 0.05
CONSTRUCTOR_DNF_RATES: dict[str, float] = {
    "McLaren": 0.032,
    "McLaren F1 Team": 0.032,
    "Red Bull Racing": 0.035,
    "Red Bull": 0.035,
    "Oracle Red Bull Racing": 0.035,
    "Mercedes": 0.036,
    "Mercedes-AMG Petronas": 0.036,
    "Mercedes-AMG Petronas F1 Team": 0.036,
    "Racing Bulls": 0.044,
    "RB": 0.044,
    "Visa Cash App RB": 0.044,
    "Visa Cash App RB F1 Team": 0.044,
    "AlphaTauri": 0.052,
    "Scuderia AlphaTauri": 0.052,
    "Toro Rosso": 0.052,
    "Ferrari": 0.046,
    "Scuderia Ferrari": 0.046,
    "Scuderia Ferrari HP": 0.046,
    "Kick Sauber": 0.048,
    "Stake F1 Team Kick Sauber": 0.048,
    "Stake F1 Team": 0.048,
    "Sauber": 0.048,
    "Stake": 0.048,
    "Kick": 0.048,
    "Haas F1 Team": 0.048,
    "Haas": 0.048,
    "MoneyGram Haas F1 Team": 0.048,
    "Audi": 0.052,
    "Audi F1 Team": 0.052,
    "Alpine": 0.055,
    "Alpine F1 Team": 0.055,
    "BWT Alpine F1 Team": 0.055,
    "Renault": 0.055,
    "Alfa Romeo": 0.058,
    "Alfa Romeo Racing": 0.058,
    "Aston Martin": 0.065,
    "Aston Martin Aramco": 0.065,
    "Aston Martin Aramco F1 Team": 0.065,
    "Williams": 0.072,
    "Williams Racing": 0.072,
    "Cadillac": 0.095,
    "Cadillac F1 Team": 0.095,
    "Cadillac Racing": 0.095,
    "Caddilac": 0.095,
    "Caddilac F1 Team": 0.095,
    "Caddilac Racing": 0.095,
}


def get_constructor_dnf_rate(
    team_name: str | None = None,
    driver_code: str | None = None,
    default_rate: float = 0.05,
    custom_rates: dict[str, float] | None = None,
) -> float:
    """Resolve calibrated DNF probability for a constructor or driver code."""
    resolved_team = str(team_name).strip() if team_name else ""
    if not resolved_team and driver_code:
        resolved_team = get_team_for_driver(driver_code)

    if resolved_team:
        team_lower = resolved_team.lower()
        if custom_rates:
            custom_lower = {str(k).strip().lower(): float(v) for k, v in custom_rates.items()}
            if team_lower in custom_lower:
                return custom_lower[team_lower]
        builtin_lower = {k.lower(): v for k, v in CONSTRUCTOR_DNF_RATES.items()}
        if team_lower in builtin_lower:
            return builtin_lower[team_lower]

    return default_rate


def get_track_characteristics(track_name: str | None = None, gp_name: str | None = None) -> dict[str, Any]:
    """Resolve circuit characteristics and encoded track features from track or GP name."""
    lower_to_key = {k.lower(): k for k in TRACK_CHARACTERISTICS}
    resolved_key: str | None = None

    if track_name:
        cleaned_track = str(track_name).strip().lower()
        if cleaned_track in lower_to_key:
            resolved_key = lower_to_key[cleaned_track]
        elif cleaned_track in TRACK_ALIASES and TRACK_ALIASES[cleaned_track] in TRACK_CHARACTERISTICS:
            resolved_key = TRACK_ALIASES[cleaned_track]

    if resolved_key is None and gp_name:
        cleaned_gp = str(gp_name).strip().lower()
        if cleaned_gp in lower_to_key:
            resolved_key = lower_to_key[cleaned_gp]
        elif cleaned_gp in TRACK_ALIASES and TRACK_ALIASES[cleaned_gp] in TRACK_CHARACTERISTICS:
            resolved_key = TRACK_ALIASES[cleaned_gp]
        else:
            for alias in sorted(TRACK_ALIASES, key=len, reverse=True):
                if alias in cleaned_gp:
                    candidate = TRACK_ALIASES[alias]
                    if candidate in TRACK_CHARACTERISTICS:
                        resolved_key = candidate
                        break

    info = dict(TRACK_CHARACTERISTICS[resolved_key]) if resolved_key else {}
    difficulty_str = str(info.get("overtaking_difficulty", "medium")).strip().lower()
    circuit_type = str(info.get("type", "permanent")).strip().lower()

    info.setdefault("overtaking_difficulty", difficulty_str)
    info.setdefault("type", circuit_type)
    info["track_key"] = resolved_key or ""
    info["overtaking_difficulty_score"] = OVERTAKING_DIFFICULTY_MAP.get(difficulty_str, 0.50)
    info["is_street_circuit"] = 1 if circuit_type == "street" else 0
    return info
