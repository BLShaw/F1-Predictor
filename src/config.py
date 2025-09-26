# Configuration and constants
CACHE_DIR = "f1_cache"
WEATHER_API_KEY = "YOURAPIKEY"  # Replace with actual API key

# Track coordinates for weather data
TRACK_COORDINATES = {
    "Monaco": {"lat": 43.7384, "lon": 7.4246},
    "Silverstone": {"lat": 52.0786, "lon": -1.0167},
    "Spa-Francorchamps": {"lat": 50.4372, "lon": 5.9713},
    "Hungaroring": {"lat": 47.5789, "lon": 19.2485},
    "Zandvoort": {"lat": 52.3888, "lon": 4.5407},
    "Monza": {"lat": 45.6234, "lon": 9.2976},
    "Suzuka": {"lat": 34.8431, "lon": 136.5387},
    "Austin": {"lat": 30.1328, "lon": -97.6411},
    "Mexico City": {"lat": 19.4042, "lon": -99.0907},
    "São Paulo": {"lat": -23.7037, "lon": -46.6995},
    "Las Vegas": {"lat": 36.1581, "lon": -115.1678},
    "Yas Marina": {"lat": 24.4749, "lon": 54.6038},
    "Bahrain": {"lat": 26.0420, "lon": 50.5042}
}

# Default forecast time (will be updated based on race)
FORECAST_TIME = "2025-05-25 13:00:00"

# Clean air race pace data (general data, can be track-specific if needed)
CLEAN_AIR_RACE_PACE = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, 
    "ALO": 94.784333, "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, 
    "STR": 95.318250, "HUL": 95.345455, "OCO": 95.682128
}

# Team points data (2024 season final points)
TEAM_POINTS = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, 
    "Ferrari": 114, "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, 
    "Racing Bulls": 10, "Alpine": 7
}

# Driver to team mapping
DRIVER_TO_TEAM = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", 
    "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", 
    "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Kick Sauber", 
    "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams"
}

# Average position change at tracks (general data, can be track-specific if needed)
# These values would be specific to each track - this is a simplified approach
AVERAGE_POSITION_CHANGE = {
    "VER": -1.0, "NOR": 1.0, "PIA": 0.2, "RUS": 0.5, "SAI": -0.3,
    "ALB": 0.8, "LEC": -1.5, "OCO": -0.2, "HAM": 0.3, "STR": 1.1,
    "GAS": -0.4, "ALO": -0.6, "HUL": 0.0
}

# Track-specific average position change (if available)
TRACK_SPECIFIC_POSITION_CHANGE = {
    "Monaco": {
        "VER": -1.0, "NOR": 1.0, "PIA": 0.2, "RUS": 0.5, "SAI": -0.3,
        "ALB": 0.8, "LEC": -1.5, "OCO": -0.2, "HAM": 0.3, "STR": 1.1,
        "GAS": -0.4, "ALO": -0.6, "HUL": 0.0
    }
}

# Default 2025 Qualifying data (placeholder - actual data should be loaded based on race)
QUALIFYING_2025_DATA = {
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [
        70.669, 69.954, 70.129, None, 71.362, 71.213, 70.063, 70.942,
        70.382, 72.563, 71.994, 70.924, 71.596
    ]
}

# F1 race schedule information
RACE_SCHEDULE = [
    {"name": "Bahrain", "location": "Bahrain International Circuit", "round": 1, "date": "2025-03-09"},
    {"name": "Saudi Arabia", "location": "Jeddah Corniche Circuit", "round": 2, "date": "2025-03-23"},
    {"name": "Australia", "location": "Albert Park Circuit", "round": 3, "date": "2025-04-06"},
    {"name": "Japan", "location": "Suzuka Circuit", "round": 4, "date": "2025-04-13"},
    {"name": "China", "location": "Shanghai International Circuit", "round": 5, "date": "2025-04-20"},
    {"name": "Miami", "location": "Miami International Autodrome", "round": 6, "date": "2025-05-04"},
    {"name": "Emilia Romagna", "location": "Imola Circuit", "round": 7, "date": "2025-05-18"},
    {"name": "Monaco", "location": "Circuit de Monaco", "round": 8, "date": "2025-05-25"},
    {"name": "Canada", "location": "Circuit Gilles Villeneuve", "round": 9, "date": "2025-06-08"},
    {"name": "Spain", "location": "Circuit de Barcelona-Catalunya", "round": 10, "date": "2025-06-22"},
    {"name": "Austria", "location": "Red Bull Ring", "round": 11, "date": "2025-06-29"},
    {"name": "Great Britain", "location": "Silverstone Circuit", "round": 12, "date": "2025-07-06"},
    {"name": "Hungary", "location": "Hungaroring", "round": 13, "date": "2025-07-20"},
    {"name": "Belgium", "location": "Circuit de Spa-Francorchamps", "round": 14, "date": "2025-07-27"},
    {"name": "Netherlands", "location": "Circuit Zandvoort", "round": 15, "date": "2025-08-24"},
    {"name": "Italy", "location": "Monza Circuit", "round": 16, "date": "2025-09-07"},
    {"name": "Azerbaijan", "location": "Baku City Circuit", "round": 17, "date": "2025-09-14"},
    {"name": "Singapore", "location": "Marina Bay Street Circuit", "round": 18, "date": "2025-09-21"},
    {"name": "United States", "location": "Circuit of the Americas", "round": 19, "date": "2025-10-05"},
    {"name": "Mexico", "location": "Autódromo Hermanos Rodríguez", "round": 20, "date": "2025-10-26"},
    {"name": "Brazil", "location": "Interlagos Circuit", "round": 21, "date": "2025-11-02"},
    {"name": "Las Vegas", "location": "Las Vegas Strip Circuit", "round": 22, "date": "2025-11-16"},
    {"name": "Qatar", "location": "Losail International Circuit", "round": 23, "date": "2025-11-23"},
    {"name": "Abu Dhabi", "location": "Yas Marina Circuit", "round": 24, "date": "2025-11-30"}
]