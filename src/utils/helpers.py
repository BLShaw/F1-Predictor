"""
Helper functions for the F1 Predictor application.
"""
import pandas as pd
from src.config import TEAM_COLORS


def get_team_color(team_name: str) -> str:
    """Get F1 team color by name."""
    for key, color in TEAM_COLORS.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return color
    return "#FFFFFF"


def format_lap_time(seconds: float) -> str:
    """Format lap time in seconds to M:SS.mmm format."""
    if pd.isna(seconds) or seconds is None:
        return "—"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def format_gap(gap_seconds: float) -> str:
    """Format gap to leader."""
    if pd.isna(gap_seconds) or gap_seconds == 0:
        return "LEADER"
    return f"+{gap_seconds:.3f}s"
