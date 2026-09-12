"""Helper functions for the F1 Predictor application."""

import math
from typing import Any

import pandas as pd

from src.config import DEFAULT_TEAM_COLOR, DRIVER_TEAMS, TEAM_COLORS


def get_team_color(team_name: Any) -> str:
    """Get F1 team color by name or driver abbreviation with robust fallback."""
    if not team_name or not isinstance(team_name, str):
        return DEFAULT_TEAM_COLOR

    cleaned_name = team_name.strip()
    if not cleaned_name:
        return DEFAULT_TEAM_COLOR

    # Driver abbreviation resolution (e.g. 'VER', 'HAM', 'BOT')
    upper_name = cleaned_name.upper()
    if upper_name in DRIVER_TEAMS:
        cleaned_name = DRIVER_TEAMS[upper_name]

    lowered = cleaned_name.lower()

    # Exact match priority
    for key, color in TEAM_COLORS.items():
        if key.lower() == lowered:
            return color

    # Substring match (prioritize longer keys for specificity)
    sorted_teams = sorted(TEAM_COLORS.items(), key=lambda x: len(x[0]), reverse=True)
    for key, color in sorted_teams:
        k_lowered = key.lower()
        if k_lowered in lowered:
            return color

    for key, color in sorted_teams:
        k_lowered = key.lower()
        if lowered in k_lowered and len(lowered) >= 3:
            return color

    return DEFAULT_TEAM_COLOR


def format_lap_time(seconds: float | None) -> str:
    """Format lap time in seconds to M:SS.mmm format."""
    if seconds is None or pd.isna(seconds):
        return "—"

    try:
        sec_val = float(seconds)
    except (ValueError, TypeError):
        return "—"

    if math.isnan(sec_val) or math.isinf(sec_val) or sec_val <= 0:
        return "—"

    mins = int(sec_val // 60)
    remaining_secs = sec_val % 60
    return f"{mins}:{remaining_secs:06.3f}"


def format_gap(gap_seconds: float | None) -> str:
    """Format gap to leader in seconds."""
    if gap_seconds is None or pd.isna(gap_seconds):
        return "LEADER"

    try:
        val = float(gap_seconds)
    except (ValueError, TypeError):
        return "LEADER"

    if math.isnan(val) or math.isinf(val) or val <= 0:
        return "LEADER"

    return f"+{val:.3f}s"
