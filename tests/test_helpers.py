"""Tests for helper formatting and color mapping functions."""

import numpy as np

from src.config import DEFAULT_TEAM_COLOR, TEAM_COLORS, get_team_for_driver
from src.utils.helpers import format_gap, format_lap_time, get_team_color


def test_get_team_color_known_teams() -> None:
    """Verify known teams map to their official hex colors."""
    assert get_team_color("Ferrari") == TEAM_COLORS["Ferrari"]
    assert get_team_color("Oracle Red Bull Racing") == TEAM_COLORS["Red Bull Racing"]
    assert get_team_color("Mercedes") == TEAM_COLORS["Mercedes"]
    assert get_team_color("McLaren") == TEAM_COLORS["McLaren"]
    assert get_team_color("Aston Martin") == TEAM_COLORS["Aston Martin"]
    assert get_team_color("Kick Sauber") == "#52E252"
    assert get_team_color("Stake F1 Team Kick Sauber") == "#52E252"
    assert get_team_color("Williams") == "#00A0DE"
    assert get_team_color("Alpine") == "#0093CC"
    assert get_team_color("RB") == "#6692FF"
    assert get_team_color("Visa Cash App RB F1 Team") == "#6692FF"
    assert get_team_color("Cadillac") == "#C5A059"
    assert get_team_color("Caddilac") == "#C5A059"
    assert get_team_color("Cadillac F1 Team") == "#C5A059"


def test_get_team_color_driver_abbreviation() -> None:
    """Verify driver abbreviations resolve to their primary team livery colors."""
    assert get_team_color("VER") == TEAM_COLORS["Red Bull Racing"]
    assert get_team_color("PER") == "#C5A059"
    assert get_team_color("BOT") == "#C5A059"
    assert get_team_color("ALB") == "#00A0DE"
    assert get_team_color("TSU") == "#6692FF"




def test_get_team_color_empty_and_whitespace() -> None:
    """Verify empty string or whitespace does not match partial keys like Red Bull."""
    assert get_team_color("") == DEFAULT_TEAM_COLOR
    assert get_team_color("   ") == DEFAULT_TEAM_COLOR
    assert get_team_color(None) == DEFAULT_TEAM_COLOR  # type: ignore[arg-type]


def test_get_team_color_unknown_fallback() -> None:
    """Verify unknown constructor names receive the fallback color."""
    assert get_team_color("Nonexistent Grand Prix Team") == DEFAULT_TEAM_COLOR


def test_format_lap_time_valid() -> None:
    """Verify standard lap seconds format into mm:ss.sss."""
    assert format_lap_time(84.123) == "1:24.123"
    assert format_lap_time(60.000) == "1:00.000"
    assert format_lap_time(125.456) == "2:05.456"
    assert format_lap_time(59.999) == "0:59.999"


def test_format_lap_time_invalid_and_edge_cases() -> None:
    """Verify invalid, negative, or missing lap times return dash placeholder."""
    assert format_lap_time(None) == "—"
    assert format_lap_time(np.nan) == "—"
    assert format_lap_time(0) == "—"
    assert format_lap_time(-10.5) == "—"
    assert format_lap_time(float("inf")) == "—"


def test_format_gap_valid() -> None:
    """Verify positive gap floats format into +x.xxxs string."""
    assert format_gap(1.234) == "+1.234s"
    assert format_gap(0.001) == "+0.001s"
    assert format_gap(10.5) == "+10.500s"


def test_format_gap_invalid_and_leader() -> None:
    """Verify zero or non-positive gaps return leader indicator."""
    assert format_gap(0.0) == "LEADER"
    assert format_gap(-0.5) == "LEADER"
    assert format_gap(None) == "LEADER"
    assert format_gap(np.nan) == "LEADER"
    assert format_gap(float("inf")) == "LEADER"


def test_get_team_for_driver() -> None:
    """Verify get_team_for_driver maps driver abbreviations to constructors."""
    assert get_team_for_driver("VER") == "Red Bull Racing"
    assert get_team_for_driver("ANT") == "Mercedes"
    assert get_team_for_driver("PER") == "Cadillac"
    assert get_team_for_driver("BOT") == "Cadillac"
    assert get_team_for_driver("UNKNOWN") == ""
    assert get_team_for_driver("") == ""
    assert get_team_for_driver(None) == ""
