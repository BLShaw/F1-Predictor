"""Tests for Plotly visualization chart generation."""

import pandas as pd
import plotly.graph_objects as go

from src.ui.charts import (
    create_pace_chart,
    create_qualifying_chart,
    create_win_probability_chart,
)


def test_create_pace_chart_normal() -> None:
    """Verify pace chart generates valid Plotly Figure for standard inputs."""
    pace_df = pd.DataFrame(
        [
            {"driver": "VER", "best": 80.5},
            {"driver": "NOR", "best": 80.8},
            {"driver": "LEC", "best": 81.2},
        ]
    )
    fig = create_pace_chart(pace_df, "FP1 PACE")
    assert isinstance(fig, go.Figure)


def test_create_pace_chart_zero_gap_protection() -> None:
    """Verify pace chart handles max_gap == 0 without ZeroDivisionError."""
    # All identical times
    pace_df = pd.DataFrame(
        [
            {"driver": "VER", "best": 80.5},
            {"driver": "NOR", "best": 80.5},
        ]
    )
    fig = create_pace_chart(pace_df, "FP1 PACE")
    assert isinstance(fig, go.Figure)

    # Single driver (gap is exactly 0.0)
    single_df = pd.DataFrame([{"driver": "VER", "best": 80.5}])
    fig_single = create_pace_chart(single_df, "FP1 PACE")
    assert isinstance(fig_single, go.Figure)


def test_create_pace_chart_empty() -> None:
    """Verify pace chart returns None on empty input."""
    assert create_pace_chart(pd.DataFrame(), "FP1 PACE") is None


def test_create_qualifying_chart_valid() -> None:
    """Verify qualifying gap chart builds valid figure."""
    quali_df = pd.DataFrame(
        [
            {"position": 1, "driver": "VER", "team": "Red Bull Racing", "q1": 81.0, "q2": 80.5, "q3": 80.1},
            {"position": 2, "driver": "NOR", "team": "McLaren", "q1": 81.2, "q2": 80.6, "q3": 80.3},
            {"position": 3, "driver": "LEC", "team": "Ferrari", "q1": 81.3, "q2": 80.8, "q3": None},
        ]
    )
    fig = create_qualifying_chart(quali_df)
    assert isinstance(fig, go.Figure)


def test_create_qualifying_chart_empty() -> None:
    """Verify qualifying chart returns None on empty input."""
    assert create_qualifying_chart(pd.DataFrame()) is None


def test_create_qualifying_chart_22_drivers_cadillac() -> None:
    """Verify qualifying chart retains all 22 drivers and sets Q2 cutoff to 16.5."""
    rows = []
    for i in range(1, 23):
        team = "Cadillac" if i >= 21 else "Ferrari"
        driver = f"D_{i}"
        rows.append(
            {
                "position": i,
                "driver": driver,
                "team": team,
                "q1": 80.0 + 0.1 * i,
                "q2": 79.5 + 0.1 * i if i <= 16 else None,
                "q3": 79.0 + 0.1 * i if i <= 10 else None,
            }
        )
    quali_22 = pd.DataFrame(rows)
    fig = create_qualifying_chart(quali_22)

    assert isinstance(fig, go.Figure)
    # Verify all 22 drivers are present in the bar trace
    assert len(fig.data[0].x) == 22
    # Verify 16.5 cutoff is among the vertical divider shapes for 22-car grid
    shape_x_positions = [s.x0 for s in fig.layout.shapes]
    assert 16.5 in shape_x_positions
    assert 10.5 in shape_x_positions



def test_create_win_probability_chart_valid() -> None:
    """Verify win probability chart builds horizontal bar chart with team colors."""
    preds_df = pd.DataFrame(
        [
            {"Driver": "VER", "Win %": 0.45},
            {"Driver": "NOR", "Win %": 0.30},
            {"Driver": "LEC", "Win %": 0.25},
        ]
    )
    quali_df = pd.DataFrame(
        [
            {"driver": "VER", "team": "Red Bull Racing"},
            {"driver": "NOR", "team": "McLaren"},
            {"driver": "LEC", "team": "Ferrari"},
        ]
    )
    # Positional
    fig = create_win_probability_chart(preds_df, quali_df)
    assert isinstance(fig, go.Figure)

    # Keyword team_data
    fig_team = create_win_probability_chart(preds_df, team_data=quali_df)
    assert isinstance(fig_team, go.Figure)

    # Keyword quali_df (backwards compatibility)
    fig_quali = create_win_probability_chart(preds_df, quali_df=quali_df)
    assert isinstance(fig_quali, go.Figure)


def test_create_win_probability_chart_empty() -> None:
    """Verify win probability chart returns None on empty input."""
    assert create_win_probability_chart(pd.DataFrame()) is None
