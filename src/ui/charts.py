"""UI charting functions for F1 Predictor using Plotly.

Enforces a unified telemetry theme, consistent typography, standard margins,
and accessible hover formatting across all analytical figures.
"""

import pandas as pd
import plotly.graph_objects as go

from src.utils.helpers import format_gap, format_lap_time, get_team_color

CHART_FONT_FAMILY: str = "Inter, -apple-system, sans-serif"
TITLE_FONT_FAMILY: str = "Orbitron, monospace"
GRID_COLOR: str = "rgba(255, 255, 255, 0.06)"
BORDER_COLOR: str = "rgba(255, 255, 255, 0.15)"


def _apply_chart_theme(
    fig: go.Figure,
    title: str,
    height: int = 420,
    show_legend: bool = False,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Apply centralized styling and typography to Plotly figures."""
    resolved_margin = margin if margin is not None else {"l": 70, "r": 40, "t": 48, "b": 38}

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": CHART_FONT_FAMILY, "color": "#F8FAFC", "size": 11},
        title={
            "text": title,
            "font": {"family": TITLE_FONT_FAMILY, "size": 13, "color": "#F8FAFC"},
            "x": 0.01,
            "xanchor": "left",
            "y": 0.98,
        },
        height=height,
        margin=resolved_margin,
        showlegend=show_legend,
        hoverlabel={
            "bgcolor": "#13141E",
            "bordercolor": BORDER_COLOR,
            "font": {"family": CHART_FONT_FAMILY, "size": 11, "color": "#F8FAFC"},
        },
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=BORDER_COLOR,
        tickfont={"family": CHART_FONT_FAMILY, "size": 10, "color": "#94A3B8"},
        title_font={"family": CHART_FONT_FAMILY, "size": 11, "color": "#94A3B8"},
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=BORDER_COLOR,
        tickfont={"family": TITLE_FONT_FAMILY, "size": 10, "color": "#F8FAFC"},
        title_font={"family": CHART_FONT_FAMILY, "size": 11, "color": "#94A3B8"},
    )
    return fig


def create_pace_chart(pace_df: pd.DataFrame | None, session_title: str = "PRACTICE PACE ANALYSIS") -> go.Figure | None:
    """Create practice pace comparison chart with gap-to-fastest bars."""
    if pace_df is None or pace_df.empty or "best" not in pace_df.columns:
        return None

    clean_df = pace_df.dropna(subset=["best"]).copy()
    if clean_df.empty:
        return None

    clean_df = clean_df.sort_values("best")
    fastest = clean_df["best"].min()
    clean_df["gap"] = clean_df["best"] - fastest

    max_gap = float(clean_df["gap"].max()) if pd.notna(clean_df["gap"].max()) else 0.0
    gap_divisor = max_gap if max_gap > 0 else 1.0

    has_teams = "team" in clean_df.columns and clean_df["team"].notna().any()
    if has_teams:
        colors = [get_team_color(row.get("team")) for _, row in clean_df.iterrows()]
    else:
        # Subtle telemetry gradient: Cyan to Amber to Red
        colors = []
        for g in clean_df["gap"]:
            ratio = min(1.0, max(0.0, g / gap_divisor))
            if ratio < 0.5:
                # Cyan (56, 189, 248) to Amber (245, 158, 11)
                r = int(56 + (245 - 56) * (ratio * 2))
                g_c = int(189 + (158 - 189) * (ratio * 2))
                b = int(248 + (11 - 248) * (ratio * 2))
            else:
                # Amber (245, 158, 11) to Red (225, 6, 0)
                r = int(245 + (225 - 245) * ((ratio - 0.5) * 2))
                g_c = int(158 + (6 - 158) * ((ratio - 0.5) * 2))
                b = int(11 + (0 - 11) * ((ratio - 0.5) * 2))
            colors.append(f"rgb({r},{g_c},{b})")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=clean_df["driver"],
            x=clean_df["gap"],
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.25)", "width": 1}},
            text=[format_gap(g) for g in clean_df["gap"]],
            textposition="outside",
            textfont={"family": TITLE_FONT_FAMILY, "size": 10, "color": "#F8FAFC"},
            hovertemplate="<b>%{y}</b><br>Gap: +%{x:.3f}s<br>Best Lap: %{customdata}<extra></extra>",
            customdata=[format_lap_time(t) for t in clean_df["best"]],
        )
    )

    fig.update_xaxes(title="Gap to Fastest (seconds)")
    fig.update_yaxes(autorange="reversed")
    return _apply_chart_theme(fig, session_title, height=440)


def create_qualifying_chart(quali_df: pd.DataFrame | None) -> go.Figure | None:
    """Create qualifying gap visualization with dynamic FIA knockout cutoff thresholds."""
    if quali_df is None or quali_df.empty:
        return None

    clean_df = quali_df.copy()
    q_cols = [c for c in ["q3", "q2", "q1", "sq3", "sq2", "sq1"] if c in clean_df.columns]
    if not q_cols:
        return None

    for col in q_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df["best_q"] = clean_df[q_cols].min(axis=1)
    clean_df = clean_df.dropna(subset=["best_q"])
    if clean_df.empty:
        return None

    sort_col = "position" if "position" in clean_df.columns else "best_q"
    clean_df = clean_df.sort_values(sort_col)


    fastest = clean_df["best_q"].min()
    clean_df["gap"] = clean_df["best_q"] - fastest
    colors = [get_team_color(row.get("team")) for _, row in clean_df.iterrows()]

    fig = go.Figure()

    total_cars = len(clean_df)
    # FIA Sporting Regulations:
    # 20-car grid (10 teams): Q1 eliminates 5 (15 into Q2, cutoff 15.5), Q2 eliminates 5 (10 into Q3, cutoff 10.5)
    # 22-car grid (11 teams, e.g. 2026 Cadillac addition): Q1 eliminates 6 (16 into Q2, cutoff 16.5), Q2 eliminates 6 (10 into Q3, cutoff 10.5)
    q2_cutoff_x = 16.5 if total_cars >= 22 else 15.5
    q2_threshold = 16 if total_cars >= 22 else 15

    if total_cars >= 10:
        fig.add_vline(
            x=10.5,
            line_dash="dash",
            line_color="rgba(225, 6, 0, 0.5)",
            annotation_text="Q3 CUTOFF",
            annotation_position="top",
            annotation_font={"family": CHART_FONT_FAMILY, "size": 9, "color": "#E10600"},
        )
    if total_cars >= q2_threshold:
        fig.add_vline(
            x=q2_cutoff_x,
            line_dash="dash",
            line_color="rgba(245, 158, 11, 0.5)",
            annotation_text="Q2 CUTOFF",
            annotation_position="top",
            annotation_font={"family": CHART_FONT_FAMILY, "size": 9, "color": "#F59E0B"},
        )


    x_values = clean_df["position"] if "position" in clean_df.columns else list(range(1, len(clean_df) + 1))

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=clean_df["gap"],
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.3)", "width": 1}},
            text=clean_df.get("driver", clean_df.index),
            textposition="outside",
            textfont={"family": TITLE_FONT_FAMILY, "size": 9, "color": "#F8FAFC"},
            hovertemplate="<b>P%{x} - %{text}</b><br>Gap: +%{y:.3f}s<br>Team: %{customdata}<extra></extra>",
            customdata=clean_df["team"] if "team" in clean_df.columns else ["" for _ in range(len(clean_df))],
        )
    )

    fig.update_xaxes(title="Grid Position", dtick=1)
    fig.update_yaxes(title="Gap to Pole (seconds)")
    return _apply_chart_theme(fig, "QUALIFYING SPREAD TO POLE", height=400)


def create_win_probability_chart(
    predictions_df: pd.DataFrame | None,
    team_data: dict[str, str] | pd.DataFrame | None = None,
    quali_df: pd.DataFrame | None = None,
) -> go.Figure | None:
    """Create clean horizontal bar chart showing win probabilities per driver."""
    if predictions_df is None or predictions_df.empty or "Win %" not in predictions_df.columns:
        return None

    resolved_team_data = team_data if team_data is not None else quali_df
    mapping: dict[str, str] = {}
    if isinstance(resolved_team_data, pd.DataFrame):
        if (
            not resolved_team_data.empty
            and "driver" in resolved_team_data.columns
            and "team" in resolved_team_data.columns
        ):
            mapping = dict(zip(resolved_team_data["driver"], resolved_team_data["team"]))
    elif isinstance(resolved_team_data, dict):
        mapping = resolved_team_data

    top_drivers = predictions_df.sort_values("Win %", ascending=False).head(15)
    colors = [get_team_color(mapping.get(d) or d) for d in top_drivers["Driver"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=top_drivers["Driver"],
            x=top_drivers["Win %"] * 100,
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.3)", "width": 1}},
            text=[f"{p * 100:.1f}%" for p in top_drivers["Win %"]],
            textposition="outside",
            textfont={"family": TITLE_FONT_FAMILY, "size": 10, "color": "#F8FAFC"},
            hovertemplate="<b>%{y}</b><br>Win Probability: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_xaxes(title="Win Probability (%)")
    fig.update_yaxes(autorange="reversed")
    return _apply_chart_theme(fig, "PREDICTED WIN PROBABILITY DISTRIBUTION", height=420)
