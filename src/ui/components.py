"""Reusable UI components and design system primitives for F1 Predictor.

Provides a unified visual language, consistent information density,
standardized typography, and component reusability across all views.
"""

from typing import Any

import pandas as pd
import streamlit as st

from src.utils.helpers import get_team_color


def render_app_header() -> None:
    """Render the primary application header."""
    st.markdown(
        """
        <div class="f1-app-header">
            <div class="header-content">
                <div class="header-brand">
                    <span class="brand-title">FORMULA 1 PREDICTOR</span>
                    <span class="brand-badge">v2.2.0</span>
                </div>
                <div class="header-subtitle">
                    Race Outcome Modeling | Calibrated Monte Carlo Simulation | Telemetry Analysis
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gp_banner(name: str, round_num: int | str, season: int | str, is_sprint: bool = False) -> None:
    """Render the Grand Prix event banner."""
    sprint_badge = (
        '<span class="badge badge-warning">SPRINT WEEKEND</span>'
        if is_sprint
        else '<span class="badge badge-neutral">STANDARD FORMAT</span>'
    )

    st.markdown(
        f"""
        <div class="f1-gp-banner">
            <div class="gp-info">
                <div class="gp-title">{name}</div>
                <div class="gp-meta">Round {round_num} | {season} Championship Season</div>
            </div>
            <div class="gp-badges">
                {sprint_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str | None = None) -> None:
    """Render a standardized section header with uniform vertical rhythm."""
    subtitle_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="f1-section-header">
            <div class="section-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(title: str, message: str, variant: str = "info") -> None:
    """Render a structured notice banner.

    Args:
        title: Notice headline.
        message: Detailed explanatory text.
        variant: Semantic style ('info', 'warning', 'neutral').
    """
    st.markdown(
        f"""
        <div class="f1-notice notice-{variant}">
            <div class="notice-title">{title}</div>
            <div class="notice-body">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[dict[str, Any]]) -> None:
    """Render a responsive horizontal grid of telemetry metric cards.

    Args:
        metrics: List of dicts with keys 'label', 'value', and optional 'delta'.
    """
    if not metrics:
        return

    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            delta_html = ""
            if "delta" in m and m["delta"] is not None:
                delta_class = "delta-neutral"
                delta_text = str(m["delta"])
                if delta_text.startswith("+") or "complete" in delta_text.lower():
                    delta_class = "delta-positive"
                elif delta_text.startswith("-") or "incomplete" in delta_text.lower():
                    delta_class = "delta-negative"
                delta_html = f'<div class="metric-delta {delta_class}">{delta_text}</div>'

            st.markdown(
                f"""
                <div class="f1-metric-card">
                    <div class="metric-label">{m.get("label", "")}</div>
                    <div class="metric-value">{m.get("value", "—")}</div>
                    {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_podium(podium_df: pd.DataFrame, points_key: str = "Points") -> None:
    """Render a standardized 3-card podium presentation.

    Args:
        podium_df: Top 3 classified rows with driver, team, and points.
        points_key: Column name holding points values.
    """
    if podium_df.empty:
        return

    podium_positions = ["P1", "P2", "P3"]
    cols = st.columns(3)

    for i, (col, (_, row)) in enumerate(zip(cols, podium_df.head(3).iterrows())):
        with col:
            team_name = str(row.get("team", "Unknown"))
            driver_name = str(row.get("driver", "—"))
            team_color = get_team_color(team_name)

            raw_pts = row.get(points_key, 0)
            pts_display = f"{int(raw_pts)} PTS" if pd.notna(raw_pts) else "—"

            st.markdown(
                f"""
                <div class="f1-podium-card" style="border-top: 3px solid {team_color};">
                    <div class="podium-rank">{podium_positions[i]}</div>
                    <div class="podium-driver">{driver_name}</div>
                    <div class="podium-team" style="color: {team_color};">{team_name}</div>
                    <div class="podium-points">{pts_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_state(title: str, message: str, hint: str | None = None) -> None:
    """Render a calm, technical empty state."""
    hint_html = f'<div class="empty-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="f1-empty-state">
            <div class="empty-title">{title}</div>
            <div class="empty-message">{message}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_specification(
    model_type: str,
    simulations: int,
    dnf_rate: float,
    train_seasons: str = "2022-2024",
    test_season: str = "2026",
) -> None:
    """Render the dual-card technical specification for ML and Monte Carlo engines."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="f1-spec-card">
                <div class="spec-header">PREDICTION REGRESSOR</div>
                <table class="spec-table">
                    <tr><td>Architecture</td><td>{model_type}</td></tr>
                    <tr><td>Estimators</td><td>300 Trees</td></tr>
                    <tr><td>Max Depth</td><td>5 Levels</td></tr>
                    <tr><td>Min Split Samples</td><td>8</td></tr>
                    <tr><td>Validation Split</td><td>Temporal Out-of-Time</td></tr>
                    <tr><td>Training Set</td><td>{train_seasons} (1,187 samples)</td></tr>
                    <tr><td>Test Set</td><td>{test_season} (227 samples)</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="f1-spec-card">
                <div class="spec-header">MONTE CARLO SIMULATION</div>
                <table class="spec-table">
                    <tr><td>Iterations</td><td>{simulations:,} Runs</td></tr>
                    <tr><td>Baseline DNF Rate</td><td>{dnf_rate * 100:.1f}%</td></tr>
                    <tr><td>Stochastic Noise</td><td>Normal(0, 1.4)</td></tr>
                    <tr><td>Points Allocation</td><td>Classified Finishers Only</td></tr>
                    <tr><td>DNF Rule</td><td>Withheld from Top-10 Points</td></tr>
                    <tr><td>Explainability</td><td>TreeSHAP Global & Local</td></tr>
                    <tr><td>Outputs</td><td>Win, Podium, Points Probabilities</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
