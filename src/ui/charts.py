"""
UI charting functions for the F1 Predictor.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.helpers import get_team_color, format_lap_time, format_gap


def create_pace_chart(pace_df: pd.DataFrame) -> go.Figure:
    """Create interactive practice pace comparison chart."""
    if pace_df.empty:
        return None
    
    # Sort by best time
    pace_df = pace_df.sort_values("best").head(15)
    
    # Calculate gap to fastest
    fastest = pace_df["best"].min()
    pace_df["gap"] = pace_df["best"] - fastest
    
    # Color scale: green (fast) to red (slow)
    max_gap = pace_df["gap"].max()
    colors = [f"rgb({min(255, int(150 + (g/max_gap)*105))}, {max(50, int(200 - (g/max_gap)*150))}, 50)" 
              for g in pace_df["gap"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=pace_df["driver"],
        x=pace_df["gap"],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[format_gap(g) for g in pace_df["gap"]],
        textposition='outside',
        textfont=dict(family="Orbitron", size=11, color="white"),
        hovertemplate="<b>%{y}</b><br>Gap: +%{x:.3f}s<br>Best: %{customdata}<extra></extra>",
        customdata=[format_lap_time(t) for t in pace_df["best"]]
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white"),
        title=dict(
            text="PRACTICE PACE ANALYSIS",
            font=dict(family="Orbitron", size=16, color="white"),
            x=0.5
        ),
        xaxis=dict(
            title="Gap to Fastest (seconds)",
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.3)"
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickfont=dict(family="Orbitron", size=11)
        ),
        height=500,
        margin=dict(l=80, r=100, t=60, b=40),
        showlegend=False
    )
    
    return fig


def create_qualifying_chart(quali_df: pd.DataFrame) -> go.Figure:
    """Create qualifying gap visualization with team colors."""
    if quali_df.empty:
        return None
    
    # Get best Q time for each driver
    quali_df = quali_df.copy()
    
    # Find the best time column available
    for q_col in ["q3", "q2", "q1"]:
        if q_col in quali_df.columns:
            quali_df[q_col] = pd.to_numeric(quali_df[q_col], errors='coerce')
    
    # Get best qualifying time
    q_cols = [c for c in ["q3", "q2", "q1"] if c in quali_df.columns]
    if not q_cols:
        return None
    
    quali_df["best_q"] = quali_df[q_cols].min(axis=1)
    quali_df = quali_df.dropna(subset=["best_q"]).sort_values("position").head(20)
    
    if quali_df.empty:
        return None
    
    fastest = quali_df["best_q"].min()
    quali_df["gap"] = quali_df["best_q"] - fastest
    
    # Get team colors for each driver
    colors = []
    for _, row in quali_df.iterrows():
        team = row.get("team", "")
        team_color = get_team_color(team) if team else "#FFFFFF"
        colors.append(team_color)
    
    fig = go.Figure()
    
    # Add Q3 cutoff line
    if len(quali_df) >= 10:
        fig.add_vline(x=10.5, line_dash="dash", line_color="rgba(225,6,0,0.5)", 
                      annotation_text="Q3", annotation_position="top")
    if len(quali_df) >= 15:
        fig.add_vline(x=15.5, line_dash="dash", line_color="rgba(255,184,0,0.5)",
                      annotation_text="Q2", annotation_position="top")
    
    fig.add_trace(go.Bar(
        x=quali_df["position"],
        y=quali_df["gap"],
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.4)', width=1)
        ),
        text=quali_df["driver"],
        textposition='outside',
        textfont=dict(family="Orbitron", size=10, color="white"),
        hovertemplate="<b>P%{x} - %{text}</b><br>Gap: +%{y:.3f}s<br>Team: %{customdata}<extra></extra>",
        customdata=quali_df["team"] if "team" in quali_df.columns else None
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white"),
        title=dict(
            text="QUALIFYING GAPS TO POLE",
            font=dict(family="Orbitron", size=16, color="white"),
            x=0.5
        ),
        xaxis=dict(
            title="Grid Position",
            gridcolor="rgba(255,255,255,0.1)",
            dtick=1
        ),
        yaxis=dict(
            title="Gap to Pole (seconds)",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        height=400,
        margin=dict(l=60, r=40, t=60, b=40),
        showlegend=False
    )
    
    return fig


def create_prediction_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Create Monte Carlo prediction visualization."""
    if predictions_df.empty:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Win Probability", "Expected Points"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    top_10 = predictions_df.head(10)
    
    # Win probability bars
    fig.add_trace(
        go.Bar(
            y=top_10["Driver"],
            x=top_10["Win %"] * 100,
            orientation='h',
            marker=dict(
                color=top_10["Win %"],
                colorscale=[[0, "#3671C6"], [0.5, "#FFB800"], [1, "#E10600"]],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f"{p*100:.1f}%" for p in top_10["Win %"]],
            textposition='outside',
            textfont=dict(family="Orbitron", size=10, color="white"),
            hovertemplate="<b>%{y}</b><br>Win: %{x:.1f}%<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Expected points bars
    fig.add_trace(
        go.Bar(
            y=top_10["Driver"],
            x=top_10["Exp. Points"],
            orientation='h',
            marker=dict(
                color=top_10["Exp. Points"],
                colorscale=[[0, "#64C4FF"], [0.5, "#00D26A"], [1, "#E10600"]],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f"{p:.1f}" for p in top_10["Exp. Points"]],
            textposition='outside',
            textfont=dict(family="Orbitron", size=10, color="white"),
            hovertemplate="<b>%{y}</b><br>Exp. Points: %{x:.1f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="white"),
        height=450,
        showlegend=False,
        margin=dict(l=80, r=80, t=60, b=40)
    )
    
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="Orbitron", size=10))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    
    return fig
