"""Streamlit caching and presentation-layer state adapter.

Isolates Streamlit caching decorators from core domain and model logic.
"""

from typing import Any

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import (
    get_available_gps as _get_available_gps,
)
from src.data_loader import (
    get_available_seasons as _get_available_seasons,
)
from src.data_loader import (
    get_season_schedule as _get_season_schedule,
)
from src.data_loader import (
    load_gp_data as _load_gp_data,
)
from src.model import F1MLPredictor
from src.model import load_saved_ml_model as _load_saved_ml_model


@st.cache_data
def get_cached_seasons() -> list[int]:
    """Return cached list of available championship season years."""
    return _get_available_seasons()


@st.cache_data
def get_cached_schedule(year: int) -> list[dict[str, Any]]:
    """Return cached Grand Prix calendar schedule for the specified season."""
    return _get_season_schedule(year)


@st.cache_data
def get_cached_gps(year: int) -> list[dict[str, Any]]:
    """Return cached list of Grand Prix events for the specified season."""
    return _get_available_gps(year)


@st.cache_data
def get_cached_gp_data(year: int, gp_folder: str) -> dict[str, Any]:
    """Return cached telemetry and classification dataset for a Grand Prix."""
    return _load_gp_data(year, gp_folder)


@st.cache_resource
def get_cached_model() -> RandomForestRegressor | None:
    """Return cached Random Forest model artifact from memory."""
    return _load_saved_ml_model()


@st.cache_data
def run_cached_ml_prediction(features_df: pd.DataFrame, n_sims: int = 2000, dnf_prob: float = 0.05) -> dict[str, Any]:
    """Compute and cache ML predictions and Monte Carlo distributions for an event."""
    model = get_cached_model()
    predictor = F1MLPredictor(model=model)
    return predictor.predict(features_df, n_sims=n_sims, dnf_prob=dnf_prob)
