"""UI tab modules for the F1 Predictor application."""

from src.ui.tabs.overview import render_overview_tab
from src.ui.tabs.practice import render_practice_tab
from src.ui.tabs.predictions import render_predictions_tab
from src.ui.tabs.qualifying import render_qualifying_tab
from src.ui.tabs.race import render_race_tab
from src.ui.tabs.sprint import render_sprint_tab

__all__ = [
    "render_overview_tab",
    "render_practice_tab",
    "render_predictions_tab",
    "render_qualifying_tab",
    "render_race_tab",
    "render_sprint_tab",
]
