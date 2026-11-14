"""Tests for data loading, grid resolution, and feature engineering."""

from typing import Any

import pandas as pd

from scripts.train_historical_model import extract_classified_race_results
from src.data_fetcher import filter_flying_laps
from src.data_loader import (
    aggregate_practice_pace,
    build_weekend_features,
    get_drivers_from_session,
    get_qualifying_results,
    get_race_results,
    resolve_starting_grid,
)


def _make_dummy_gp_data(
    year: int = 2024,
    has_sprint: bool = False,
    include_practice: bool = True,
) -> dict[str, Any]:
    """Helper to generate structured GP mock data."""
    gp_data: dict[str, Any] = {
        "year": year,
        "round": 1,
        "name": "Mock Grand Prix",
        "sessions": {},
    }

    # Qualifying
    gp_data["sessions"]["qualifying"] = {
        "session_type": "qualifying",
        "results": [
            {"driver": "VER", "position": 1, "team": "Red Bull Racing", "q3": 80.1},
            {"driver": "NOR", "position": 2, "team": "McLaren", "q3": 80.3},
            {"driver": "LEC", "position": 3, "team": "Ferrari", "q3": 80.5},
            {"driver": "HAM", "position": 4, "team": "Mercedes", "q3": 80.8},
        ],
    }

    if has_sprint:
        # Sprint race results (different finishing order)
        gp_data["sessions"]["sprint"] = {
            "session_type": "sprint",
            "results": [
                {"driver": "NOR", "position": 1, "team": "McLaren"},
                {"driver": "VER", "position": 2, "team": "Red Bull Racing"},
                {"driver": "HAM", "position": 3, "team": "Mercedes"},
                {"driver": "LEC", "position": 4, "team": "Ferrari"},
            ],
        }

    if include_practice:
        gp_data["sessions"]["fp1"] = {
            "session_type": "fp1",
            "best_times": {"VER": 81.2, "NOR": 81.5, "LEC": 81.6, "HAM": 82.0},
            "driver_laps": {
                "VER": [81.2, 81.4, 81.8],
                "NOR": [81.5, 81.7, 81.9],
                "LEC": [81.6, 81.9, 82.1],
                "HAM": [82.0, 82.2, 82.5],
            },
        }

    return gp_data


def test_resolve_starting_grid_2022_sprint_regulation() -> None:
    """Verify that in 2022 sprint weekends, Saturday Sprint results set the Sunday grid."""
    gp_2022 = _make_dummy_gp_data(year=2022, has_sprint=True)
    grid_df = resolve_starting_grid(gp_2022)

    assert not grid_df.empty
    # In sprint, NOR was 1st and VER was 2nd
    p1 = grid_df[grid_df["grid"] == 1]["driver"].iloc[0]
    p2 = grid_df[grid_df["grid"] == 2]["driver"].iloc[0]
    assert p1 == "NOR"
    assert p2 == "VER"


def test_resolve_starting_grid_2024_sprint_regulation() -> None:
    """Verify that in 2024 sprint weekends, Friday Qualifying sets the Sunday grid."""
    gp_2024 = _make_dummy_gp_data(year=2024, has_sprint=True)
    grid_df = resolve_starting_grid(gp_2024)

    assert not grid_df.empty
    # In qualifying, VER was 1st and NOR was 2nd
    p1 = grid_df[grid_df["grid"] == 1]["driver"].iloc[0]
    p2 = grid_df[grid_df["grid"] == 2]["driver"].iloc[0]
    assert p1 == "VER"
    assert p2 == "NOR"


def test_resolve_starting_grid_standard_weekend() -> None:
    """Verify standard qualifying sets the grid when no sprint is present."""
    gp_std = _make_dummy_gp_data(year=2024, has_sprint=False)
    grid_df = resolve_starting_grid(gp_std)

    assert not grid_df.empty
    p1 = grid_df[grid_df["grid"] == 1]["driver"].iloc[0]
    assert p1 == "VER"


def test_build_weekend_features_left_join_preservation() -> None:
    """Verify build_weekend_features preserves all grid drivers even if a driver missed practice."""
    gp_data = _make_dummy_gp_data(year=2024, has_sprint=False, include_practice=True)
    # Remove HAM from practice
    del gp_data["sessions"]["fp1"]["best_times"]["HAM"]
    del gp_data["sessions"]["fp1"]["driver_laps"]["HAM"]

    features = build_weekend_features(gp_data)

    # All 4 drivers from qualifying must be in features (no silent deletion)
    assert len(features) == 4
    assert "HAM" in features["driver"].values

    # HAM should have imputed pace, not NaN
    ham_row = features[features["driver"] == "HAM"].iloc[0]
    assert ham_row["has_practice_data"] == 0
    assert pd.notna(ham_row["best"])
    assert pd.notna(ham_row["consistency"])


def test_aggregate_practice_pace_empty() -> None:
    """Verify aggregate_practice_pace handles empty practice sessions gracefully."""
    gp_data: dict[str, Any] = {"year": 2024, "round": 1, "sessions": {}}
    pace_df = aggregate_practice_pace(gp_data)
    assert pace_df.empty


def test_get_qualifying_results_standard() -> None:
    """Verify get_qualifying_results parses standard session results with phase times."""
    session = {
        "session_type": "Q",
        "results": [
            {"position": 1, "driver": "VER", "team": "Red Bull Racing", "q1": 90.0, "q3": 88.5},
            {"position": 2, "driver": "NOR", "team": "McLaren", "q1": 90.2, "q3": 88.6},
        ],
    }
    df = get_qualifying_results(session)
    assert len(df) == 2
    assert df.iloc[0]["driver"] == "VER"
    assert df.iloc[0]["position"] == 1
    assert df.iloc[0]["q3"] == 88.5


def test_get_qualifying_results_shootout_null_positions_reconstruction() -> None:
    """Verify shootout session with null positions and best_times reconstructs positions and times."""
    session = {
        "session_type": "SQ",
        "session_name": "Sprint Qualifying",
        "best_times": {"VER": 88.461, "NOR": 87.869, "ANT": 88.091},
        "results": [
            {"position": None, "driver": "VER", "team": "Red Bull Racing"},
            {"position": None, "driver": "NOR", "team": "McLaren"},
            {"position": None, "driver": "ANT", "team": "Mercedes"},
        ],
    }
    df = get_qualifying_results(session)
    assert len(df) == 3
    # Sorted by best lap time: NOR (87.869) -> ANT (88.091) -> VER (88.461)
    assert df.iloc[0]["driver"] == "NOR"
    assert df.iloc[0]["position"] == 1
    assert df.iloc[0]["sq1"] == 87.869
    assert df.columns.is_unique

    assert df.iloc[1]["driver"] == "ANT"
    assert df.iloc[1]["position"] == 2
    assert df.iloc[1]["sq1"] == 88.091

    assert df.iloc[2]["driver"] == "VER"
    assert df.iloc[2]["position"] == 3


def test_get_qualifying_results_from_best_times_only() -> None:
    """Verify get_qualifying_results constructs a valid classification from best_times alone."""
    session = {
        "session_type": "SS",
        "session_name": "Sprint Shootout",
        "best_times": {"LEC": 90.1, "PIA": 89.9, "HAM": 90.5},
    }
    df = get_qualifying_results(session)
    assert len(df) == 3
    # PIA fastest (89.9) -> LEC (90.1) -> HAM (90.5)
    assert df.iloc[0]["driver"] == "PIA"
    assert df.iloc[0]["position"] == 1
    assert df.iloc[0]["team"] == "McLaren"

    assert df.iloc[1]["driver"] == "LEC"
    assert df.iloc[1]["position"] == 2
    assert df.iloc[1]["team"] == "Ferrari"


def test_get_qualifying_results_empty_or_none() -> None:
    """Verify empty or None session input returns empty DataFrame."""
    assert get_qualifying_results(None).empty
    assert get_qualifying_results({}).empty
    assert get_qualifying_results({"results": []}).empty


def test_get_qualifying_results_columns_are_strictly_unique() -> None:
    """Verify get_qualifying_results never returns duplicate column names."""
    session = {
        "session_type": "SQ",
        "session_name": "Sprint Qualifying",
        "best_times": {"VER": 88.461, "NOR": 87.869},
        "results": [
            {"position": None, "driver": "VER", "team": "Red Bull Racing"},
            {"position": None, "driver": "NOR", "team": "McLaren"},
        ],
    }
    df = get_qualifying_results(session)
    assert df.columns.is_unique
    assert len(df.columns) == len(set(df.columns))


def test_get_race_results_malformed_or_corrupt() -> None:
    """Verify get_race_results gracefully handles None, non-dict, and non-list results."""
    assert get_race_results(None).empty
    assert get_race_results({}).empty
    assert get_race_results({"results": "corrupted_string"}).empty
    assert get_race_results({"results": 12345}).empty
    assert get_race_results({"results": None}).empty
    valid_data = {"results": [{"position": 1, "driver": "VER", "team": "Red Bull Racing"}]}
    df = get_race_results(valid_data)
    assert not df.empty
    assert df.iloc[0]["driver"] == "VER"


def test_get_drivers_from_session_malformed_or_corrupt() -> None:
    """Verify get_drivers_from_session gracefully handles None, non-dict, and non-list results."""
    assert get_drivers_from_session(None).empty
    assert get_drivers_from_session({}).empty
    assert get_drivers_from_session({"results": "corrupted_string"}).empty
    assert get_drivers_from_session({"results": 42}).empty
    assert get_drivers_from_session({"results": None}).empty
    valid_data = {
        "results": [
            {"driver": "VER", "driver_number": "1", "team": "Red Bull Racing"},
            {"driver": "VER", "driver_number": "1", "team": "Red Bull Racing"},
        ]
    }
    df = get_drivers_from_session(valid_data)
    assert len(df) == 1
    assert df.iloc[0]["driver"] == "VER"


def test_build_weekend_features_is_sprint_flag() -> None:
    """Verify build_weekend_features sets is_sprint=1 on sprint weekends and 0 on standard weekends."""
    gp_std = _make_dummy_gp_data(year=2024, has_sprint=False, include_practice=True)
    features_std = build_weekend_features(gp_std)
    assert (features_std["is_sprint"] == 0).all()

    gp_sprint = _make_dummy_gp_data(year=2024, has_sprint=True, include_practice=True)
    features_sprint = build_weekend_features(gp_sprint)
    assert (features_sprint["is_sprint"] == 1).all()


def test_filter_flying_laps_handles_none_deleted_column() -> None:
    """Verify filter_flying_laps handles object Deleted values (None/True/False) without TypeError."""
    laps = pd.DataFrame(
        {
            "LapTime": pd.to_timedelta([80.1, 81.5, 82.0], unit="s"),
            "Deleted": pd.Series([True, None, False], dtype=object),
            "IsAccurate": pd.Series([True, True, None], dtype=object),
        }
    )
    valid = filter_flying_laps(laps)
    assert valid == [81.5]


def test_extract_classified_race_results_excludes_retired_top16() -> None:
    """Verify retired drivers finishing inside top 16 are excluded from classified training labels."""
    race_session = {
        "results": [
            {"driver": "VER", "position": 1, "status": "Finished"},
            {"driver": "NOR", "position": 2, "status": "+1 Lap"},
            {"driver": "ALB", "position": 3, "status": "Lapped"},
            {"driver": "HAM", "position": 14, "status": "Retired"},
            {"driver": "LEC", "position": 15, "status": "Accident"},
            {"driver": "SAI", "position": 16, "status": "Engine"},
        ]
    }
    classified = extract_classified_race_results(race_session)
    assert classified == {"VER": 1, "NOR": 2, "ALB": 3}


def test_build_weekend_features_enriched_columns() -> None:
    """Verify quali_gap_norm, circuit metadata, and pre-race rolling form columns are populated."""
    gp_data = _make_dummy_gp_data(year=2024, has_sprint=False, include_practice=True)
    gp_data["sessions"]["qualifying"]["track"] = "Monte Carlo"
    gp_data["sessions"]["qualifying"]["gp_name"] = "Monaco Grand Prix"

    rolling = ({"VER": 0.05, "NOR": 0.15}, {"Red Bull Racing": 0.10, "McLaren": 0.20})
    features = build_weekend_features(gp_data, rolling_form=rolling)

    for col in ["quali_gap_norm", "overtaking_difficulty", "is_street_circuit", "driver_form", "constructor_form"]:
        assert col in features.columns
        assert features[col].notna().all()

    assert (features["is_street_circuit"] == 1).all()
    assert (features["overtaking_difficulty"] == 1.0).all()

    ver_row = features[features["driver"] == "VER"].iloc[0]
    ham_row = features[features["driver"] == "HAM"].iloc[0]
    assert ver_row["quali_gap_norm"] == 0.0
    assert ham_row["quali_gap_norm"] == 1.0
    assert ver_row["driver_form"] == 0.05
    assert ver_row["constructor_form"] == 0.10
    # Drivers absent from rolling history fall back to normalized grid position
    assert ham_row["driver_form"] == 1.0

