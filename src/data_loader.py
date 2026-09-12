"""F1 Data Loader Module.

Framework-agnostic functions to load and transform Grand Prix session data from disk.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import SEASONS_DIR, get_team_for_driver

logger = logging.getLogger(__name__)


def get_available_seasons(seasons_path: Path = SEASONS_DIR) -> list[int]:
    """Get sorted list of seasons with available local JSON data."""
    if not seasons_path.exists():
        return []

    seasons: list[int] = []
    for folder in seasons_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            seasons.append(int(folder.name))

    return sorted(seasons, reverse=True)


def get_season_schedule(year: int, seasons_path: Path = SEASONS_DIR) -> list[dict[str, Any]]:
    """Load season calendar schedule from JSON file."""
    schedule_path = seasons_path / str(year) / "schedule.json"
    if schedule_path.exists():
        try:
            with open(schedule_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load schedule for %d: %s", year, exc)
    return []


def get_available_gps(year: int, seasons_path: Path = SEASONS_DIR) -> list[dict[str, Any]]:
    """Get list of Grand Prix events with round number and available sessions."""
    season_path = seasons_path / str(year)
    if not season_path.exists():
        return []

    gps: list[dict[str, Any]] = []
    for folder in sorted(season_path.iterdir()):
        if folder.is_dir() and not folder.name.startswith("."):
            parts = folder.name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                round_num = int(parts[0])
                gp_name = parts[1].replace("_", " ")
                sessions = get_available_sessions(year, folder.name, seasons_path)

                gps.append(
                    {
                        "round": round_num,
                        "name": gp_name,
                        "folder": folder.name,
                        "sessions": sessions,
                    }
                )

    return sorted(gps, key=lambda x: x["round"])


def get_available_sessions(year: int, gp_folder: str, seasons_path: Path = SEASONS_DIR) -> dict[str, bool]:
    """Check which session JSON files exist for a Grand Prix event."""
    gp_path = seasons_path / str(year) / gp_folder
    session_files = {
        "fp1": "fp1.json",
        "fp2": "fp2.json",
        "fp3": "fp3.json",
        "qualifying": "qualifying.json",
        "sprint_qualifying": "sprint_qualifying.json",
        "sprint_shootout": "sprint_shootout.json",
        "sprint": "sprint.json",
        "race": "race.json",
    }
    return {session_name: (gp_path / filename).exists() for session_name, filename in session_files.items()}


def load_session(
    year: int, gp_folder: str, session_type: str, seasons_path: Path = SEASONS_DIR
) -> dict[str, Any] | None:
    """Load a single session from a JSON file."""
    gp_path = seasons_path / str(year) / gp_folder
    session_path = gp_path / f"{session_type}.json"

    if session_path.exists():
        try:
            with open(session_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load session %s from %s: %s", session_type, session_path, exc)
    return None


def load_gp_data(year: int, gp_folder: str, seasons_path: Path = SEASONS_DIR) -> dict[str, Any]:
    """Load all available session data and metadata for a Grand Prix."""
    sessions = get_available_sessions(year, gp_folder, seasons_path)

    gp_data: dict[str, Any] = {
        "year": year,
        "gp_folder": gp_folder,
        "sessions": {},
    }

    for session_type, is_available in sessions.items():
        if is_available:
            loaded = load_session(year, gp_folder, session_type, seasons_path)
            if loaded is not None:
                gp_data["sessions"][session_type] = loaded

    metadata_path = seasons_path / str(year) / gp_folder / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                gp_data["metadata"] = json.load(f)
        except Exception:
            gp_data["metadata"] = {}

    return gp_data


def get_driver_best_times(session_data: dict[str, Any] | None) -> dict[str, float]:
    """Extract best lap times per driver from session dictionary."""
    if not session_data or "best_times" not in session_data:
        return {}
    return session_data.get("best_times", {})


def get_qualifying_results(session_data: dict[str, Any] | None) -> pd.DataFrame:
    """Convert qualifying or sprint shootout session results to a DataFrame.

    If stored session results lack classified positions or lap times, attempts to
    reconstruct the classification using session best lap times and driver team mappings.
    """
    if not session_data or not isinstance(session_data, dict):
        return pd.DataFrame()

    results = session_data.get("results")
    raw_best_times = session_data.get("best_times")
    best_times: dict[str, float] = raw_best_times if isinstance(raw_best_times, dict) else {}

    session_name = str(session_data.get("session_name", "")).lower()
    is_shootout = session_data.get("session_type") in ("SQ", "SS") or "sprint" in session_name
    primary_phase_col = "sq1" if is_shootout else "q1"

    if results and isinstance(results, list):
        df = pd.DataFrame(results)
    elif best_times:
        sorted_times = sorted(
            best_times.items(),
            key=lambda x: (x[1] is None, x[1] if x[1] is not None else float("inf")),
        )
        rows = [
            {
                "position": idx,
                "driver": drv,
                "team": get_team_for_driver(drv),
                primary_phase_col: t,
            }
            for idx, (drv, t) in enumerate(sorted_times, 1)
        ]
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    # Populate phase lap times from best_times if phase times are absent
    has_times = any(c in df.columns for c in ["q1", "q2", "q3", "sq1", "sq2", "sq3"])
    if not has_times and best_times and "driver" in df.columns:
        df[primary_phase_col] = df["driver"].map(best_times)

    # Reconstruct rank order if positions are unassigned or null
    has_valid_positions = "position" in df.columns and df["position"].notna().any()
    if not has_valid_positions:
        time_col = None
        for col in ["q3", "sq3", "q2", "sq2", "q1", "sq1"]:
            if col in df.columns and df[col].notna().any():
                time_col = col
                break

        if time_col:
            df = df.sort_values(time_col, na_position="last").reset_index(drop=True)
            df["position"] = range(1, len(df) + 1)
        elif best_times and "driver" in df.columns:
            df["_sort_time"] = df["driver"].map(best_times)
            df = df.sort_values("_sort_time", na_position="last").reset_index(drop=True)
            df["position"] = range(1, len(df) + 1)
            df = df.drop(columns=["_sort_time"])

    # Fallback to driver dictionary for missing team values
    if "team" in df.columns and "driver" in df.columns:
        df["team"] = df.apply(
            lambda r: r["team"]
            if pd.notna(r["team"]) and str(r["team"]).strip()
            else get_team_for_driver(str(r["driver"])),
            axis=1,
        )

    return df


def get_race_results(session_data: dict[str, Any] | None) -> pd.DataFrame:
    """Convert race or sprint session results to a DataFrame."""
    if not session_data or not isinstance(session_data, dict):
        return pd.DataFrame()
    raw_results = session_data.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        return pd.DataFrame()
    return pd.DataFrame(raw_results)


def get_drivers_from_session(session_data: dict[str, Any] | None) -> pd.DataFrame:
    """Extract driver code, number, and team from session results."""
    if not session_data or not isinstance(session_data, dict):
        return pd.DataFrame()

    raw_results = session_data.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        return pd.DataFrame()

    results = pd.DataFrame(raw_results)
    if results.empty:
        return pd.DataFrame()

    driver_cols = [c for c in ["driver", "driver_number", "team"] if c in results.columns]
    return results[driver_cols].drop_duplicates()



def resolve_starting_grid(gp_data: dict[str, Any]) -> pd.DataFrame:
    """Resolve the authentic starting grid for the Sunday Grand Prix.

    Accounts for 2022 Sporting Regulations where Saturday Sprint determined the Sunday grid.
    For 2023+, Qualifying determines the Sunday Grand Prix grid.
    """
    sessions = gp_data.get("sessions", {})
    year = gp_data.get("year", 2024)

    # 2022 Sprint Weekend: Sprint race results set the Sunday Grand Prix starting grid
    if year == 2022 and "sprint" in sessions:
        sprint_results = get_race_results(sessions["sprint"])
        if not sprint_results.empty and "position" in sprint_results.columns:
            grid_df = sprint_results[["driver", "position", "team"]].copy()
            grid_df = grid_df.rename(columns={"position": "grid"})
            grid_df["grid"] = pd.to_numeric(grid_df["grid"], errors="coerce")
            grid_df = grid_df.dropna(subset=["grid"])
            if not grid_df.empty:
                return grid_df.sort_values("grid").reset_index(drop=True)

    # Standard / 2023+ Sprint: Qualifying sets the Sunday Grand Prix starting grid
    if "qualifying" in sessions:
        quali_results = get_qualifying_results(sessions["qualifying"])
        if not quali_results.empty and "position" in quali_results.columns:
            grid_df = quali_results[["driver", "position"]].copy()
            grid_df = grid_df.rename(columns={"position": "grid"})
            if "team" in quali_results.columns:
                grid_df["team"] = quali_results["team"]
            else:
                grid_df["team"] = ""
            grid_df["grid"] = pd.to_numeric(grid_df["grid"], errors="coerce")
            grid_df = grid_df.dropna(subset=["grid"])
            if not grid_df.empty:
                return grid_df.sort_values("grid").reset_index(drop=True)

    # Fallback to Sprint Qualifying or Sprint Shootout if full Qualifying is absent
    for sq_type in ["sprint_qualifying", "sprint_shootout"]:
        if sq_type in sessions:
            sq_results = get_qualifying_results(sessions[sq_type])
            if not sq_results.empty and "position" in sq_results.columns:
                grid_df = sq_results[["driver", "position"]].copy()
                grid_df = grid_df.rename(columns={"position": "grid"})
                grid_df["team"] = sq_results.get("team", "")
                grid_df["grid"] = pd.to_numeric(grid_df["grid"], errors="coerce")
                grid_df = grid_df.dropna(subset=["grid"])
                if not grid_df.empty:
                    return grid_df.sort_values("grid").reset_index(drop=True)

    # Ultimate fallback: grid_position from race.json if race already ran
    if "race" in sessions:
        race_results = get_race_results(sessions["race"])
        if not race_results.empty and "grid_position" in race_results.columns:
            grid_df = race_results[["driver", "grid_position", "team"]].copy()
            grid_df = grid_df.rename(columns={"grid_position": "grid"})
            grid_df["grid"] = pd.to_numeric(grid_df["grid"], errors="coerce")
            grid_df = grid_df.dropna(subset=["grid"])
            if not grid_df.empty:
                return grid_df.sort_values("grid").reset_index(drop=True)

    return pd.DataFrame(columns=["driver", "grid", "team"])


def aggregate_practice_pace(gp_data: dict[str, Any]) -> pd.DataFrame:
    """Aggregate pace data from practice sessions (FP1, FP2, FP3).

    Calculates best lap time, stint consistency (standard deviation),
    and whether only a single practice session was held (e.g. sprint weekends).
    """
    sessions = gp_data.get("sessions", {})
    practice_keys = ["fp1", "fp2", "fp3"]

    available_fp = [k for k in practice_keys if sessions.get(k)]
    if not available_fp:
        return pd.DataFrame(columns=["driver", "best", "avg", "consistency", "is_single_practice"])

    is_single_practice = len(available_fp) == 1
    driver_paces: dict[str, list[float]] = {}
    stint_consistencies: dict[str, list[float]] = {}

    for fp_key in available_fp:
        session = sessions[fp_key]
        best_times = get_driver_best_times(session)
        pace_const_map = session.get("pace_consistency", {})

        for driver, best_time in best_times.items():
            if pd.notna(best_time) and best_time > 0:
                driver_paces.setdefault(driver, []).append(float(best_time))

        if isinstance(pace_const_map, dict):
            for driver, stdev in pace_const_map.items():
                if pd.notna(stdev) and stdev is not None and stdev > 0:
                    stint_consistencies.setdefault(driver, []).append(float(stdev))

    if not driver_paces:
        return pd.DataFrame(columns=["driver", "best", "avg", "consistency", "is_single_practice"])

    rows: list[dict[str, Any]] = []
    for driver, times in driver_paces.items():
        best_lap = float(min(times))
        avg_lap = float(sum(times) / len(times))

        driver_consistencies = stint_consistencies.get(driver, [])
        if driver_consistencies:
            consistency = float(sum(driver_consistencies) / len(driver_consistencies))
        elif len(times) >= 2:
            # Fallback consistency from variation between session bests
            consistency = float(pd.Series(times).std())
        else:
            # Explicit indicator of unmeasured stint consistency
            consistency = None

        rows.append(
            {
                "driver": driver,
                "best": round(best_lap, 3),
                "avg": round(avg_lap, 3),
                "consistency": round(consistency, 3) if consistency is not None else None,
                "is_single_practice": is_single_practice,
            }
        )

    return pd.DataFrame(rows)


def build_weekend_features(gp_data: dict[str, Any]) -> pd.DataFrame:
    """Build pre-race feature set anchored to starting grid using LEFT JOIN.

    Prevents silent driver deletion: drivers on grid without practice laps
    are retained with missingness indicators and circuit-relative imputation.
    """
    grid_df = resolve_starting_grid(gp_data)
    if grid_df.empty:
        return pd.DataFrame()

    pace_df = aggregate_practice_pace(gp_data)

    # Grid-anchored merge (LEFT JOIN) preserves all qualified drivers
    if not pace_df.empty:
        features = grid_df.merge(pace_df, on="driver", how="left")
    else:
        features = grid_df.copy()
        features["best"] = None
        features["avg"] = None
        features["consistency"] = None
        features["is_single_practice"] = False

    # Mark whether authentic practice pace was recorded
    features["has_practice_data"] = features["best"].notna().astype(int)

    # Circuit-relative pace imputation: 102% of slowest observed practice time
    if features["best"].notna().any():
        slowest_observed = float(features["best"].dropna().max())
        float(features["best"].dropna().min())
        imputed_best = slowest_observed * 1.02
        features["best"] = features["best"].fillna(imputed_best)
        features["avg"] = features["avg"].fillna(imputed_best)
    else:
        # If no practice took place, use grid rank ordinal
        features["best"] = features["grid"].astype(float)
        features["avg"] = features["grid"].astype(float)

    # Fill consistency: median of observed consistencies or default neutral spread
    features["consistency"] = pd.to_numeric(features["consistency"], errors="coerce")
    if features["consistency"].notna().any():
        med_consistency = float(features["consistency"].dropna().median())
        features["consistency"] = features["consistency"].fillna(med_consistency)
    else:
        features["consistency"] = features["consistency"].fillna(0.35)

    if "is_single_practice" not in features.columns:
        features["is_single_practice"] = 0
    else:
        features["is_single_practice"] = features["is_single_practice"].astype(bool).astype(int)

    return features.sort_values("grid").reset_index(drop=True)
