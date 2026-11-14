"""F1 Data Loader Module.

Framework-agnostic functions to load and transform Grand Prix session data from disk.
"""

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import SEASONS_DIR, get_team_for_driver, get_track_characteristics

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


def _extract_classified_finishes(race_session: dict[str, Any] | None) -> dict[str, int]:
    """Extract classified finishing positions from a race session dictionary."""
    if not race_session or not isinstance(race_session, dict):
        return {}

    results = race_session.get("results")
    if not isinstance(results, list):
        return {}

    classified: dict[str, int] = {}
    for entry in results:
        driver = entry.get("driver")
        pos = entry.get("position")
        status = str(entry.get("status", "")).strip()
        if not driver or pos is None:
            continue
        try:
            pos_int = int(float(pos))
        except (ValueError, TypeError):
            continue

        if status == "Finished" or status.startswith("+") or "Lap" in status:
            classified[str(driver)] = pos_int
    return classified


def compute_rolling_form(
    target_year: int | None = None,
    target_round: int | None = None,
    seasons_path: Path = SEASONS_DIR,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute pre-race EWMA form for drivers and constructors strictly prior to target event."""
    if not seasons_path.exists():
        return {}, {}

    driver_hist: dict[str, list[float]] = {}
    team_hist: dict[str, list[float]] = {}

    season_dirs = sorted(
        (d for d in seasons_path.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    for year_dir in season_dirs:
        season_year = int(year_dir.name)
        if target_year is not None and season_year > target_year:
            continue

        gp_entries: list[tuple[int, Path]] = []
        for gp_dir in year_dir.iterdir():
            if not gp_dir.is_dir() or gp_dir.name.startswith("."):
                continue
            prefix = gp_dir.name.split("_", 1)[0]
            if prefix.isdigit():
                gp_entries.append((int(prefix), gp_dir))

        for round_num, gp_dir in sorted(gp_entries, key=lambda x: x[0]):
            if (
                target_year is not None
                and season_year == target_year
                and target_round is not None
                and round_num >= target_round
            ):
                continue

            race_session = load_session(season_year, gp_dir.name, "race", seasons_path)
            if not race_session:
                continue

            classified = _extract_classified_finishes(race_session)
            if not classified:
                continue

            raw_results = race_session.get("results", [])
            n_cars = max(len(raw_results), max(classified.values(), default=1), 1)
            driver_teams = {
                str(entry.get("driver", "")): str(entry.get("team", "")).strip()
                for entry in raw_results
                if entry.get("driver")
            }

            for drv, finish_pos in classified.items():
                norm_finish = (finish_pos - 1.0) / max(n_cars - 1, 1)
                driver_hist.setdefault(drv, []).append(norm_finish)
                team_name = driver_teams.get(drv) or get_team_for_driver(drv)
                if team_name:
                    team_hist.setdefault(team_name, []).append(norm_finish)

    driver_form = {
        drv: float(pd.Series(hist[-5:]).ewm(span=3).mean().iloc[-1])
        for drv, hist in driver_hist.items()
        if hist
    }
    constructor_form = {
        tm: float(pd.Series(hist[-10:]).ewm(span=5).mean().iloc[-1])
        for tm, hist in team_hist.items()
        if hist
    }
    return driver_form, constructor_form


def extract_qualifying_gaps(gp_data: dict[str, Any], grid_df: pd.DataFrame) -> pd.Series:
    """Compute normalized qualifying lap time delta to pole per grid driver."""
    n_drivers = max(len(grid_df), 1)
    grid_norm = (pd.to_numeric(grid_df["grid"], errors="coerce").fillna(n_drivers) - 1.0) / max(n_drivers - 1, 1)

    sessions = gp_data.get("sessions", {})
    q_session = (
        sessions.get("qualifying")
        or sessions.get("sprint_qualifying")
        or sessions.get("sprint_shootout")
    )
    if not q_session:
        return grid_norm

    q_df = get_qualifying_results(q_session)
    q_times: dict[str, float] = {}
    if not q_df.empty and "driver" in q_df.columns:
        q_cols = [c for c in ["q3", "q2", "q1", "sq3", "sq2", "sq1"] if c in q_df.columns]
        for _, row in q_df.iterrows():
            drv = str(row["driver"])
            vals = [float(row[c]) for c in q_cols if pd.notna(row.get(c)) and float(row[c]) > 0]
            if vals:
                q_times[drv] = min(vals)

    raw_best = q_session.get("best_times", {})
    if isinstance(raw_best, dict):
        for drv, t in raw_best.items():
            if str(drv) not in q_times and pd.notna(t) and float(t) > 0:
                q_times[str(drv)] = float(t)

    if not q_times:
        return grid_norm

    q_min = min(q_times.values())
    q_max = max(q_times.values())
    q_range = q_max - q_min if q_max > q_min else 1.0

    gaps: list[float] = []
    for idx, row in grid_df.iterrows():
        drv = str(row["driver"])
        q_time = q_times.get(drv)
        if q_time is not None:
            gaps.append(float(min(1.5, max(0.0, (q_time - q_min) / q_range))))
        else:
            gaps.append(float(grid_norm.loc[idx]))
    return pd.Series(gaps, index=grid_df.index, dtype=float)


def _resolve_gp_context(gp_data: dict[str, Any]) -> tuple[str, str, int | None, int | None]:
    """Extract track name, GP name, season year, and round number from gp_data."""
    sessions = gp_data.get("sessions", {})
    track_name = ""
    gp_name = str(gp_data.get("name", "") or gp_data.get("gp_folder", ""))
    round_num: int | None = gp_data.get("round")

    for key in ["race", "qualifying", "fp1", "fp2", "fp3", "sprint", "sprint_qualifying", "sprint_shootout"]:
        session = sessions.get(key)
        if isinstance(session, dict):
            if not track_name and session.get("track"):
                track_name = str(session["track"])
            if session.get("gp_name"):
                gp_name = str(session["gp_name"])
            if round_num is None and session.get("round") is not None:
                with contextlib.suppress(ValueError, TypeError):
                    round_num = int(session["round"])

    if round_num is None and gp_data.get("gp_folder"):
        prefix = str(gp_data["gp_folder"]).split("_", 1)[0]
        if prefix.isdigit():
            round_num = int(prefix)

    year_val = gp_data.get("year")
    try:
        year_int = int(year_val) if year_val is not None else None
    except (ValueError, TypeError):
        year_int = None

    return track_name, gp_name, year_int, round_num


def extract_weekend_rainfall(gp_data: dict[str, Any]) -> bool:
    """Detect whether rainfall was recorded for the race or weekend sessions."""
    sessions = gp_data.get("sessions", {})
    if not isinstance(sessions, dict):
        return False

    race_session = sessions.get("race")
    if isinstance(race_session, dict):
        race_weather = race_session.get("weather")
        if isinstance(race_weather, dict) and "rainfall" in race_weather:
            return bool(race_weather.get("rainfall", False))

    for key in ["sprint", "qualifying", "sprint_qualifying", "sprint_shootout", "fp3", "fp2", "fp1"]:
        session = sessions.get(key)
        if isinstance(session, dict):
            weather = session.get("weather")
            if isinstance(weather, dict) and bool(weather.get("rainfall", False)):
                return True

    return False


def build_weekend_features(
    gp_data: dict[str, Any],
    seasons_path: Path = SEASONS_DIR,
    rolling_form: tuple[dict[str, float], dict[str, float]] | None = None,
) -> pd.DataFrame:
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

    sessions = gp_data.get("sessions", {})
    is_sprint_weekend = bool(
        sessions.get("sprint") or sessions.get("sprint_qualifying") or sessions.get("sprint_shootout")
    )
    features["is_sprint"] = int(is_sprint_weekend)

    features = features.sort_values("grid").reset_index(drop=True)
    n_drivers = max(len(features), 1)
    grid_norm_series = (features["grid"].astype(float) - 1.0) / max(n_drivers - 1, 1)

    features["quali_gap_norm"] = extract_qualifying_gaps(gp_data, features)

    track_name, gp_name, year_int, round_num = _resolve_gp_context(gp_data)
    track_meta = get_track_characteristics(track_name=track_name, gp_name=gp_name)
    features["overtaking_difficulty"] = float(track_meta["overtaking_difficulty_score"])
    features["is_street_circuit"] = int(track_meta["is_street_circuit"])
    features["rainfall"] = int(extract_weekend_rainfall(gp_data))

    if rolling_form is not None:
        driver_form_map, team_form_map = rolling_form
    elif year_int is not None and round_num is not None:
        driver_form_map, team_form_map = compute_rolling_form(year_int, round_num, seasons_path)
    else:
        driver_form_map, team_form_map = {}, {}

    driver_forms: list[float] = []
    constructor_forms: list[float] = []
    for idx, row in features.iterrows():
        drv = str(row["driver"])
        fallback_val = float(grid_norm_series.loc[idx])
        driver_forms.append(float(driver_form_map.get(drv, fallback_val)))

        team_name = str(row.get("team", "")).strip() or get_team_for_driver(drv)
        constructor_forms.append(float(team_form_map.get(team_name, fallback_val)))

    features["driver_form"] = driver_forms
    features["constructor_form"] = constructor_forms

    return features
