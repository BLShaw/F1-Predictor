"""F1 Data Fetcher - FastF1 Integration Module.

Extracts telemetry, weather, and classifications from FastF1 with accurate flying-lap filtering.
"""

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd

from src.config import CACHE_DIR, SEASONS_DIR

logger = logging.getLogger(__name__)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

DATA_DIR: Path = SEASONS_DIR

SESSION_TYPES: dict[str, str] = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "SQ": "Sprint Qualifying",
    "SS": "Sprint Shootout",
    "S": "Sprint",
    "R": "Race",
}


def get_gp_folder_name(round_num: int, gp_name: str) -> str:
    """Create standardized Grand Prix folder name."""
    clean_name = gp_name.replace(" ", "_").replace("Grand_Prix", "GP")
    return f"{round_num:02d}_{clean_name}"


def ensure_gp_folder(year: int, gp_name: str, round_num: int) -> Path:
    """Ensure Grand Prix directory exists and return resolved path."""
    folder_name = get_gp_folder_name(round_num, gp_name)
    gp_path = DATA_DIR / str(year) / folder_name
    gp_path.mkdir(parents=True, exist_ok=True)
    return gp_path


def extract_results(results: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract driver classifications and session results from FastF1 results DataFrame."""
    result_data: list[dict[str, Any]] = []
    if results is None or results.empty:
        return result_data

    for _, row in results.iterrows():
        entry: dict[str, Any] = {
            "position": int(row.get("Position", 0)) if pd.notna(row.get("Position")) else None,
            "driver": str(row.get("Abbreviation", "")),
            "driver_number": int(row.get("DriverNumber", 0)) if pd.notna(row.get("DriverNumber")) else None,
            "team": str(row.get("TeamName", "")),
            "grid_position": int(row.get("GridPosition", 0)) if pd.notna(row.get("GridPosition")) else None,
            "status": str(row.get("Status", "")),
            "points": float(row.get("Points", 0)) if pd.notna(row.get("Points")) else 0.0,
        }
        for q in ["Q1", "Q2", "Q3", "SQ1", "SQ2", "SQ3"]:
            if q in row and pd.notna(row.get(q)):
                with contextlib.suppress(AttributeError, TypeError):
                    entry[q.lower()] = row.get(q).total_seconds()

        if "Time" in row and pd.notna(row.get("Time")):
            try:
                entry["time"] = row.get("Time").total_seconds()
            except (AttributeError, TypeError):
                entry["time"] = str(row.get("Time"))

        result_data.append(entry)

    return result_data


def extract_weather(session: Any) -> dict[str, Any]:
    """Extract aggregated weather conditions from session."""
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            return {
                "air_temp": float(weather["AirTemp"].mean()) if "AirTemp" in weather else None,
                "track_temp": float(weather["TrackTemp"].mean()) if "TrackTemp" in weather else None,
                "humidity": float(weather["Humidity"].mean()) if "Humidity" in weather else None,
                "rainfall": bool(weather["Rainfall"].any()) if "Rainfall" in weather else False,
                "wind_speed": float(weather["WindSpeed"].mean()) if "WindSpeed" in weather else None,
            }
    except (AttributeError, KeyError, TypeError) as exc:
        logger.warning("Could not extract weather data: %s", exc)
    return {}


def filter_flying_laps(driver_laps: pd.DataFrame) -> list[float]:
    """Filter driver laps to isolate legitimate flying laps excluding in/out/deleted laps.

    Applies pick_quicklaps(), pick_wo_box(), and checks IsAccurate == True.
    """
    if driver_laps.empty or "LapTime" not in driver_laps.columns:
        return []

    candidate_laps = driver_laps

    # Exclude pit-in and pit-out laps using FastF1 built-ins if possible
    try:
        if hasattr(candidate_laps, "pick_quicklaps") and hasattr(candidate_laps, "pick_wo_box"):
            filtered = candidate_laps.pick_quicklaps().pick_wo_box()
            if not filtered.empty:
                candidate_laps = filtered
    except Exception as exc:
        logger.debug("FastF1 quicklap filtering fallback: %s", exc)

    # Exclude deleted track-limit laps when IsAccurate flag is available
    if "IsAccurate" in candidate_laps.columns:
        accurate = candidate_laps[candidate_laps["IsAccurate"]]
        if not accurate.empty:
            candidate_laps = accurate

    # Extract valid seconds (> 20.0s floor to eliminate zero/corrupt records)
    valid_seconds: list[float] = []
    for lap_time in candidate_laps["LapTime"].dropna():
        try:
            sec = lap_time.total_seconds()
            if sec > 20.0:
                valid_seconds.append(sec)
        except (AttributeError, TypeError):
            continue

    # Fallback to general LapTime if strict filtering removed all laps
    if not valid_seconds:
        for lap_time in driver_laps["LapTime"].dropna():
            try:
                sec = lap_time.total_seconds()
                if sec > 20.0:
                    valid_seconds.append(sec)
            except (AttributeError, TypeError):
                continue

    return valid_seconds


def fetch_session(year: int, gp: str | int, session_type: str) -> tuple[dict[str, Any], int, str] | None:
    """Fetch a single session from FastF1 and return structured session dictionary.

    Args:
        year: Season year (e.g., 2024)
        gp: GP name or round number
        session_type: One of FP1, FP2, FP3, Q, SQ, SS, S, R

    Returns:
        tuple of (session_data, round_num, gp_name) or None if unavailable
    """
    try:
        logger.info("Fetching %s for %s %d...", session_type, gp, year)
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=False, messages=False)

        event = session.event
        round_num = int(event["RoundNumber"])
        gp_name = str(event["EventName"])

        session_data: dict[str, Any] = {
            "session_type": session_type,
            "session_name": SESSION_TYPES.get(session_type, session_type),
            "year": year,
            "round": round_num,
            "gp_name": gp_name,
            "date": str(session.date.date()) if session.date else None,
            "track": str(event.get("Location", "")),
            "country": str(event.get("Country", "")),
        }

        # Extract representative best lap times and stint consistency
        if hasattr(session, "laps") and session.laps is not None and not session.laps.empty:
            best_times: dict[str, float] = {}
            pace_consistency: dict[str, float | None] = {}

            for driver in session.laps["Driver"].dropna().unique():
                driver_str = str(driver)
                driver_laps = session.laps[session.laps["Driver"] == driver]
                valid_laps = filter_flying_laps(driver_laps)

                if valid_laps:
                    best_times[driver_str] = round(float(min(valid_laps)), 3)
                    # Compute stint consistency standard deviation if at least 3 flying laps exist
                    if len(valid_laps) >= 3:
                        pace_consistency[driver_str] = round(float(pd.Series(valid_laps).std()), 3)
                    else:
                        pace_consistency[driver_str] = None

            session_data["best_times"] = best_times
            session_data["pace_consistency"] = pace_consistency

        if hasattr(session, "results") and session.results is not None and not session.results.empty:
            session_data["results"] = extract_results(session.results)

        session_data["weather"] = extract_weather(session)
        session_data["fetched_at"] = datetime.now().isoformat()

        logger.info("Successfully fetched %s for %s %d", session_type, gp_name, year)
        return session_data, round_num, gp_name

    except Exception as exc:
        logger.error("Error fetching %s for %s %d: %s", session_type, gp, year, exc)
        return None


def save_session_json(session_data: dict[str, Any], year: int, round_num: int, gp_name: str, session_type: str) -> Path:
    """Save session data to standardized JSON file."""
    gp_path = ensure_gp_folder(year, gp_name, round_num)

    filename_map = {
        "FP1": "fp1.json",
        "FP2": "fp2.json",
        "FP3": "fp3.json",
        "Q": "qualifying.json",
        "SQ": "sprint_qualifying.json",
        "SS": "sprint_shootout.json",
        "S": "sprint.json",
        "R": "race.json",
    }

    filename = filename_map.get(session_type, f"{session_type.lower()}.json")
    filepath = gp_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, default=str)

    logger.info("Saved session data to %s", filepath)
    update_metadata(gp_path, session_type)
    return filepath


def update_metadata(gp_path: Path, session_type: str) -> None:
    """Update Grand Prix metadata JSON file with session completion status."""
    metadata_path = gp_path / "metadata.json"
    metadata: dict[str, Any] = {"sessions": {}}

    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {"sessions": {}}

    metadata.setdefault("sessions", {})[session_type] = {
        "status": "complete",
        "fetched_at": datetime.now().isoformat(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def fetch_gp(year: int, gp: str | int, sessions: list[str] | None = None) -> dict[str, bool]:
    """Fetch specified sessions for a Grand Prix."""
    if sessions is None:
        sessions = ["FP1", "FP2", "FP3", "Q", "R"]

    results: dict[str, bool] = {}
    for session_type in sessions:
        result = fetch_session(year, gp, session_type)
        if result:
            session_data, round_num, gp_name = result
            save_session_json(session_data, year, round_num, gp_name, session_type)
            results[session_type] = True
        else:
            results[session_type] = False

    return results


def fetch_season(year: int, sessions: list[str] | None = None) -> None:
    """Fetch all Grand Prix events for an entire season."""
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule_path = DATA_DIR / str(year) / "schedule.json"
        schedule_path.parent.mkdir(parents=True, exist_ok=True)

        schedule_data: list[dict[str, Any]] = []
        for _, event in schedule.iterrows():
            if event.get("EventFormat", "") not in ["testing"]:
                schedule_data.append(
                    {
                        "round": int(event["RoundNumber"]),
                        "name": str(event["EventName"]),
                        "location": str(event.get("Location", "")),
                        "country": str(event.get("Country", "")),
                        "date": str(event.get("EventDate", "")),
                        "format": str(event.get("EventFormat", "conventional")),
                    }
                )

        with open(schedule_path, "w", encoding="utf-8") as f:
            json.dump(schedule_data, f, indent=2)

        for event in schedule_data:
            if event["round"] > 0:
                gp_sessions = sessions or ["FP1", "FP2", "FP3", "Q", "R"]
                if event["format"] == "sprint_shootout":
                    gp_sessions = ["FP1", "Q", "SS", "S", "R"]
                elif event["format"] == "sprint":
                    gp_sessions = ["FP1", "SQ", "S", "FP2", "Q", "R"]

                fetch_gp(year, event["round"], gp_sessions)

    except Exception as exc:
        logger.error("Error fetching season %d: %s", year, exc)


def get_available_sessions(year: int, gp_folder: str) -> dict[str, bool]:
    """Check which session JSON files exist for a Grand Prix."""
    gp_path = DATA_DIR / str(year) / gp_folder
    session_files = [
        "fp1.json",
        "fp2.json",
        "fp3.json",
        "qualifying.json",
        "sprint_qualifying.json",
        "sprint_shootout.json",
        "sprint.json",
        "race.json",
    ]
    return {sf.replace(".json", ""): (gp_path / sf).exists() for sf in session_files}


def update_latest_session(year: int | None = None) -> str | None:
    """Fetch the most recent completed session."""
    if year is None:
        year = datetime.now().year

    try:
        schedule = fastf1.get_event_schedule(year)
        now = datetime.now()

        for _, event in schedule.iterrows():
            event_date = pd.to_datetime(event.get("Session5Date"))
            if event_date and event_date < now:
                round_num = int(event["RoundNumber"])
                for session_type in ["R", "Q", "FP3", "FP2", "FP1"]:
                    result = fetch_session(year, round_num, session_type)
                    if result:
                        session_data, r_num, gp_name = result
                        save_session_json(session_data, year, r_num, gp_name, session_type)
                        return f"Updated {session_type} for {gp_name}"

        return "No new sessions available"
    except Exception as exc:
        logger.error("Error updating latest session: %s", exc)
        return None


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for targeted telemetry fetching via FastF1."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch F1 telemetry data via FastF1.")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Championship season year")
    parser.add_argument("--gp", type=str, default=None, help="Grand Prix name or substring (e.g. Monaco, Bahrain)")
    parser.add_argument("--round", type=int, default=None, help="Round number")
    parser.add_argument("--sessions", nargs="+", default=None, help="Session codes (e.g. FP1 FP2 Q R)")

    args = parser.parse_args(argv)

    if args.round is not None:
        fetch_gp(args.year, args.round, args.sessions)
    elif args.gp is not None:
        schedule = fastf1.get_event_schedule(args.year)
        matched_round = None
        for _, event in schedule.iterrows():
            event_name = str(event.get("EventName", "")).lower()
            if args.gp.lower() in event_name:
                matched_round = int(event["RoundNumber"])
                logger.info("Matched %s to %s (Round %d)", args.gp, event["EventName"], matched_round)
                break
        if matched_round is not None:
            fetch_gp(args.year, matched_round, args.sessions)
        else:
            logger.error("No Grand Prix matching '%s' found for season %d", args.gp, args.year)
    else:
        fetch_season(args.year, args.sessions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

