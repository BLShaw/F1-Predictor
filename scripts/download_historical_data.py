"""Automated F1 Historical Data Downloader with rate-limit resiliency."""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEASONS_DIR
from src.data_fetcher import SESSION_TYPES, fetch_gp, get_gp_folder_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HistoricalDownloader")

original_get_session = fastf1.get_session
original_load = fastf1.core.Session.load
original_get_event_schedule = fastf1.get_event_schedule


def handle_rate_limit(func: Any, *args: Any, max_retries: int = 4, **kwargs: Any) -> Any:
    """Execute FastF1 API call with exponential backoff on HTTP 429 / rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = any(
                term in msg for term in ["429", "rate limit", "too many requests", "quota", "500 calls"]
            )
            if is_rate_limit and attempt < max_retries:
                backoff_seconds = 30 * (2 ** (attempt - 1))
                logger.warning(
                    "API rate limit detected on attempt %d/%d. Backing off for %d seconds...",
                    attempt,
                    max_retries,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
            else:
                raise


def patched_get_session(*args: Any, **kwargs: Any) -> Any:
    return handle_rate_limit(original_get_session, *args, **kwargs)


def patched_load(self: Any, *args: Any, **kwargs: Any) -> Any:
    return handle_rate_limit(original_load, self, *args, **kwargs)


def patched_get_event_schedule(*args: Any, **kwargs: Any) -> Any:
    return handle_rate_limit(original_get_event_schedule, *args, **kwargs)


# Apply rate-limit resilient wrappers
fastf1.get_session = patched_get_session
fastf1.core.Session.load = patched_load
fastf1.get_event_schedule = patched_get_event_schedule

REVERSE_SESSION_TYPES = {v: k for k, v in SESSION_TYPES.items()}
FILENAME_MAP = {
    "FP1": "fp1.json",
    "FP2": "fp2.json",
    "FP3": "fp3.json",
    "Q": "qualifying.json",
    "SQ": "sprint_qualifying.json",
    "SS": "sprint_shootout.json",
    "S": "sprint.json",
    "R": "race.json",
}


def get_scheduled_sessions(event: Any) -> list[str]:
    """Determine which sessions are scheduled for this Grand Prix event."""
    scheduled: list[str] = []
    for i in range(1, 6):
        session_name = event.get(f"Session{i}")
        if session_name and session_name in REVERSE_SESSION_TYPES:
            scheduled.append(REVERSE_SESSION_TYPES[session_name])
    return scheduled


def is_valid_session_file(file_path: Path) -> bool:
    """Verify that a session file exists, is non-empty, and contains valid JSON."""
    if not file_path.exists() or file_path.stat().st_size < 100:
        return False
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            return isinstance(data, dict) and "session_type" in data
    except Exception:
        return False


def main() -> int:
    """Execute historical data download from 2022 to current season."""
    logger.info("Starting F1 Historical Data Downloader")
    logger.info("Target: 2022 Season -> Present (Completed Races)")

    current_year = datetime.now().year
    new_downloads_count = 0

    for year in range(2022, current_year + 1):
        logger.info("Fetching calendar schedule for %d season...", year)
        schedule = fastf1.get_event_schedule(year)

        for _, event in schedule.iterrows():
            if event.get("EventFormat") == "testing":
                continue

            round_num = int(event["RoundNumber"])
            gp_name = str(event["EventName"])
            event_date = pd.to_datetime(event["EventDate"])
            naive_event_date = event_date.tz_localize(None) if event_date.tzinfo else event_date

            if naive_event_date > datetime.now():
                logger.info(
                    "Reached future event: %d Round %d: %s. Completed historical sync up to today.",
                    year,
                    round_num,
                    gp_name,
                )
                return new_downloads_count

            scheduled_sessions = get_scheduled_sessions(event)
            if not scheduled_sessions:
                scheduled_sessions = ["FP1", "FP2", "FP3", "Q", "R"]

            gp_folder = get_gp_folder_name(round_num, gp_name)
            gp_path = SEASONS_DIR / str(year) / gp_folder

            sessions_to_fetch: list[str] = []
            for session in scheduled_sessions:
                file_path = gp_path / FILENAME_MAP.get(session, f"{session.lower()}.json")
                if not is_valid_session_file(file_path):
                    sessions_to_fetch.append(session)

            if not sessions_to_fetch:
                continue

            logger.info(
                "Downloading %d Round %d: %s (Sessions: %s)...", year, round_num, gp_name, ", ".join(sessions_to_fetch)
            )
            results = fetch_gp(year, round_num, sessions_to_fetch)
            success_count = sum(1 for v in results.values() if v)
            new_downloads_count += success_count
            time.sleep(1.5)

    logger.info("Historical download complete. Ingested %d new sessions.", new_downloads_count)
    return new_downloads_count


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user.")
        sys.exit(0)
