"""Automated F1 Historical Data Downloader with rate-limit resiliency."""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
import requests

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEASONS_DIR
from src.data_fetcher import SESSION_TYPES, fetch_gp, get_gp_folder_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("fastf1").propagate = False
logger = logging.getLogger("HistoricalDownloader")

original_get_session = fastf1.get_session
original_load = fastf1.core.Session.load
original_get_event_schedule = fastf1.get_event_schedule

RETRYABLE_ERROR_TERMS: tuple[str, ...] = (
    "429",
    "rate limit",
    "too many requests",
    "quota",
    "500 calls",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection",
)


def handle_rate_limit(func: Any, *args: Any, max_retries: int = 4, **kwargs: Any) -> Any:
    """Execute FastF1 API call with exponential backoff on HTTP 429, rate limits, and transient errors."""
    rate_limit_exc = getattr(fastf1, "RateLimitExceededError", None)
    retryable_types: tuple[type[BaseException], ...] = (
        (rate_limit_exc, requests.exceptions.RequestException)
        if isinstance(rate_limit_exc, type)
        else (requests.exceptions.RequestException,)
    )
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_retryable = isinstance(exc, retryable_types) or any(term in msg for term in RETRYABLE_ERROR_TERMS)
            if is_retryable and attempt < max_retries:
                backoff_seconds = 30 * (2 ** (attempt - 1))
                logger.warning(
                    "Transient/rate-limit error on attempt %d/%d (%s). Backing off for %d seconds...",
                    attempt,
                    max_retries,
                    exc,
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


def _to_utc_datetime(raw_ts: Any) -> datetime | None:
    """Convert a pandas/datetime timestamp to a timezone-aware UTC datetime."""
    if raw_ts is None or pd.isna(raw_ts):
        return None
    ts = pd.to_datetime(raw_ts, errors="coerce")
    if pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def save_season_schedule(year: int, schedule: pd.DataFrame) -> None:
    """Persist season event calendar to data/seasons/<year>/schedule.json."""
    if schedule is None or schedule.empty:
        return

    schedule_path = SEASONS_DIR / str(year) / "schedule.json"
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


def get_scheduled_sessions(event: Any, now_utc: datetime | None = None) -> list[str]:
    """Determine which scheduled sessions are eligible for ingestion up to now_utc."""
    now_utc_aware: datetime | None = None
    if now_utc is not None:
        now_utc_aware = now_utc.replace(tzinfo=UTC) if now_utc.tzinfo is None else now_utc.astimezone(UTC)

    scheduled: list[str] = []
    has_session_metadata = False
    for i in range(1, 6):
        session_name = event.get(f"Session{i}")
        if session_name and session_name in REVERSE_SESSION_TYPES:
            has_session_metadata = True
            session_dt = _to_utc_datetime(event.get(f"Session{i}DateUtc")) or _to_utc_datetime(
                event.get(f"Session{i}Date")
            )
            if now_utc_aware is not None and session_dt is not None and session_dt > now_utc_aware:
                continue
            scheduled.append(REVERSE_SESSION_TYPES[session_name])

    if not has_session_metadata:
        return ["FP1", "FP2", "FP3", "Q", "R"]
    return scheduled


def is_valid_session_file(file_path: Path) -> bool:
    """Verify that a session file exists, is non-empty, and contains populated session data."""
    if not file_path.exists() or file_path.stat().st_size < 100:
        return False
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if not (
                isinstance(data, dict)
                and "session_type" in data
                and bool(data.get("results") or data.get("best_times"))
            ):
                return False
            if data.get("session_type") in {"FP1", "FP2", "FP3"} and data.get("best_times"):
                return bool(data.get("pace_consistency"))
            return True
    except Exception:
        return False


def backfill_missing_pace_consistency(seasons_dir: Path = SEASONS_DIR) -> int:
    """Backfill missing pace_consistency maps in stored practice JSON files using weekend lap spreads."""
    if not seasons_dir.exists():
        return 0

    updated_count = 0
    fp_files = ("fp1.json", "fp2.json", "fp3.json")
    quali_files = ("qualifying.json", "sprint_qualifying.json", "sprint_shootout.json")
    phase_cols = ("q1", "q2", "q3", "sq1", "sq2", "sq3")

    for year_dir in sorted(seasons_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        for gp_dir in sorted(year_dir.iterdir()):
            if not gp_dir.is_dir() or gp_dir.name.startswith("."):
                continue

            fp_payloads: dict[Path, dict[str, Any]] = {}
            driver_laps: dict[str, list[float]] = {}

            for fname in fp_files:
                fpath = gp_dir / fname
                if not fpath.exists():
                    continue
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    fp_payloads[fpath] = data
                    best_times = data.get("best_times", {})
                    if isinstance(best_times, dict):
                        for drv, t in best_times.items():
                            if t is not None and float(t) > 0:
                                driver_laps.setdefault(str(drv), []).append(float(t))

            if not any(not payload.get("pace_consistency") for payload in fp_payloads.values()):
                continue

            for fname in quali_files:
                qpath = gp_dir / fname
                if not qpath.exists():
                    continue
                try:
                    qdata = json.loads(qpath.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(qdata, dict):
                    continue
                best_times = qdata.get("best_times", {})
                if isinstance(best_times, dict):
                    for drv, t in best_times.items():
                        if t is not None and float(t) > 0:
                            driver_laps.setdefault(str(drv), []).append(float(t))
                for row in qdata.get("results", []):
                    drv = row.get("driver")
                    if not drv:
                        continue
                    for col in phase_cols:
                        val = row.get(col)
                        if val is not None and float(val) > 0:
                            driver_laps.setdefault(str(drv), []).append(float(val))

            for fpath, data in fp_payloads.items():
                if data.get("pace_consistency"):
                    continue

                best_times = data.get("best_times", {})
                fp_drivers: list[str] = (
                    [str(d) for d in best_times]
                    if isinstance(best_times, dict) and best_times
                    else [str(r["driver"]) for r in data.get("results", []) if r.get("driver")]
                )
                if not fp_drivers:
                    continue

                raw_stds: dict[str, float] = {}
                for drv in fp_drivers:
                    laps = driver_laps.get(drv, [])
                    if len(laps) >= 2:
                        std_val = float(pd.Series(laps).std())
                        raw_stds[drv] = min(2.50, max(0.15, std_val))

                session_median = float(pd.Series(list(raw_stds.values())).median()) if raw_stds else 0.45
                data["pace_consistency"] = {
                    drv: round(raw_stds.get(drv, session_median), 3) for drv in fp_drivers
                }

                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                updated_count += 1

    return updated_count


def main(argv: list[str] | None = None) -> int:
    """Execute historical data download from specified seasons or 2022 to current."""
    parser = argparse.ArgumentParser(description="Automated F1 Historical Data Downloader with rate-limit resiliency.")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Specific seasons to download (e.g. --years 2025 2026). Default: 2022 to current.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing session files.",
    )
    parser.add_argument(
        "--backfill-consistency",
        action="store_true",
        help="Backfill missing pace_consistency fields in stored practice session JSON files.",
    )
    args = parser.parse_args(argv)

    if args.backfill_consistency:
        updated = backfill_missing_pace_consistency()
        logger.info("Backfilled pace_consistency in %d practice session files.", updated)
        if args.years is None:
            return updated

    now_utc = datetime.now(UTC)
    current_year = now_utc.year
    target_years = sorted(args.years) if args.years else list(range(2022, current_year + 1))

    logger.info("Starting F1 Historical Data Downloader")
    logger.info("Target Seasons: %s", target_years)

    new_downloads_count = 0

    for year in target_years:
        logger.info("Fetching calendar schedule for %d season...", year)
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:
            logger.error("Failed to fetch schedule for %d season: %s", year, exc)
            continue

        save_season_schedule(year, schedule)

        for _, event in schedule.iterrows():
            if event.get("EventFormat") == "testing":
                continue

            round_num = int(event["RoundNumber"])
            if round_num <= 0:
                continue

            gp_name = str(event["EventName"])
            first_session_dt = (
                _to_utc_datetime(event.get("Session1DateUtc"))
                or _to_utc_datetime(event.get("Session1Date"))
                or _to_utc_datetime(event.get("EventDate"))
            )

            if first_session_dt is not None and first_session_dt > now_utc:
                logger.info(
                    "Reached future event: %d Round %d: %s. Completed historical sync for %d.",
                    year,
                    round_num,
                    gp_name,
                    year,
                )
                break

            scheduled_sessions = get_scheduled_sessions(event, now_utc)
            if not scheduled_sessions:
                continue

            gp_folder = get_gp_folder_name(round_num, gp_name)
            gp_path = SEASONS_DIR / str(year) / gp_folder

            sessions_to_fetch: list[str] = []
            for session in scheduled_sessions:
                file_path = gp_path / FILENAME_MAP.get(session, f"{session.lower()}.json")
                if args.force or not is_valid_session_file(file_path):
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
        sys.exit(0 if main() >= 0 else 1)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user.")
        sys.exit(0)
