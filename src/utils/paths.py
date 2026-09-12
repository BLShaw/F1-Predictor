"""Path resolution and filesystem safety utilities."""

import gc
import logging
import os
import shutil
import time
from pathlib import Path

from src.config import BASE_DIR, CACHE_DIR, DATA_DIR, MODELS_DIR, SEASONS_DIR

__all__ = [
    "BASE_DIR",
    "CACHE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "SEASONS_DIR",
    "clear_fastf1_cache",
    "ensure_directories",
]

logger = logging.getLogger(__name__)


def ensure_directories(directories: list[Path] | None = None) -> None:
    """Ensure all core project directories or provided directories exist."""
    dirs_to_check = directories if directories is not None else [DATA_DIR, SEASONS_DIR, MODELS_DIR, CACHE_DIR]
    for directory in dirs_to_check:
        directory.mkdir(parents=True, exist_ok=True)


def clear_fastf1_cache(cache_path: Path = CACHE_DIR, retries: int = 3, delay: float = 0.5) -> tuple[bool, str]:
    """Safely clear FastF1 cache directory with Windows file lock handling.

    Returns:
        tuple of (success: bool, message: str)
    """
    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)
        return True, "Cache directory is already empty."

    # Force garbage collection to release any dangling SQLite/file handles
    gc.collect()

    errors: list[str] = []

    # Attempt full directory removal with retries
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            return True, "Cache successfully cleared."
        except PermissionError as exc:
            logger.warning("Cache file locked on attempt %d: %s", attempt, exc)
            time.sleep(delay)
        except OSError as exc:
            logger.warning("OS error clearing cache on attempt %d: %s", attempt, exc)
            time.sleep(delay)

    # Fallback: remove unlocked files individually
    cleared_count = 0
    locked_count = 0
    for root, _, files in os.walk(cache_path):
        for file in files:
            file_path = Path(root) / file
            try:
                file_path.unlink(missing_ok=True)
                cleared_count += 1
            except (PermissionError, OSError) as exc:
                locked_count += 1
                errors.append(f"{file}: {exc}")

    if locked_count > 0:
        return False, f"Partially cleared ({cleared_count} files removed, {locked_count} locked)."
    return True, f"Cache cleared ({cleared_count} files removed)."
