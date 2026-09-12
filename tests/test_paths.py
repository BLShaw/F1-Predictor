"""Tests for path resolution and cache utilities."""

from pathlib import Path

from src.utils.paths import (
    BASE_DIR,
    CACHE_DIR,
    DATA_DIR,
    MODELS_DIR,
    SEASONS_DIR,
    clear_fastf1_cache,
    ensure_directories,
)


def test_paths_are_absolute() -> None:
    """Verify that all core project paths resolve to absolute paths."""
    assert BASE_DIR.is_absolute()
    assert DATA_DIR.is_absolute()
    assert SEASONS_DIR.is_absolute()
    assert MODELS_DIR.is_absolute()
    assert CACHE_DIR.is_absolute()


def test_ensure_directories(tmp_path: Path) -> None:
    """Verify that directory creation handles existing and new paths."""
    test_dir = tmp_path / "test_data" / "models"
    assert not test_dir.exists()
    ensure_directories([test_dir])
    assert test_dir.exists()
    # Idempotent call
    ensure_directories([test_dir])
    assert test_dir.exists()


def test_clear_fastf1_cache(tmp_path: Path) -> None:
    """Verify cache clearing deletes contents without raising on missing paths."""
    test_cache = tmp_path / "cache_test"
    test_cache.mkdir(parents=True)
    sample_file = test_cache / "sample.cache"
    sample_file.write_text("telemetry")

    assert sample_file.exists()
    success, msg = clear_fastf1_cache(test_cache)
    assert success is True
    assert test_cache.exists()
    assert not sample_file.exists()

    # Calling on non-existent path should succeed gracefully
    non_existent = tmp_path / "does_not_exist"
    success_none, _ = clear_fastf1_cache(non_existent)
    assert success_none is True
