"""Smoke tests for the data layer.

Keeps the suite green from day one and verifies the config module wires
up project paths correctly.
"""

from pathlib import Path

from src import config


def test_config_paths_import():
    """Config exposes the expected path constants as pathlib.Path objects."""
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.RAW_DIR, Path)
    assert isinstance(config.PROCESSED_DIR, Path)
    assert isinstance(config.REPORTS_DIR, Path)


def test_config_paths_are_rooted():
    """Data and report dirs live under the project root."""
    assert config.RAW_DIR == config.PROJECT_ROOT / "data" / "raw"
    assert config.PROCESSED_DIR == config.PROJECT_ROOT / "data" / "processed"
    assert config.REPORTS_DIR == config.PROJECT_ROOT / "reports"


def test_placeholder_constants():
    """Placeholder constants have their expected defaults."""
    assert config.CUTOFF_DATE is None
    assert config.ACTION_GRANULARITY == "product_group_name"
