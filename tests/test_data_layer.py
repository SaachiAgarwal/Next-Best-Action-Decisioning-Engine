"""Tests for the data layer: config wiring + sampled dataset integrity.

The dataset tests read the *sampled* parquet tables in ``data/processed`` that
``python -m src.data.build_dataset`` produces. If those files are not present
(e.g. a fresh checkout where the pipeline has not been run), the dataset tests
are skipped with a clear message rather than re-scanning the 3.5 GB raw CSV.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.data import load as load_mod


# --------------------------------------------------------------------------
# Config smoke tests
# --------------------------------------------------------------------------
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


def test_sampling_config():
    """Sampling constants exist and are the values the whole project relies on."""
    assert config.SAMPLE_CUSTOMERS == 100000
    assert config.SAMPLE_SEED == 42


# --------------------------------------------------------------------------
# Fixtures: sampled parquet tables
# --------------------------------------------------------------------------
def _load_parquet(name):
    path = config.PROCESSED_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not built yet — run `python -m src.data.build_dataset`")
    return pd.read_parquet(path, engine="pyarrow")


@pytest.fixture(scope="module")
def transactions():
    return _load_parquet("transactions.parquet")


@pytest.fixture(scope="module")
def articles():
    return _load_parquet("articles.parquet")


@pytest.fixture(scope="module")
def customers():
    return _load_parquet("customers.parquet")


# --------------------------------------------------------------------------
# Data-layer tests
# --------------------------------------------------------------------------
def test_article_loader_keeps_leading_zeros():
    """load_articles() reads article_id as string, preserving leading zeros."""
    df = load_mod.load_articles()
    assert str(df["article_id"].dtype) == "string"
    # At least one real H&M article_id starts with a leading zero.
    assert df["article_id"].str.startswith("0").any()


def test_ids_are_string_dtype(transactions):
    """article_id and customer_id are string dtype (never int) in the dataset."""
    assert str(transactions["article_id"].dtype) == "string"
    assert str(transactions["customer_id"].dtype) == "string"
    # Leading zeros survived into the working dataset.
    assert transactions["article_id"].str.startswith("0").any()


def test_no_duplicate_primary_keys(customers, articles):
    """Sampled customers/articles have unique primary keys."""
    assert customers["customer_id"].duplicated().sum() == 0
    assert articles["article_id"].duplicated().sum() == 0


def test_parquet_roundtrip_preserves_article_id_string():
    """Re-reading transactions.parquet keeps article_id as string dtype."""
    path = config.PROCESSED_DIR / "transactions.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet — run `python -m src.data.build_dataset`")
    rt = pd.read_parquet(path, engine="pyarrow")
    assert str(rt["article_id"].dtype) == "string"


def test_transaction_date_range_nonempty(transactions):
    """Transactions have a real, non-empty date range."""
    dmin = transactions["t_dat"].min()
    dmax = transactions["t_dat"].max()
    assert pd.notna(dmin) and pd.notna(dmax)
    assert dmin <= dmax


def test_sampled_customer_count_within_cap(customers):
    """Sampled customer count never exceeds SAMPLE_CUSTOMERS.

    It equals SAMPLE_CUSTOMERS unless the full dataset has fewer customers.
    """
    assert customers["customer_id"].nunique() <= config.SAMPLE_CUSTOMERS
