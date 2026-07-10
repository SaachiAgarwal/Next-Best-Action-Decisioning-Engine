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


@pytest.fixture(scope="module")
def actions():
    return _load_parquet("actions.parquet")


@pytest.fixture(scope="module")
def article_action_map():
    return _load_parquet("article_action_map.parquet")


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


# --------------------------------------------------------------------------
# Cleaning + action-space tests (Day 3)
# --------------------------------------------------------------------------
def test_no_transaction_has_nonpositive_price(transactions):
    """Cleaning removes impossible transactions: no price <= 0 remains."""
    assert (transactions["price"] <= 0).sum() == 0


def test_article_id_string_after_cleaning(transactions, articles, article_action_map):
    """article_id stays string dtype through cleaning and the action map."""
    assert str(transactions["article_id"].dtype) == "string"
    assert str(articles["article_id"].dtype) == "string"
    assert str(article_action_map["article_id"].dtype) == "string"


def test_every_article_maps_to_exactly_one_action(articles, article_action_map):
    """Each cleaned article maps to exactly one action (no dup / no missing)."""
    # No article_id appears more than once in the map.
    assert article_action_map["article_id"].duplicated().sum() == 0
    # Every article in the cleaned articles table is present in the map.
    mapped = set(article_action_map["article_id"])
    missing = set(articles["article_id"]) - mapped
    assert not missing, f"{len(missing)} articles have no action mapping"
    # Every mapped article has a non-null action_id.
    assert article_action_map["action_id"].notna().all()


def test_action_id_unique_per_product_type(actions, article_action_map):
    """action_id is a stable 1:1 id per product_type_name."""
    # In the actions table: unique action_id and unique product_type_name.
    assert actions["action_id"].duplicated().sum() == 0
    assert actions["product_type_name"].duplicated().sum() == 0
    # In the map: exactly one action_id per product_type_name.
    per_type = article_action_map.groupby("product_type_name")["action_id"].nunique()
    assert (per_type == 1).all()


def test_map_covers_every_transaction_article(transactions, article_action_map):
    """The article->action map covers every article_id present in transactions."""
    tx_articles = set(transactions["article_id"].unique())
    mapped = set(article_action_map["article_id"])
    uncovered = tx_articles - mapped
    assert not uncovered, f"{len(uncovered)} transaction articles are unmapped"
