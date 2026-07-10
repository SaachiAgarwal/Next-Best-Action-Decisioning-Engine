"""Data loading, sampling, and validation layer for the NBA Decisioning Engine.

Reads the raw H&M source files (transactions, articles, customers) from
``data/raw`` into pandas DataFrames with explicit, correct dtypes, then builds
the *sampled* working dataset that the entire project runs on.

CRITICAL — ``article_id`` and ``customer_id`` MUST stay strings.
    These id columns are loaded as strings and must NEVER be read as integers.
    ``article_id`` carries leading zeros (e.g. ``"0663713001"``); loading it as
    an int silently drops those zeros and turns it into ``663713001``. That
    quietly breaks every join between transactions, articles, and customers —
    the failure is invisible until joins start returning empty. We therefore
    pin the string dtype here at the source, and assert it survives the parquet
    round-trip downstream.

Public API
    load_transactions()  -- transactions with correct dtypes (optionally filtered)
    load_articles()      -- articles master with article_id as string
    load_customers()     -- customers master with customer_id as string
    load_all()           -- all three, with the customer sample applied
    validate()           -- print a validation report over the sampled tables
    optimize_memory()    -- downcast price/sales_channel to compact dtypes
    save_parquet()       -- write sampled tables to data/processed as parquet
"""

from __future__ import annotations

import pandas as pd

from src import config

# --- File locations --------------------------------------------------------
TRANSACTIONS_PATH = config.RAW_DIR / "transactions_train.csv"
ARTICLES_PATH = config.RAW_DIR / "articles.csv"
CUSTOMERS_PATH = config.RAW_DIR / "customers.csv"

# --- Column dtypes (pinned at read time) -----------------------------------
# IDs are strings on purpose — see module docstring.
TRANSACTION_DTYPES = {
    "customer_id": "string",
    "article_id": "string",
    "price": "float32",
    "sales_channel_id": "category",
}
ARTICLE_DTYPES = {"article_id": "string"}
CUSTOMER_DTYPES = {"customer_id": "string"}

# Read the transactions CSV in chunks so peak memory stays bounded on the
# 3.5 GB / ~31.8M-row file (the machine has 16 GB of RAM).
_CHUNK_SIZE = 3_000_000


def _require(path):
    """Raise a clear error (never fabricate data) if a raw file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required raw data file is missing: {path}\n"
            "STOP — do not fabricate or download data. Place the H&M CSVs in "
            f"{config.RAW_DIR} and re-run."
        )


def load_customers() -> pd.DataFrame:
    """Load the customers master. ``customer_id`` is a string (never an int)."""
    _require(CUSTOMERS_PATH)
    return pd.read_csv(CUSTOMERS_PATH, dtype=CUSTOMER_DTYPES)


def load_articles() -> pd.DataFrame:
    """Load the articles master. ``article_id`` is a string (leading zeros kept)."""
    _require(ARTICLES_PATH)
    return pd.read_csv(ARTICLES_PATH, dtype=ARTICLE_DTYPES)


def load_transactions(customer_ids=None):
    """Load transactions with correct dtypes.

    Reads in chunks to keep memory bounded. If ``customer_ids`` is provided
    (any iterable of sampled customer ids), only rows for those customers are
    returned. The full, unfiltered row count is always tracked so callers can
    report the sample's share of total volume.

    Returns
    -------
    (df, total_rows) : tuple[pd.DataFrame, int]
        The (optionally filtered) transactions and the total unfiltered count.
    """
    _require(TRANSACTIONS_PATH)
    filter_set = set(customer_ids) if customer_ids is not None else None

    total_rows = 0
    parts = []
    reader = pd.read_csv(
        TRANSACTIONS_PATH,
        dtype=TRANSACTION_DTYPES,
        parse_dates=["t_dat"],
        chunksize=_CHUNK_SIZE,
    )
    for chunk in reader:
        total_rows += len(chunk)
        if filter_set is not None:
            chunk = chunk[chunk["customer_id"].isin(filter_set)]
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    # Category codes are assigned per-chunk; re-tighten after concat.
    df["sales_channel_id"] = df["sales_channel_id"].astype("category")
    return df, total_rows


def load_all(return_stats: bool = False):
    """Load all three tables with the customer sample applied.

    Sampling (the working dataset for the whole project):
      1. Load the full customers master.
      2. Randomly select ``config.SAMPLE_CUSTOMERS`` customer_ids using
         ``random_state = config.SAMPLE_SEED`` (fully reproducible).
      3. Keep only transactions for those customers, only those customers, and
         only the articles that appear in the sampled transactions.

    Returns the three sampled DataFrames ``(transactions, articles, customers)``.
    If ``return_stats`` is True, a fourth element (a stats dict) is appended.
    """
    customers = load_customers()
    total_customers = len(customers)

    # Sample customer ids reproducibly. Never take more than exist.
    n_sample = min(config.SAMPLE_CUSTOMERS, total_customers)
    sampled_ids = (
        customers["customer_id"]
        .drop_duplicates()
        .sample(n=n_sample, random_state=config.SAMPLE_SEED)
    )
    sampled_id_set = set(sampled_ids)

    # Transactions belonging to the sampled customers (+ full total for share).
    transactions, total_tx = load_transactions(customer_ids=sampled_id_set)

    # Keep only the sampled customers.
    customers = customers[
        customers["customer_id"].isin(sampled_id_set)
    ].reset_index(drop=True)

    # Keep only the articles that appear in the sampled transactions.
    sampled_article_ids = set(transactions["article_id"].unique())
    articles = load_articles()
    articles = articles[
        articles["article_id"].isin(sampled_article_ids)
    ].reset_index(drop=True)

    share = 100.0 * len(transactions) / total_tx if total_tx else 0.0
    stats = {
        "sampled_customers": len(customers),
        "sampled_transactions": len(transactions),
        "sampled_articles": len(articles),
        "total_customers": total_customers,
        "total_transactions": total_tx,
        "transaction_share_pct": share,
        "sample_seed": config.SAMPLE_SEED,
    }

    print("=" * 70)
    print(f"SAMPLING SUMMARY (seed = {config.SAMPLE_SEED})")
    print("=" * 70)
    print(f"  Sampled customers   : {stats['sampled_customers']:>12,}  "
          f"(of {total_customers:,} total)")
    print(f"  Sampled transactions: {stats['sampled_transactions']:>12,}  "
          f"(of {total_tx:,} total)")
    print(f"  Sampled articles    : {stats['sampled_articles']:>12,}")
    print(f"  Share of total tx   : {share:>12.2f}%  (sampled tx / full tx)")
    print("=" * 70)

    if return_stats:
        return transactions, articles, customers, stats
    return transactions, articles, customers


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _mem_mb(df: pd.DataFrame) -> float:
    """Deep memory footprint of a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1e6


def _report_table(name: str, df: pd.DataFrame) -> dict:
    """Print per-table stats and return null counts for the report."""
    print(f"\n--- {name} ---")
    print(f"  rows      : {len(df):,}")
    print(f"  columns   : {list(df.columns)}")
    print(f"  memory    : {_mem_mb(df):.2f} MB")
    nulls = df.isnull().sum()
    non_zero = nulls[nulls > 0]
    if non_zero.empty:
        print("  nulls     : none")
    else:
        print("  nulls     :")
        for col, n in non_zero.items():
            print(f"      {col:<28}: {int(n):,}")
    return {col: int(n) for col, n in nulls.items()}


def validate(transactions, articles, customers) -> dict:
    """Validate the SAMPLED tables; print a clear report and return findings.

    Assertions fail loudly (never silently drop):
      - no duplicate customer_id in the sampled customers table
      - no duplicate article_id in the sampled articles table
    Orphan references are *counted*, not dropped, so we can see join integrity.
    """
    print("\n" + "=" * 70)
    print("VALIDATION REPORT (sampled data)")
    print("=" * 70)

    null_tx = _report_table("transactions", transactions)
    null_art = _report_table("articles", articles)
    null_cust = _report_table("customers", customers)

    # Transaction date & price ranges.
    dmin, dmax = transactions["t_dat"].min(), transactions["t_dat"].max()
    pmin = float(transactions["price"].min())
    pmax = float(transactions["price"].max())
    print(f"\n  transaction date range : {dmin.date()}  ->  {dmax.date()}")
    print(f"  transaction price range: {pmin:.6f}  ->  {pmax:.6f}")

    # Primary-key uniqueness — fail loudly.
    dup_cust = int(customers["customer_id"].duplicated().sum())
    dup_art = int(articles["article_id"].duplicated().sum())
    assert dup_cust == 0, f"Duplicate customer_id in sampled customers: {dup_cust}"
    assert dup_art == 0, f"Duplicate article_id in sampled articles: {dup_art}"
    print("\n  [OK] no duplicate customer_id in sampled customers")
    print("  [OK] no duplicate article_id in sampled articles")

    # Orphan references — count, don't crash.
    known_articles = set(articles["article_id"])
    known_customers = set(customers["customer_id"])
    orphan_articles = int((~transactions["article_id"].isin(known_articles)).sum())
    orphan_customers = int((~transactions["customer_id"].isin(known_customers)).sum())
    print(f"\n  orphan transaction article_ids (missing from articles) : {orphan_articles:,}")
    print(f"  orphan transaction customer_ids (missing from customers): {orphan_customers:,}")

    return {
        "nulls": {"transactions": null_tx, "articles": null_art, "customers": null_cust},
        "date_min": dmin,
        "date_max": dmax,
        "price_min": pmin,
        "price_max": pmax,
        "orphan_articles": orphan_articles,
        "orphan_customers": orphan_customers,
        "memory_mb": {
            "transactions": _mem_mb(transactions),
            "articles": _mem_mb(articles),
            "customers": _mem_mb(customers),
        },
    }


# --------------------------------------------------------------------------
# Memory optimization & parquet
# --------------------------------------------------------------------------
def optimize_memory(transactions: pd.DataFrame) -> pd.DataFrame:
    """Downcast transactions to compact dtypes (ids stay string).

    - price            -> float32
    - sales_channel_id -> category
    - customer_id / article_id kept as string
    """
    out = transactions.copy()
    out["price"] = out["price"].astype("float32")
    out["sales_channel_id"] = out["sales_channel_id"].astype("category")
    out["customer_id"] = out["customer_id"].astype("string")
    out["article_id"] = out["article_id"].astype("string")
    return out


def save_parquet(transactions, articles, customers) -> None:
    """Write the sampled tables to data/processed as parquet."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(config.PROCESSED_DIR / "transactions.parquet", engine="pyarrow")
    articles.to_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    customers.to_parquet(config.PROCESSED_DIR / "customers.parquet", engine="pyarrow")
