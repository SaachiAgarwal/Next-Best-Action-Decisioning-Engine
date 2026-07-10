"""Data cleaning layer for the NBA Decisioning Engine.

Cleaning is **explicit and logged** — we never drop rows silently. Every
transformation records what it touched and how many rows/values it affected, so
the cleaning log in ``reports/week1_data_profile.md`` is a faithful audit trail.

Scope (Week 1 / Day 3):
  1. Null handling — documented per column with a chosen strategy
     (impute / keep-as-category / keep). We only *drop* rows that are genuinely
     unusable; categorical metadata nulls become an explicit "unknown" category.
  2. Text standardization — strip whitespace and lowercase the categorical
     "name" columns on articles. ``article_id`` is left untouched (string).
  3. Impossible-transaction removal — only price <= 0 or ``t_dat`` outside the
     known dataset date window. Each rule's removed-row count is reported.

``article_id`` and ``customer_id`` remain strings throughout (see load.py).
"""

from __future__ import annotations

import pandas as pd

# Known valid transaction window for the H&M dataset (the competition's range).
# Rows with t_dat outside this window are considered impossible.
KNOWN_DATE_MIN = pd.Timestamp("2018-09-20")
KNOWN_DATE_MAX = pd.Timestamp("2020-09-22")

# Categorical "name" columns on articles that we strip + lowercase.
ARTICLE_NAME_COLS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]

# Explicit null strategy per (table, column). Any column with nulls that is not
# listed here falls back to keep-as-category ("unknown") for text columns and is
# logged, so new nulls can never pass through undocumented.
NULL_STRATEGY = {
    # customers
    ("customers", "FN"): ("impute", 0.0,
                          "binary flag; missing means not flagged -> 0"),
    ("customers", "Active"): ("impute", 0.0,
                             "binary flag; missing means not active -> 0"),
    ("customers", "club_member_status"): ("keep-as-category", "unknown",
                                          "categorical metadata"),
    ("customers", "fashion_news_frequency"): ("keep-as-category", "unknown",
                                              "categorical metadata"),
    ("customers", "age"): ("keep", None,
                          "numeric; imputation deferred to feature engineering"),
    # articles
    ("articles", "detail_desc"): ("keep-as-category", "unknown",
                                  "free-text description; fill placeholder"),
}


def _handle_nulls(df: pd.DataFrame, table: str, log: list) -> pd.DataFrame:
    """Apply the documented null strategy to every column that has nulls."""
    df = df.copy()
    null_counts = df.isnull().sum()
    for col, n in null_counts.items():
        if n == 0:
            continue
        strategy, fill, reason = NULL_STRATEGY.get(
            (table, col), ("keep-as-category", "unknown", "categorical metadata (default)")
        )
        if strategy in ("impute", "keep-as-category") and fill is not None:
            df[col] = df[col].fillna(fill)
        # strategy == "keep": leave nulls in place (documented, not dropped).
        log.append({
            "step": "nulls",
            "table": table,
            "target": col,
            "strategy": strategy,
            "affected": int(n),
            "detail": reason + (f" (fill='{fill}')" if fill is not None else " (left as-is)"),
        })
    return df


def _standardize_article_text(articles: pd.DataFrame, log: list) -> pd.DataFrame:
    """Strip whitespace and lowercase categorical name columns on articles."""
    articles = articles.copy()
    for col in ARTICLE_NAME_COLS:
        if col not in articles.columns:
            continue
        before = articles[col].astype("string")
        after = before.str.strip().str.lower()
        changed = int((before.fillna("\x00") != after.fillna("\x00")).sum())
        articles[col] = after
        log.append({
            "step": "text",
            "table": "articles",
            "target": col,
            "strategy": "strip+lowercase",
            "affected": changed,
            "detail": "standardized categorical name text",
        })
    return articles


def _remove_impossible_transactions(transactions: pd.DataFrame, log: list) -> pd.DataFrame:
    """Remove only genuinely impossible transactions; report each rule's count."""
    n0 = len(transactions)

    bad_price = transactions["price"] <= 0
    n_price = int(bad_price.sum())

    out_of_range = (transactions["t_dat"] < KNOWN_DATE_MIN) | (transactions["t_dat"] > KNOWN_DATE_MAX)
    n_date = int(out_of_range.sum())

    keep = ~(bad_price | out_of_range)
    cleaned = transactions[keep].reset_index(drop=True)

    log.append({
        "step": "filter",
        "table": "transactions",
        "target": "price <= 0",
        "strategy": "drop (impossible)",
        "affected": n_price,
        "detail": "non-positive price",
    })
    log.append({
        "step": "filter",
        "table": "transactions",
        "target": f"t_dat outside [{KNOWN_DATE_MIN.date()}, {KNOWN_DATE_MAX.date()}]",
        "strategy": "drop (impossible)",
        "affected": n_date,
        "detail": "date outside known dataset window",
    })
    log.append({
        "step": "filter",
        "table": "transactions",
        "target": "TOTAL removed",
        "strategy": "drop (impossible)",
        "affected": n0 - len(cleaned),
        "detail": f"{n0:,} -> {len(cleaned):,} rows",
    })
    return cleaned


def clean(transactions, articles, customers):
    """Clean the three sampled tables. Returns (transactions, articles, customers, log).

    ``log`` is a list of transformation records (dicts) suitable for rendering
    into the report's Cleaning section.
    """
    log = []

    customers = _handle_nulls(customers, "customers", log)
    articles = _handle_nulls(articles, "articles", log)
    transactions = _handle_nulls(transactions, "transactions", log)

    articles = _standardize_article_text(articles, log)

    transactions = _remove_impossible_transactions(transactions, log)

    # Ids must remain strings.
    transactions["article_id"] = transactions["article_id"].astype("string")
    transactions["customer_id"] = transactions["customer_id"].astype("string")
    articles["article_id"] = articles["article_id"].astype("string")
    customers["customer_id"] = customers["customer_id"].astype("string")

    return transactions, articles, customers, log


def format_cleaning_log_md(log: list) -> str:
    """Render the cleaning log as a markdown table body."""
    lines = ["| step | table | target | strategy | affected | note |",
             "|---|---|---|---|---|---|"]
    for e in log:
        lines.append(
            f"| {e['step']} | {e['table']} | `{e['target']}` | {e['strategy']} "
            f"| {e['affected']:,} | {e['detail']} |"
        )
    return "\n".join(lines)


def print_cleaning_log(log: list) -> None:
    """Print the cleaning log to stdout."""
    print("=" * 70)
    print("CLEANING LOG")
    print("=" * 70)
    for e in log:
        print(f"  [{e['step']:<6}] {e['table']:<12} {e['target']:<45} "
              f"{e['strategy']:<18} affected={e['affected']:>9,}  ({e['detail']})")
