"""Time-ordered event log construction for the NBA Decisioning Engine.

Transforms the cleaned transactions into a per-customer, chronological event
log — the temporal backbone of the whole project. Every downstream temporal
operation (feature windows, train/eval splits, next-action targets) depends on
this ordering being correct and reproducible, so we pin the sort here.

Event schema:
    customer_id, t_dat, article_id, action_id, price,
    purchase_number, days_since_first_purchase, days_since_prev_purchase

Ordering & same-day tie-break:
    Rows are sorted by (customer_id asc, t_dat asc, article_id asc). Multiple
    purchases on the same day are common; ``article_id`` is a stable, data-driven
    tie-break so ``purchase_number`` is deterministic and reproducible across
    runs and machines (it does not imply real intra-day time order — the source
    has only day resolution).

First-purchase convention:
    ``days_since_prev_purchase`` is **0** for each customer's first purchase
    (there is no prior event). This keeps the column non-null and integer;
    repurchase-gap statistics are computed only over purchase_number > 1.

Ids stay strings; action_id is int; dates are datetime.
"""

from __future__ import annotations

import pandas as pd

from src import config

TIE_BREAK = "article_id"  # stable within-customer-day ordering key
EVENT_COLUMNS = [
    "customer_id", "t_dat", "article_id", "action_id", "price",
    "purchase_number", "days_since_first_purchase", "days_since_prev_purchase",
]


def build_event_log(transactions, article_action_map) -> pd.DataFrame:
    """Build the time-ordered event log with per-customer sequence features.

    Parameters
    ----------
    transactions : cleaned transactions (customer_id, t_dat, article_id, price, ...)
    article_action_map : article_id -> action_id mapping

    Returns the event log DataFrame sorted by (customer_id, t_dat, article_id).
    """
    ev = transactions[["customer_id", "t_dat", "article_id", "price"]].copy()

    # Attach each event's action via the article->action map.
    ev = ev.merge(
        article_action_map[["article_id", "action_id"]], on="article_id", how="left"
    )
    if ev["action_id"].isna().any():
        n = int(ev["action_id"].isna().sum())
        raise ValueError(f"{n} events have no action_id — article->action map is incomplete")
    ev["action_id"] = ev["action_id"].astype("int64")

    # THE ordering: customer, then time, then a stable tie-break for same-day rows.
    ev = ev.sort_values(
        ["customer_id", "t_dat", TIE_BREAK], kind="stable"
    ).reset_index(drop=True)

    grp = ev.groupby("customer_id", sort=False)

    # 1..n ordinal position of each purchase within the customer's history.
    ev["purchase_number"] = (grp.cumcount() + 1).astype("int32")

    # Days since the customer's first (earliest) purchase.
    first_dat = grp["t_dat"].transform("first")
    ev["days_since_first_purchase"] = (ev["t_dat"] - first_dat).dt.days.astype("int32")

    # Days since the customer's immediately prior purchase (0 for the first).
    prev_dat = grp["t_dat"].shift(1)
    gap = (ev["t_dat"] - prev_dat).dt.days
    ev["days_since_prev_purchase"] = gap.fillna(0).astype("int32")

    # Efficient dtypes for ids/price.
    ev["customer_id"] = ev["customer_id"].astype("string")
    ev["article_id"] = ev["article_id"].astype("string")
    ev["price"] = ev["price"].astype("float32")

    return ev[EVENT_COLUMNS]


def profile_behavior(event_log) -> dict:
    """Compute the behavioral profile over the event log. Returns a stats dict."""
    per_customer = event_log.groupby("customer_id", sort=False)
    counts = per_customer.size()
    n_customers = len(counts)

    # Tenure: date span actually covered per customer, in days.
    span_days = (per_customer["t_dat"].max() - per_customer["t_dat"].min()).dt.days

    single = int((counts == 1).sum())
    rich = int((counts >= 10).sum())

    # Typical repurchase gap: mean gap across repeat purchases only.
    repeat_gap = event_log.loc[
        event_log["purchase_number"] > 1, "days_since_prev_purchase"
    ]
    avg_repurchase_gap = float(repeat_gap.mean()) if len(repeat_gap) else 0.0

    return {
        "n_customers": n_customers,
        "n_events": len(event_log),
        "tx_per_customer_min": int(counts.min()),
        "tx_per_customer_median": float(counts.median()),
        "tx_per_customer_mean": float(counts.mean()),
        "tx_per_customer_max": int(counts.max()),
        "pct50": float(counts.quantile(0.50)),
        "pct90": float(counts.quantile(0.90)),
        "pct99": float(counts.quantile(0.99)),
        "single_count": single,
        "single_pct": 100.0 * single / n_customers,
        "rich_count": rich,
        "rich_pct": 100.0 * rich / n_customers,
        "avg_repurchase_gap_days": avg_repurchase_gap,
        "median_tenure_days": float(span_days.median()),
    }


def print_profile(stats: dict) -> None:
    """Print the behavioral profile."""
    print("=" * 70)
    print("BEHAVIORAL PROFILE")
    print("=" * 70)
    print(f"  customers                         : {stats['n_customers']:,}")
    print(f"  events (purchases)                : {stats['n_events']:,}")
    print("\n  transactions per customer:")
    print(f"    min / median / mean / max       : "
          f"{stats['tx_per_customer_min']} / {stats['tx_per_customer_median']:.0f} / "
          f"{stats['tx_per_customer_mean']:.2f} / {stats['tx_per_customer_max']}")
    print(f"    percentiles (50 / 90 / 99)      : "
          f"{stats['pct50']:.0f} / {stats['pct90']:.0f} / {stats['pct99']:.0f}")
    print(f"\n  single-purchase customers (cold)  : {stats['single_count']:,} "
          f"({stats['single_pct']:.1f}%)")
    print(f"  rich-history customers (>=10 tx)  : {stats['rich_count']:,} "
          f"({stats['rich_pct']:.1f}%)")
    print(f"\n  avg repurchase gap (repeat buys)  : "
          f"{stats['avg_repurchase_gap_days']:.1f} days")
    print(f"  median customer tenure            : {stats['median_tenure_days']:.0f} days")


def save_event_log(event_log) -> None:
    """Persist the event log to data/processed/event_log.parquet."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    event_log.to_parquet(config.PROCESSED_DIR / "event_log.parquet", engine="pyarrow")
