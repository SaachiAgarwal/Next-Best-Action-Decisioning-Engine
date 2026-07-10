"""Week 1 / Day 4 pipeline: build the time-ordered event log + behavioral profile.

Run with:  python -m src.data.build_events

Steps:
  1. Load the sampled parquet and re-apply cleaning (idempotent).
  2. Build the time-ordered event log with per-customer sequence features.
  3. Compute + print the behavioral profile.
  4. Save event_log.parquet and verify it on reload.
  5. Append the "Event Log & Customer Behavior" section to the data profile.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import clean as clean_mod
from src.data import event_log as ev_mod
from src.data import load as load_mod

_DAY4_MARKER = "<!-- day4: event log + behavior (regenerated) -->"


def _verify_event_log():
    """Reload the event log and assert the invariants Day 4 requires."""
    el = pd.read_parquet(config.PROCESSED_DIR / "event_log.parquet", engine="pyarrow")

    assert str(el["article_id"].dtype) == "string", \
        f"article_id lost string dtype on reload: {el['article_id'].dtype}"

    # purchase_number starts at 1 for every customer.
    firsts = el.groupby("customer_id", sort=False)["purchase_number"].min()
    assert (firsts == 1).all(), "some customers do not start at purchase_number 1"

    # Sorted correctly within a representative sample of customers.
    sample_ids = el["customer_id"].drop_duplicates().head(500)
    sample = el[el["customer_id"].isin(set(sample_ids))]
    expected = sample.sort_values(["customer_id", "t_dat", ev_mod.TIE_BREAK], kind="stable")
    assert sample.reset_index(drop=True).equals(expected.reset_index(drop=True)), \
        "event log is not correctly sorted within sampled customers"

    print("  [OK] reload: article_id is string, purchase_number starts at 1, "
          "ordering verified on 500-customer sample")


def _append_report(stats):
    path = config.REPORTS_DIR / "week1_data_profile.md"
    base = path.read_text() if path.exists() else "# Week 1 Data Profile\n"
    base = base.split(_DAY4_MARKER)[0].rstrip() + "\n"

    section = f"""{_DAY4_MARKER}

## Event Log & Customer Behavior

The **event log** (`event_log.parquet`) is every purchase as a time-ordered,
per-customer sequence: one row per `(customer, article)` purchase, sorted by
`customer_id`, then `t_dat`, then `article_id`. This ordering is the temporal
backbone of the project — all downstream feature windows, train/eval splits, and
next-action targets read history strictly in time order, which is what prevents
**leakage** (using the future to predict the past). Same-day purchases (only
day-resolution timestamps exist) are tie-broken by `article_id` so the sequence
is deterministic and reproducible.

**Sequence features added (within each customer's ordered history):**

- `purchase_number` — 1..n ordinal position of the purchase.
- `days_since_first_purchase` — days between this event and the customer's first.
- `days_since_prev_purchase` — days since the immediately prior event; **0 for
  the first purchase** (no prior event), kept non-null and integer.

**Behavioral profile ({stats['n_customers']:,} customers, {stats['n_events']:,} events):**

| metric | value |
|---|---|
| transactions/customer — min / median / mean / max | {stats['tx_per_customer_min']} / {stats['tx_per_customer_median']:.0f} / {stats['tx_per_customer_mean']:.2f} / {stats['tx_per_customer_max']} |
| purchases/customer percentiles — 50th / 90th / 99th | {stats['pct50']:.0f} / {stats['pct90']:.0f} / {stats['pct99']:.0f} |
| single-purchase customers (cold-start) | {stats['single_count']:,} ({stats['single_pct']:.1f}%) |
| customers with ≥10 purchases (rich history) | {stats['rich_count']:,} ({stats['rich_pct']:.1f}%) |
| avg repurchase gap (repeat purchases) | {stats['avg_repurchase_gap_days']:.1f} days |
| median customer tenure (date span covered) | {stats['median_tenure_days']:.0f} days |

**Sparsity / cold-start note.** {stats['single_pct']:.1f}% of customers have only
a single purchase in the sampled window — a non-trivial cold-start segment with
no personal repeat-behavior signal at prediction time. Next-action modeling must
handle these customers explicitly — e.g. fall back to popularity / action-prior
recommendations — rather than assuming a rich per-customer sequence. At the other
end, {stats['rich_pct']:.1f}% of customers have ≥10 purchases; this is where
personalized sequence signal is strongest. The long right tail (99th percentile
= {stats['pct99']:.0f} purchases, max {stats['tx_per_customer_max']}) means a
small set of very active customers contributes a disproportionate share of events.
"""
    path.write_text(base + "\n" + section)
    print(f"\nAppended Event Log & Customer Behavior section -> {path}")


def main():
    # 1. Load + clean (idempotent on already-cleaned processed data).
    transactions, articles, customers = load_mod.load_processed()
    transactions, articles, customers, _log = clean_mod.clean(transactions, articles, customers)
    article_action_map = pd.read_parquet(
        config.PROCESSED_DIR / "article_action_map.parquet", engine="pyarrow"
    )

    # 2. Build event log.
    event_log = ev_mod.build_event_log(transactions, article_action_map)
    print(f"Built event log: {len(event_log):,} events, "
          f"{event_log['customer_id'].nunique():,} customers\n")

    # 3. Profile.
    stats = ev_mod.profile_behavior(event_log)
    ev_mod.print_profile(stats)

    # 4. Save + verify.
    ev_mod.save_event_log(event_log)
    print(f"\n  wrote data/processed/event_log.parquet ({len(event_log):,} rows)")
    _verify_event_log()

    # 5. Report.
    _append_report(stats)

    print("\nDONE.")


if __name__ == "__main__":
    main()
