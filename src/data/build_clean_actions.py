"""Week 1 / Day 3 pipeline: clean the sampled data and define the action space.

Run with:  python -m src.data.build_clean_actions

Steps:
  1. Load the sampled parquet (data/processed) via the load layer.
  2. Clean explicitly + logged (nulls, text standardization, impossible tx).
  3. Build the action space at product_type_name granularity.
  4. Save actions.parquet + article_action_map.parquet.
  5. Append "Cleaning" and "Action Space" sections to the data profile report.
"""

from __future__ import annotations

from src import config
from src.data import action_space as asp
from src.data import clean as clean_mod
from src.data import load as load_mod

# Everything below this marker in the report is regenerated on each run.
_DAY3_MARKER = "<!-- day3: cleaning + action space (regenerated) -->"


def _append_report(log, actions, stats):
    path = config.REPORTS_DIR / "week1_data_profile.md"
    base = path.read_text() if path.exists() else "# Week 1 Data Profile\n"
    # Drop any previously-appended Day 3 content so re-runs stay idempotent.
    base = base.split(_DAY3_MARKER)[0].rstrip() + "\n"

    top15 = stats["top15"]
    top15_rows = "\n".join(
        f"| {int(r.action_id)} | {r.product_type_name} | {int(r.article_count):,} "
        f"| {int(r.total_purchases):,} | {int(r.distinct_customers):,} |"
        for r in top15.itertuples()
    )

    section = f"""{_DAY3_MARKER}

## Cleaning

Cleaning is explicit and logged — no rows are dropped silently. Categorical
metadata nulls are kept as an explicit `unknown` category; only genuinely
impossible transactions (non-positive price, or a date outside the known
dataset window {clean_mod.KNOWN_DATE_MIN.date()} → {clean_mod.KNOWN_DATE_MAX.date()})
are removed. Every transformation and the number of rows/values it affected:

{clean_mod.format_cleaning_log_md(log)}

## Action Space

**Granularity: `product_type_name`.** Product-type granularity balances
recommendation precision against statistical learnability — it avoids
article-level sparsity (too many actions, too little signal each) while staying
more actionable than broad product groups.

- **Total actions (action-space size): {stats['action_space_size']}**
- An `article_id → action` mapping (`article_action_map.parquet`) is retained so
  a recommended action can be drilled down to the specific articles behind it.
- **Long tail:** {stats['long_tail_count']} of {stats['action_space_size']} actions
  have fewer than {stats['long_tail_threshold']} purchases in the sampled data;
  these thin actions will be the hardest to learn reliably downstream.

Top 15 actions by purchase volume:

| action_id | product_type_name | article_count | total_purchases | distinct_customers |
|---|---|---|---|---|
{top15_rows}
"""
    path.write_text(base + "\n" + section)
    print(f"\nAppended Cleaning + Action Space sections -> {path}")


def main():
    # 1. Load sampled parquet.
    transactions, articles, customers = load_mod.load_processed()

    # 2. Clean (logged).
    transactions, articles, customers, log = clean_mod.clean(transactions, articles, customers)
    clean_mod.print_cleaning_log(log)

    # Persist cleaned tables back to processed so downstream stages use them.
    load_mod.save_parquet(transactions, articles, customers)

    # 3. Build action space.
    actions, article_action_map, stats = asp.build_action_space(transactions, articles)
    print()
    asp.print_action_space(stats)

    # 4. Save action-space parquet.
    asp.save_action_space(actions, article_action_map)
    print(f"\n  wrote data/processed/actions.parquet ({len(actions)} rows)")
    print(f"  wrote data/processed/article_action_map.parquet ({len(article_action_map):,} rows)")

    # 5. Report.
    _append_report(log, actions, stats)

    print("\nDONE.")


if __name__ == "__main__":
    main()
