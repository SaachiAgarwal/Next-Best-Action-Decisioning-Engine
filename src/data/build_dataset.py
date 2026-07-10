"""Build the sampled working dataset: load -> validate -> optimize -> parquet.

Run with:  python -m src.data.build_dataset

This is the Week 1 / Day 2 pipeline. It:
  1. Loads the three raw H&M CSVs and applies the customer sample (config).
  2. Prints sample sizes and the sample's share of total transaction volume.
  3. Validates the sampled tables (nulls, ranges, PK uniqueness, orphans).
  4. Reports memory before vs after optimization.
  5. Writes the sampled tables to data/processed as parquet and verifies the
     article_id string dtype survives the round-trip.
  6. Writes reports/week1_data_profile.md.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import load as load_mod


def _mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e6


def _write_report(stats, findings, mem_before, mem_after, dtypes):
    """Write the standalone markdown data profile a PM could read cold."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "week1_data_profile.md"

    n_tx = f"{stats['sampled_transactions']:,}"
    n_cust = f"{stats['sampled_customers']:,}"
    n_art = f"{stats['sampled_articles']:,}"
    share = stats["transaction_share_pct"]
    total_tx = f"{stats['total_transactions']:,}"

    def null_block(name):
        nulls = findings["nulls"][name]
        rows = [(c, n) for c, n in nulls.items() if n > 0]
        if not rows:
            return "_No nulls._\n"
        out = "| column | null count |\n|---|---|\n"
        for c, n in rows:
            out += f"| `{c}` | {n:,} |\n"
        return out

    lines = f"""# Week 1 Data Profile — NBA Decisioning Engine

## Sampling note (read this first)

The working dataset for this project is **{n_cust} customers** (random sample,
**seed = {stats['sample_seed']}**) together with their **complete purchase
history**. This sample was chosen so development runs quickly and reproducibly
on a local machine. The pipeline is **size-agnostic**: it runs identically on
the full H&M dataset ({total_tx} transactions) — only `SAMPLE_CUSTOMERS` in
`src/config.py` changes.

The sample covers **{share:.2f}%** of total transaction volume
({n_tx} of {total_tx} transactions). Because we sample whole customers and keep
their full histories, every sampled transaction, customer, and article is
internally consistent (see orphan counts below).

## Dataset shape (sampled)

| table | rows |
|---|---|
| customers | {n_cust} |
| articles | {n_art} |
| transactions | {n_tx} |

## Transactions: date & price range

- **Date range:** {findings['date_min'].date()} → {findings['date_max'].date()}
- **Price range:** {findings['price_min']:.6f} → {findings['price_max']:.6f}
  (price is **normalized to [0, 1]** in the source data — not a currency amount)

## Null summary

**transactions**

{null_block('transactions')}
**articles**

{null_block('articles')}
**customers**

{null_block('customers')}
## Referential integrity (orphans)

Counts of transaction references that point at a missing master row. Because we
sample whole customers and derive the article set from their transactions, both
should be ~0.

| orphan type | count |
|---|---|
| transaction `article_id` missing from articles | {findings['orphan_articles']:,} |
| transaction `customer_id` missing from customers | {findings['orphan_customers']:,} |

## Memory footprint

Transactions memory, before vs after optimization (price → float32,
sales_channel_id → category, ids kept as string):

| stage | memory (MB) |
|---|---|
| before optimization | {mem_before:.2f} |
| after optimization | {mem_after:.2f} |
| reduction | {mem_before - mem_after:.2f} ({100 * (mem_before - mem_after) / mem_before:.1f}%) |

Optimized footprint of each sampled table:

| table | memory (MB) |
|---|---|
| transactions | {findings['memory_mb']['transactions']:.2f} |
| articles | {findings['memory_mb']['articles']:.2f} |
| customers | {findings['memory_mb']['customers']:.2f} |

## Column dtypes (transactions)

| column | dtype |
|---|---|
"""
    for col, dt in dtypes.items():
        lines += f"| `{col}` | `{dt}` |\n"

    lines += """
## Notes

- `article_id` and `customer_id` are stored as **strings** to preserve leading
  zeros; loading them as integers would silently break every join.
- The sampled parquet files live in `data/processed/` and are **git-ignored** —
  this report is the only record of the data that survives in the repo.
"""
    path.write_text(lines)
    print(f"\nWrote data profile -> {path}")


def main():
    # 1. Load + sample.
    transactions, articles, customers, stats = load_mod.load_all(return_stats=True)

    # 2. Validate.
    findings = load_mod.validate(transactions, articles, customers)

    # 3. Memory before vs after optimization.
    print("\n" + "=" * 70)
    print("MEMORY OPTIMIZATION (transactions)")
    print("=" * 70)
    # "Before": naive dtypes a plain read would produce (float64 price,
    # de-categorized sales_channel_id) so the optimization delta is visible.
    before = transactions.copy()
    before["price"] = before["price"].astype("float64")
    before["sales_channel_id"] = before["sales_channel_id"].astype("string")
    mem_before = _mem_mb(before)

    optimized = load_mod.optimize_memory(transactions)
    mem_after = _mem_mb(optimized)
    print(f"  before: {mem_before:8.2f} MB")
    print(f"  after : {mem_after:8.2f} MB")
    print(f"  saved : {mem_before - mem_after:8.2f} MB "
          f"({100 * (mem_before - mem_after) / mem_before:.1f}%)")

    transactions = optimized

    # 4. Dtypes of each table.
    print("\n" + "=" * 70)
    print("DTYPES")
    print("=" * 70)
    for name, df in (("transactions", transactions), ("articles", articles),
                     ("customers", customers)):
        print(f"\n--- {name} ---")
        print(df.dtypes.to_string())

    # 5. Save parquet + round-trip check.
    print("\n" + "=" * 70)
    print("PARQUET WRITE + ROUND-TRIP CHECK")
    print("=" * 70)
    load_mod.save_parquet(transactions, articles, customers)
    for f in ("transactions.parquet", "articles.parquet", "customers.parquet"):
        print(f"  wrote data/processed/{f}")

    rt = pd.read_parquet(config.PROCESSED_DIR / "transactions.parquet", engine="pyarrow")
    assert str(rt["article_id"].dtype) == "string", (
        f"parquet round-trip lost string dtype: {rt['article_id'].dtype}"
    )
    print(f"  [OK] transactions.parquet article_id dtype after reload: "
          f"{rt['article_id'].dtype} (leading-zero example: {rt['article_id'].iloc[0]})")

    # 6. Report.
    _write_report(
        stats, findings, mem_before, mem_after,
        dtypes={c: str(t) for c, t in transactions.dtypes.items()},
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
