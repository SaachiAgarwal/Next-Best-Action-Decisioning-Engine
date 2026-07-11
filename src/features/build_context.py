"""Phase 1 driver: build the customer context layer + report.

Run with:  python -m src.features.build_context
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features import context as ctx_mod

OUT_PATH = config.PROCESSED_DIR / "customer_context.parquet"


def _dist(series):
    q = series.describe(percentiles=[0.5, 0.9])
    return q


def _write_report(ctx, model_ready, feature_names):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "context_layer.md"

    warm = ctx[~ctx["is_cold_start"]]
    n_total = len(ctx)
    n_cold = int(ctx["is_cold_start"].sum())

    def stat_row(name, s):
        return (f"| {name} | {s.mean():.2f} | {s.median():.2f} | "
                f"{s.quantile(0.9):.2f} | {s.min():.2f} | {s.max():.2f} |")

    rfm_rows = "\n".join([
        stat_row("recency_days", warm["recency_days"]),
        stat_row("frequency", warm["frequency"]),
        stat_row("monetary_total", warm["monetary_total"]),
        stat_row("monetary_avg", warm["monetary_avg"]),
        stat_row("distinct_actions", warm["distinct_actions"]),
        stat_row("tenure_days", warm["tenure_days"]),
        stat_row("avg_repurchase_gap_days", warm["avg_repurchase_gap_days"]),
    ])

    def cat_table(col):
        vc = ctx[col].value_counts(dropna=False)
        return "\n".join(f"| {k} | {v:,} | {100*v/n_total:.1f}% |" for k, v in vc.items())

    content = f"""# Phase 1 — Customer Context / Feature Layer

## What this is

The **customer context vector** is the per-customer state the contextual bandit
conditions on: one row per customer (`customer_context.parquet`), computed
strictly from pre-cutoff (feature-side) events. It answers **"who is this
customer"** — how recently/often/valuably they buy, how broad their taste, and
their attributes.

## Design note — context vs. per-action scores (deliberate)

The context is **aggregate customer-state**, intentionally distinct from the
recommender's **per-action affinity scores**. The scorer answers "which action is
good for this customer"; the context answers "who is this customer". Keeping them
separate is a deliberate architecture decision: it lets the bandit's context
**complement** the candidate scorer rather than duplicate its signal. The scorer
proposes and ranks actions; the context describes the person the policy is
deciding for — together they give the bandit both *what's available* and *who
it's for*, without redundancy.

## Feature list

**RFM (customer-level, from feature-side events):**
- `recency_days` — days from the customer's last pre-cutoff purchase to the cutoff.
- `frequency` — total pre-cutoff transaction count.
- `monetary_total` / `monetary_avg` — sum / mean of `price` (price is normalized
  to [0, 1], so this is a *relative engagement* signal, not currency).

**Attributes (from customers.parquet; nulls -> "unknown", never dropped):**
- `age` + `age_band` (`<=25 / 26-35 / 36-45 / 46-55 / 56+ / unknown`).
- `club_member_status`, `fashion_news_frequency`.

**Behavioral breadth:**
- `distinct_actions` — number of distinct product-type actions bought (variety).
- `tenure_days` — days from first pre-cutoff purchase to the cutoff.
- `avg_repurchase_gap_days` — mean gap between consecutive purchases
  (**0 by convention for single-purchase customers**; equals
  (last - first) / (frequency - 1)).
- `dominant_action_id` + `dominant_action_share` — the customer's most-purchased
  action and its share of their purchases (cold-start sentinel `-1`).

Plus `is_cold_start` — a flag distinguishing history-less customers.

## Distributions (customers with history, n={len(warm):,})

| feature | mean | median | p90 | min | max |
|---|---|---|---|---|---|
{rfm_rows}

**age_band**

| band | customers | share |
|---|---|---|
{cat_table('age_band')}

**club_member_status**

| status | customers | share |
|---|---|---|
{cat_table('club_member_status')}

**fashion_news_frequency**

| frequency | customers | share |
|---|---|---|
{cat_table('fashion_news_frequency')}

## Cold-start handling

Of **{n_total:,}** customers in the base, **{n_cold:,}**
({100*n_cold/n_total:.1f}%) have no pre-cutoff history. They are **retained**,
not dropped, with explicit safe defaults: `frequency=0`, `monetary_*=0`,
`distinct_actions=0`, `dominant_action_id=-1`, `avg_repurchase_gap_days=0`,
`recency_days`/`tenure_days` left NaN (flagged), and `is_cold_start=True`. The
bandit must be able to decide for these customers (it falls back to popularity via
the recommender), so the context layer represents them explicitly rather than
omitting them.

## Model-ready encoding

`build_model_ready()` deterministically regenerates a numeric matrix from the raw
table: numeric features median-imputed (cold-start recency/tenure/age) then
standardized, categoricals one-hot encoded. Result: **{model_ready.shape[0]:,}
rows x {model_ready.shape[1]-1} feature columns**, **no NaNs**. The raw table is
saved as `customer_context.parquet`; the encoded matrix is regenerated on demand
(deterministic), so raw stays human-readable for the report and constraints layer.

## Leakage guarantee

Every feature derives only from events with `t_dat < {config.CUTOFF_DATE}`;
`context.build` asserts the input's max date is strictly before the cutoff. No
post-cutoff (label-window) information enters the context vector.
"""
    path.write_text(content)
    print(f"\nWrote {path}")


def main():
    fe, customers = ctx_mod.load_inputs()

    ctx = ctx_mod.build(fe, customers)
    # Leakage guard (build already asserts; make it explicit + printed).
    assert fe["t_dat"].max() < pd.Timestamp(config.CUTOFF_DATE)
    print("CONTEXT LEAKAGE CHECK PASSED — all features from t_dat < "
          f"{config.CUTOFF_DATE} (feature max {fe['t_dat'].max().date()})")

    # One row per customer, no duplicates.
    assert ctx["customer_id"].is_unique, "duplicate customer_id in context table"
    print(f"\nContext table: {len(ctx):,} customers x {ctx.shape[1]} raw columns "
          f"({int(ctx['is_cold_start'].sum()):,} cold-start)")

    model_ready, scaler, feature_names = ctx_mod.build_model_ready(ctx)
    assert not model_ready.drop(columns=["customer_id"]).isna().any().any(), \
        "NaN leaked into the model-ready matrix"
    print(f"Model-ready matrix: {model_ready.shape[0]:,} x {len(feature_names)} features, "
          f"no NaNs. Features: {feature_names[:6]} ...")

    ctx.to_parquet(OUT_PATH, engine="pyarrow")
    print(f"\nWrote {OUT_PATH} ({len(ctx):,} rows)")

    # Quick printed distributions.
    warm = ctx[~ctx["is_cold_start"]]
    print("\nSummary (customers with history):")
    for f in ["recency_days", "frequency", "monetary_total", "distinct_actions", "tenure_days"]:
        s = warm[f]
        print(f"  {f:<24} mean={s.mean():8.2f}  median={s.median():8.2f}  p90={s.quantile(0.9):8.2f}")
    print("\n  age_band counts:", dict(ctx["age_band"].value_counts()))
    print("  club_member_status counts:", dict(ctx["club_member_status"].value_counts()))

    _write_report(ctx, model_ready, feature_names)
    print("\nDONE.")


if __name__ == "__main__":
    main()
