"""Week 1 / Day 5 pipeline: leakage-safe temporal split + verification.

Run with:  python -m src.data.build_splits

Steps:
  1. Window sensitivity (7/14/28 days) — the basis for the chosen cutoff.
  2. Split the event log into feature (pre-cutoff) and label (window) sets.
  3. Build per-customer label targets (distinct action_ids in the window).
  4. Segment the customer base (core / history-no-label / cold-start / no-events).
  5. Verify no leakage and print LEAKAGE CHECK PASSED with boundary dates.
  6. Save features_events.parquet + labels.parquet; append the report section.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import splits as sp

_DAY5_MARKER = "<!-- day5: temporal split (regenerated) -->"


def _append_report(sensitivity, cutoff, feat, labels, label_events, seg, leak):
    path = config.REPORTS_DIR / "week1_data_profile.md"
    base = path.read_text() if path.exists() else "# Week 1 Data Profile\n"
    base = base.split(_DAY5_MARKER)[0].rstrip() + "\n"

    sens_rows = "\n".join(
        f"| {r.window_days} | {r.evaluable_customers:,} | {r.pct_of_active:.1f}% "
        f"| {r.feature_events_remaining:,} | {r.cutoff_date} |"
        for r in sensitivity.itertuples()
    )

    label_min = leak["label_min_date"].date()
    label_max = label_events["t_dat"].max().date()

    section = f"""{_DAY5_MARKER}

## Temporal Split

**Cutoff: `{cutoff.date()}`. Label window: `{cutoff.date()}` → `{label_max}`
({config.LABEL_WINDOW_DAYS} days).** Everything strictly before the cutoff is
feature history; the last {config.LABEL_WINDOW_DAYS} days are the prediction
target. The cutoff is computed in code as `max(t_dat) - LABEL_WINDOW_DAYS + 1`.

### Window sensitivity

Measured back from `max(t_dat) = {config.DATASET_MAX_DATE}`:

| window_days | evaluable_customers | % of active | feature_events_remaining | cutoff |
|---|---|---|---|---|
{sens_rows}

**Why 28 days.** None of the candidate windows reach a ~20,000 evaluable-customer
set. The 28-day window yields the largest evaluable set (16,895 customers, ~17%
of active) while still being a realistic 4-week short-horizon prediction task and
retaining 98.6% of events as features. The 7- and 14-day windows leave the
evaluation set too thin (5.2% and 9.4% of active), so we chose 28 over the
nominal default of 14 to get a healthier evaluable set.

### Why this prevents leakage

Features may only be computed from events **before** the cutoff; the label is the
set of distinct actions purchased **on/after** the cutoff. The two sets are a
clean partition of the event log on `t_dat`, so no information from the
prediction window can leak into the features. The boundary date is the guarantee.

### Feature / label sizes

| set | rows (events) | distinct customers |
|---|---|---|
| feature (pre-cutoff) | {leak['feature_rows']:,} | {feat['customer_id'].nunique():,} |
| label window | {leak['label_rows']:,} | {label_events['customer_id'].nunique():,} |
| labels.parquet (customer, action) pairs | {len(labels):,} | {labels['customer_id'].nunique():,} |

### Customer segmentation (of {seg['total_customers']:,} in the customer base)

| segment | customers | % |
|---|---|---|
| core — pre-cutoff history **and** ≥1 label purchase (trainable + evaluable) | {seg['core']:,} | {seg['core_pct']:.1f}% |
| history, no label-window purchase (nothing to predict this window) | {seg['history_no_label']:,} | {seg['history_no_label_pct']:.1f}% |
| label-window only, no feature history (**cold-start** at prediction time) | {seg['label_only']:,} | {seg['label_only_pct']:.1f}% |
| no events at all (in master, never purchased) | {seg['no_events']:,} | {seg['no_events_pct']:.1f}% |

**Cold-start note.** {seg['label_only']:,} customers have purchases only inside the
label window and therefore no pre-cutoff history to build features from; another
{seg['no_events']:,} have no events at all. These are flagged for **Week 2**
cold-start handling (popularity / action-prior fallback rather than a
personalized model).

### Leakage check

**LEAKAGE CHECK PASSED** — feature events end `{leak['feature_max_date'].date()}`
(strictly before the cutoff `{cutoff.date()}`) and label events run
`{label_min}` → `{label_max}`, with the two sets forming a clean, non-overlapping
partition of the event log. This guarantees no future information is available
when computing features.
"""
    path.write_text(base + "\n" + section)
    print(f"\nAppended Temporal Split section -> {path}")


def main():
    event_log = sp.load_event_log()

    # 1. Window sensitivity.
    sensitivity = sp.window_sensitivity(event_log)
    print("=" * 70)
    print("WINDOW SENSITIVITY (days back from max t_dat)")
    print("=" * 70)
    print(sensitivity.to_string(index=False))
    print(f"\n  -> chosen LABEL_WINDOW_DAYS = {config.LABEL_WINDOW_DAYS}, "
          f"CUTOFF_DATE = {config.CUTOFF_DATE}")

    # 2. Split.
    features_events, label_events, cutoff = sp.build_splits(event_log)

    # 3. Labels.
    labels = sp.build_labels(label_events)

    # 4. Segmentation (against the full customer base).
    customers = pd.read_parquet(config.PROCESSED_DIR / "customers.parquet", engine="pyarrow")
    seg = sp.segment_customers(
        customers["customer_id"],
        features_events["customer_id"].unique(),
        label_events["customer_id"].unique(),
    )
    print("\n" + "=" * 70)
    print("CUSTOMER SEGMENTATION (of {:,} in customer base)".format(seg["total_customers"]))
    print("=" * 70)
    print(f"  core (history + label purchase)   : {seg['core']:>7,} ({seg['core_pct']:.1f}%)")
    print(f"  history, no label purchase        : {seg['history_no_label']:>7,} ({seg['history_no_label_pct']:.1f}%)")
    print(f"  label-only (cold-start, no history): {seg['label_only']:>7,} ({seg['label_only_pct']:.1f}%)")
    print(f"  no events at all                  : {seg['no_events']:>7,} ({seg['no_events_pct']:.1f}%)")

    # 5. Leakage verification.
    leak = sp.verify_no_leakage(event_log, features_events, label_events, cutoff)
    print("\n" + "=" * 70)
    print(f"  feature events: {leak['feature_rows']:,} rows, "
          f"{features_events['customer_id'].nunique():,} customers, "
          f"max date {leak['feature_max_date'].date()}")
    print(f"  label events  : {leak['label_rows']:,} rows, "
          f"{label_events['customer_id'].nunique():,} customers, "
          f"min date {leak['label_min_date'].date()}")
    print(f"\nLEAKAGE CHECK PASSED — features end {leak['feature_max_date'].date()} "
          f"< cutoff {cutoff.date()} <= labels start {leak['label_min_date'].date()}")

    # 6. Save + report.
    sp.save_splits(features_events, labels)
    print(f"\n  wrote data/processed/features_events.parquet ({len(features_events):,} rows)")
    print(f"  wrote data/processed/labels.parquet ({len(labels):,} customer-action pairs)")

    _append_report(sensitivity, cutoff, features_events, labels, label_events, seg, leak)
    print("\nDONE.")


if __name__ == "__main__":
    main()
