"""Leakage-safe temporal split for the NBA Decisioning Engine.

Splits the event log at ``config.CUTOFF_DATE`` into a FEATURE set (history) and
a LABEL set (the prediction window):

    FEATURE = events with t_dat <  CUTOFF_DATE
    LABEL   = events with t_dat >= CUTOFF_DATE   (the last LABEL_WINDOW_DAYS)

Why this prevents leakage: features may only ever be computed from events that
happened strictly before the cutoff, and we predict distinct actions purchased
in the label window. Because the two sets are a clean partition of the event log
on ``t_dat``, no information from the prediction window can flow into the
features — the time boundary is the guarantee.

Customers with no label-window purchase are **kept** (they simply have no
positive labels); dropping them would bias the evaluation toward active buyers.

Outputs (data/processed/):
    features_events.parquet  -- the pre-cutoff event log (feature side)
    labels.parquet           -- distinct (customer_id, action_id) pairs bought
                                in the label window (the per-customer label set)
"""

from __future__ import annotations

import pandas as pd

from src import config

SENSITIVITY_WINDOWS = (7, 14, 28)


def load_event_log() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "event_log.parquet", engine="pyarrow")


def cutoff_timestamp() -> pd.Timestamp:
    """Config cutoff as a pandas Timestamp for comparison against t_dat."""
    return pd.Timestamp(config.CUTOFF_DATE)


def window_sensitivity(event_log, windows=SENSITIVITY_WINDOWS) -> pd.DataFrame:
    """Evaluable-set sensitivity for each candidate label window.

    For each window (days back from max t_dat): how many customers have >=1
    purchase in the window (evaluable set), its share of active customers, and
    how many feature-side events remain pre-cutoff.
    """
    max_d = event_log["t_dat"].max()
    active = event_log["customer_id"].nunique()
    rows = []
    for w in windows:
        cutoff = max_d - pd.Timedelta(days=w) + pd.Timedelta(days=1)
        in_window = event_log["t_dat"] >= cutoff
        evaluable = event_log.loc[in_window, "customer_id"].nunique()
        feature_events = int((~in_window).sum())
        rows.append({
            "window_days": w,
            "evaluable_customers": evaluable,
            "pct_of_active": 100.0 * evaluable / active,
            "feature_events_remaining": feature_events,
            "cutoff_date": cutoff.date(),
        })
    return pd.DataFrame(rows)


def build_splits(event_log):
    """Partition the event log into feature and label events at the cutoff.

    Returns (features_events, label_events, cutoff_ts). Also asserts the
    config cutoff matches ``max(t_dat) - LABEL_WINDOW_DAYS + 1`` for this data.
    """
    cutoff = cutoff_timestamp()

    # Cross-check the config cutoff against the actual data (computed in code).
    max_d = event_log["t_dat"].max()
    expected = max_d - pd.Timedelta(days=config.LABEL_WINDOW_DAYS) + pd.Timedelta(days=1)
    assert cutoff == expected, (
        f"config.CUTOFF_DATE {cutoff.date()} != max(t_dat)-{config.LABEL_WINDOW_DAYS}+1 "
        f"= {expected.date()}; update config.DATASET_MAX_DATE/LABEL_WINDOW_DAYS"
    )

    features_events = event_log[event_log["t_dat"] < cutoff].reset_index(drop=True)
    label_events = event_log[event_log["t_dat"] >= cutoff].reset_index(drop=True)
    return features_events, label_events, cutoff


def build_labels(label_events) -> pd.DataFrame:
    """Per-customer label target: distinct action_ids bought in the label window.

    Long form: one row per (customer_id, action_id) pair — the set membership.
    """
    labels = (
        label_events[["customer_id", "action_id"]]
        .drop_duplicates()
        .sort_values(["customer_id", "action_id"])
        .reset_index(drop=True)
    )
    labels["customer_id"] = labels["customer_id"].astype("string")
    labels["action_id"] = labels["action_id"].astype("int64")
    return labels


def segment_customers(all_customer_ids, feature_ids, label_ids) -> dict:
    """Segment the full customer base by feature/label presence."""
    all_set = set(all_customer_ids)
    feat = set(feature_ids)
    lab = set(label_ids)
    active = feat | lab

    core = feat & lab                 # history + something to predict
    history_no_label = feat - lab     # history, no label-window purchase
    label_only = lab - feat           # cold-start: only label-window events
    no_events = all_set - active      # in customer master, never purchased

    total = len(all_set)
    def pct(n):
        return 100.0 * n / total if total else 0.0

    return {
        "total_customers": total,
        "core": len(core), "core_pct": pct(len(core)),
        "history_no_label": len(history_no_label), "history_no_label_pct": pct(len(history_no_label)),
        "label_only": len(label_only), "label_only_pct": pct(len(label_only)),
        "no_events": len(no_events), "no_events_pct": pct(len(no_events)),
    }


def verify_no_leakage(event_log, features_events, label_events, cutoff) -> dict:
    """The critical checks: feature/label time boundary is clean, no overlap."""
    feat_max = features_events["t_dat"].max()
    label_min = label_events["t_dat"].min()

    assert feat_max < cutoff, f"feature max {feat_max} is not strictly before cutoff {cutoff}"
    assert label_min >= cutoff, f"label min {label_min} is before cutoff {cutoff}"
    # Clean partition => no row in both sets and none dropped.
    assert len(features_events) + len(label_events) == len(event_log), \
        "feature + label rows do not partition the event log"

    return {
        "feature_max_date": feat_max,
        "label_min_date": label_min,
        "cutoff": cutoff,
        "feature_rows": len(features_events),
        "label_rows": len(label_events),
    }


def save_splits(features_events, labels) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_events.to_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    labels.to_parquet(config.PROCESSED_DIR / "labels.parquet", engine="pyarrow")
