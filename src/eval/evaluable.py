"""The evaluable customer set and label lookup for evaluation.

The **evaluable / core set** is customers who have at least one label-window
purchase *and* pre-cutoff feature history — the Week 1 "core" segment (~15,246).
Requiring history keeps the set identical across models: personalized models
later need history to score a customer, so the popularity baseline is evaluated
on the same customers for an apples-to-apples comparison. Cold-start (label-only)
customers are excluded here and handled separately as the fallback scenario.

Note: ``labels.parquet`` itself lists every customer with >=1 label action
(~16,895, including 1,649 cold-start). ``evaluable_customers()`` intersects that
with feature-side customers to recover the core set.
"""

from __future__ import annotations

import pandas as pd

from src import config

# The Week 1 core segment size; the evaluable count should reproduce this.
EXPECTED_CORE = 15246
_CORE_TOLERANCE = 300  # allow small drift; fail loudly if far off


def load_labels() -> pd.DataFrame:
    """Load labels.parquet (customer_id, action_id pairs in the label window)."""
    return pd.read_parquet(config.PROCESSED_DIR / "labels.parquet", engine="pyarrow")


def load_label_sets() -> dict:
    """All label customers as customer_id -> set(action_id) (includes cold-start)."""
    labels = load_labels()
    return {
        cid: set(grp)
        for cid, grp in labels.groupby("customer_id", sort=False)["action_id"]
    }


def _feature_customer_ids() -> set:
    feats = pd.read_parquet(
        config.PROCESSED_DIR / "features_events.parquet", columns=["customer_id"],
        engine="pyarrow",
    )
    return set(feats["customer_id"].unique())


def get_evaluable(assert_count: bool = True):
    """Return (evaluable_ids, label_sets) for the core evaluable set.

    evaluable_ids : set of customer_ids with >=1 label action AND feature history
    label_sets    : customer_id -> set(action_id), restricted to evaluable_ids
    """
    all_label_sets = load_label_sets()
    feature_ids = _feature_customer_ids()
    evaluable_ids = set(all_label_sets) & feature_ids
    label_sets = {cid: all_label_sets[cid] for cid in evaluable_ids}

    if assert_count:
        n = len(evaluable_ids)
        assert abs(n - EXPECTED_CORE) <= _CORE_TOLERANCE, (
            f"evaluable core count {n:,} is not near the Week 1 core {EXPECTED_CORE:,}"
        )
    return evaluable_ids, label_sets
