"""Tests for the evaluation harness and the popularity baseline.

The metric tests use tiny synthetic cases with known answers. The data-backed
tests (evaluable count, real recommend) skip if the Week 1/2 parquet outputs
have not been built yet.
"""

import pandas as pd
import pytest

from src import config
from src.eval import metrics
from src.models.popularity import PopularityModel


# --------------------------------------------------------------------------
# Metric correctness on synthetic cases
# --------------------------------------------------------------------------
def test_hit_rate_hits_and_misses():
    """1.0 when a label is in top-k, 0.0 when not; mean across customers."""
    recs = {"a": [1, 2, 3], "b": [1, 2, 3]}
    labels = {"a": {1}, "b": {9}}  # a hits, b misses
    assert metrics.hit_rate_at_k(recs, labels, 3) == 0.5
    # Single customer whose label is present -> 1.0
    assert metrics.hit_rate_at_k({"a": [1, 2, 3]}, {"a": {2}}, 3) == 1.0
    # Single customer whose label is absent -> 0.0
    assert metrics.hit_rate_at_k({"a": [1, 2, 3]}, {"a": {8}}, 3) == 0.0


def test_hit_rate_respects_k_cutoff():
    """A label sitting beyond position k does not count as a hit."""
    recs = {"a": [1, 2, 3, 4]}
    labels = {"a": {4}}
    assert metrics.hit_rate_at_k(recs, labels, 3) == 0.0  # 4 is at rank 4, k=3
    assert metrics.hit_rate_at_k(recs, labels, 4) == 1.0


def test_recall_fraction():
    """recall@k = fraction of label actions found in the top-k."""
    recs = {"a": [1, 3, 4]}
    labels = {"a": {1, 2}}  # 1 found, 2 not -> 1/2
    assert metrics.recall_at_k(recs, labels, 3) == 0.5
    # Two customers: 1/2 and 2/2 -> mean 0.75
    recs2 = {"a": [1, 3, 4], "b": [5, 6, 7]}
    labels2 = {"a": {1, 2}, "b": {5, 6}}
    assert metrics.recall_at_k(recs2, labels2, 3) == pytest.approx(0.75)


def test_precision_fraction():
    """precision@k = fraction of the k recs that were purchased."""
    recs = {"a": [1, 2, 3, 4]}
    labels = {"a": {1, 2}}  # 2 of top-4 correct -> 0.5
    assert metrics.precision_at_k(recs, labels, 4) == pytest.approx(0.5)


def test_evaluate_table_shape():
    """evaluate() returns one row per k with the three metric columns."""
    recs = {"a": [1, 2, 3]}
    labels = {"a": {1}}
    out = metrics.evaluate(recs, labels, ks=[1, 3])
    assert list(out["k"]) == [1, 3]
    assert set(out.columns) == {"k", "hit_rate", "recall", "precision"}


# --------------------------------------------------------------------------
# Popularity model
# --------------------------------------------------------------------------
def _synthetic_feature_events():
    """Small feature-side event log, all before the cutoff."""
    before = pd.Timestamp(config.CUTOFF_DATE) - pd.Timedelta(days=1)
    return pd.DataFrame({
        "customer_id": pd.array(["c1", "c1", "c2", "c3", "c2"], dtype="string"),
        "action_id": [10, 10, 10, 20, 30],  # 10 most popular, then 20, 30
        "t_dat": [before] * 5,
    })


def test_recommend_returns_exactly_k():
    """recommend() returns exactly k actions, most-popular first."""
    model = PopularityModel().fit(_synthetic_feature_events())
    assert model.recommend(k=2) == [10, 20]  # 10 has 3 buys, tie 20/30 -> action_id
    assert len(model.recommend(k=3)) == 3


def test_recommend_is_non_personalized_and_ignores_labels():
    """Every customer (incl. cold-start) gets the same list; no label data used."""
    model = PopularityModel().fit(_synthetic_feature_events())
    known = model.recommend("c1", k=3)
    cold = model.recommend("brand_new_customer", k=3)
    assert known == cold  # identical regardless of customer / history
    recs = model.recommend_all(["c1", "cold"], k=2)
    assert recs["c1"] == recs["cold"] == [10, 20]


def test_fit_rejects_label_window_events():
    """Fitting on events on/after the cutoff raises — guards against leakage."""
    on_cutoff = pd.Timestamp(config.CUTOFF_DATE)
    bad = pd.DataFrame({
        "customer_id": pd.array(["c1"], dtype="string"),
        "action_id": [10],
        "t_dat": [on_cutoff],
    })
    with pytest.raises(AssertionError):
        PopularityModel().fit(bad)


# --------------------------------------------------------------------------
# Evaluable set (data-backed)
# --------------------------------------------------------------------------
def test_evaluable_count_in_expected_range():
    """The evaluable core set reproduces the Week 1 core (~15,246)."""
    from src.eval import evaluable
    path = config.PROCESSED_DIR / "labels.parquet"
    if not path.exists():
        pytest.skip("labels.parquet not built yet — run the Week 1 split pipeline")
    evaluable_ids, label_sets = evaluable.get_evaluable()
    assert 15000 <= len(evaluable_ids) <= 15500
    assert len(label_sets) == len(evaluable_ids)
    # Every evaluable customer has at least one label action.
    assert all(len(s) >= 1 for s in label_sets.values())
