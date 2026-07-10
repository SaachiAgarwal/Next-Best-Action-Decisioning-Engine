"""Tests for the article-level granularity experiment (Experiment B).

Includes a guard that the product-type artifacts (Experiment A) are untouched.
Synthetic article_ids carry leading zeros to lock in string handling — the bug
that would silently coerce them to int (dropping zeros) must never return.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import config
from src.models.popularity_article import ArticlePopularityModel
from src.models.item_cf_article import ArticleItemCF


# Article ids with leading zeros (as in the real H&M data).
A0, A1, A2, A3, A4 = "0000000", "0000001", "0000002", "0000003", "0000004"


def _synthetic_feature_events():
    """A0 popular; {A1,A2} and {A3,A4} are niche co-purchase pairs."""
    baskets = {
        "c1": [A0, A1, A2], "c2": [A0, A1, A2], "c5": [A0, A1, A2],
        "c3": [A0, A3, A4], "c4": [A0, A3, A4], "c6": [A0, A3, A4],
    }
    before = pd.Timestamp(config.CUTOFF_DATE) - pd.Timedelta(days=1)
    rows = [(c, a) for c, acts in baskets.items() for a in acts]
    return pd.DataFrame({
        "customer_id": pd.array([c for c, _ in rows], dtype="string"),
        "article_id": pd.array([a for _, a in rows], dtype="string"),
        "t_dat": [before] * len(rows),
    })


@pytest.fixture
def model():
    return ArticleItemCF().fit(_synthetic_feature_events(), verbose=False)


# --------------------------------------------------------------------------
# Sparse structure
# --------------------------------------------------------------------------
def test_similarity_is_sparse_and_self_zeroed(model):
    """Similarity is a scipy.sparse matrix (never dense) with a zero diagonal."""
    assert sp.issparse(model.similarity)
    assert model.similarity.diagonal().max() == 0.0


def test_article_ids_stay_strings_with_leading_zeros(model):
    """Recommendations are string article_ids with leading zeros preserved."""
    recs = model.recommend("c1", k=2, include_repeats=False)
    assert all(isinstance(a, str) for a in recs)
    assert set(recs) == {A3, A4}  # niche complement of c1's basket, zeros intact


# --------------------------------------------------------------------------
# Recommendation behavior
# --------------------------------------------------------------------------
def test_recommend_returns_k_and_excludes_bought(model):
    """recommend() returns exactly k and excludes already-bought (no repeats)."""
    recs = model.recommend("c1", k=2, include_repeats=False)  # c1 bought A0,A1,A2
    assert len(recs) == 2
    assert set(recs).isdisjoint({A0, A1, A2})


def test_cold_start_falls_back_to_article_popularity(model):
    """A customer with no history gets the article-popularity top-k."""
    cold = model.recommend("ghost", k=2)
    assert cold == model.popularity_model.recommend(k=2)
    assert cold[0] == A0  # most popular article in the synthetic data


def test_recommend_all_matches_recommend(model):
    """Batched recommend_all matches per-customer recommend."""
    ids = ["c1", "c3", "ghost"]
    batched = model.recommend_all(ids, k=2, include_repeats=False)
    for c in ids:
        assert batched[c] == model.recommend(c, k=2, include_repeats=False)


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------
def test_popularity_fit_rejects_label_window_events():
    on_cutoff = pd.Timestamp(config.CUTOFF_DATE)
    bad = pd.DataFrame({
        "customer_id": pd.array(["c1"], dtype="string"),
        "article_id": pd.array([A0], dtype="string"),
        "t_dat": [on_cutoff],
    })
    with pytest.raises(AssertionError):
        ArticlePopularityModel().fit(bad)


def test_itemcf_fit_rejects_label_window_events():
    on_cutoff = pd.Timestamp(config.CUTOFF_DATE)
    bad = pd.DataFrame({
        "customer_id": pd.array(["c1"], dtype="string"),
        "article_id": pd.array([A0], dtype="string"),
        "t_dat": [on_cutoff],
    })
    with pytest.raises(AssertionError):
        ArticleItemCF().fit(bad, verbose=False)


# --------------------------------------------------------------------------
# Cross-experiment consistency (data-backed)
# --------------------------------------------------------------------------
def test_evaluable_set_identical_to_experiment_a():
    """Article experiment uses the same 15,246 core customers as Experiment A."""
    from src.eval import evaluable
    if not (config.PROCESSED_DIR / "labels.parquet").exists():
        pytest.skip("labels.parquet not built yet")
    evaluable_ids, _ = evaluable.get_evaluable()
    assert 15000 <= len(evaluable_ids) <= 15500
    la_path = config.PROCESSED_DIR / "labels_article.parquet"
    if la_path.exists():
        la = pd.read_parquet(la_path, columns=["customer_id"], engine="pyarrow")
        assert set(la["customer_id"].unique()) == set(evaluable_ids)


def test_product_type_artifacts_untouched():
    """Guard: the product-type action space still has exactly 128 actions."""
    path = config.PROCESSED_DIR / "actions.parquet"
    if not path.exists():
        pytest.skip("actions.parquet not built yet")
    actions = pd.read_parquet(path, engine="pyarrow")
    assert len(actions) == 128
    assert actions["action_id"].nunique() == 128
