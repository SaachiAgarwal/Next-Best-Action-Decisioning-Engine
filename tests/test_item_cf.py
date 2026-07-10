"""Tests for item-to-item collaborative filtering.

Uses a small synthetic feature-side log with a deliberately *popular* action and
two niche co-purchase pairs, so we can check that cosine normalization prevents
the model from collapsing into popularity.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.item_cf import ItemCF


def _synthetic_feature_events():
    """Action 0 is bought by everyone (popular); {1,2} and {3,4} are niche pairs."""
    baskets = {
        "c1": [0, 1, 2],
        "c2": [0, 1, 2],
        "c3": [0, 3, 4],
        "c4": [0, 3, 4],
        "c5": [0, 1, 2],
        "c6": [0, 3, 4],
    }
    before = pd.Timestamp(config.CUTOFF_DATE) - pd.Timedelta(days=1)
    rows = [(c, a) for c, acts in baskets.items() for a in acts]
    return pd.DataFrame({
        "customer_id": pd.array([c for c, _ in rows], dtype="string"),
        "action_id": [a for _, a in rows],
        "t_dat": [before] * len(rows),
    })


@pytest.fixture
def model():
    return ItemCF().fit(_synthetic_feature_events())


def test_cooccurrence_symmetric_and_self_similarity_zeroed(model):
    """Co-occurrence is symmetric and the similarity diagonal is zeroed."""
    assert np.allclose(model.cooccurrence, model.cooccurrence.T)
    assert np.allclose(np.diagonal(model.similarity), 0.0)


def test_normalization_prevents_popularity_collapse(model):
    """The popular action (0) is not the top neighbor of every other action."""
    popular = 0
    for aid in model.action_ids:
        if aid == popular:
            continue
        top_neighbor = model.similar_actions(aid, 1)[0][0]
        assert top_neighbor != popular, (
            f"popular action {popular} is the top neighbor of {aid} — "
            "similarity collapsed into popularity"
        )
    # Concretely: action 1's top neighbor is its niche pair (2), not the popular 0.
    assert model.similar_actions(1, 1)[0][0] == 2


def test_recommend_returns_k_and_excludes_bought(model):
    """recommend() returns exactly k and excludes already-bought (no repeats)."""
    recs = model.recommend("c1", k=2, include_repeats=False)  # c1 bought {0,1,2}
    assert len(recs) == 2
    assert set(recs).isdisjoint({0, 1, 2})       # excluded already-bought
    assert set(recs) == {3, 4}                    # the niche complement


def test_include_repeats_allows_previously_bought(model):
    """With include_repeats=True, already-bought actions may be recommended."""
    recs = model.recommend("c1", k=4, include_repeats=True)
    assert len(recs) == 4
    assert set(recs) & {0, 1, 2}  # at least one prior purchase can resurface


def test_cold_start_falls_back_to_popularity(model):
    """A customer with no history gets the popularity top-k."""
    cold = model.recommend("ghost_customer", k=2)
    assert cold == model.popularity_model.recommend(k=2)
    assert cold[0] == 0  # action 0 is the most popular in the synthetic data


def test_fit_rejects_label_window_events():
    """Fitting on events on/after the cutoff raises — leakage guard."""
    on_cutoff = pd.Timestamp(config.CUTOFF_DATE)
    bad = pd.DataFrame({
        "customer_id": pd.array(["c1"], dtype="string"),
        "action_id": [0],
        "t_dat": [on_cutoff],
    })
    with pytest.raises(AssertionError):
        ItemCF().fit(bad)
