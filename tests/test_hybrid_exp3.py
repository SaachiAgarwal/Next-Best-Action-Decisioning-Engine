"""Tests for Experiment 3 — the recency + frequency weighted hybrid.

Synthetic feature-side events (all pre-cutoff) let us check the recency/frequency
math and the popularity-floor property exactly. Data-backed tests guard that
Experiment A / B artifacts are intact.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.hybrid import HybridModel, _row_minmax
from src.models.popularity import PopularityModel


CUTOFF = pd.Timestamp(config.CUTOFF_DATE)
ACTIONS = np.arange(5)  # tiny 5-action space for the synthetic tests


def _events(rows):
    """rows: list of (customer_id, action_id, t_dat)."""
    return pd.DataFrame({
        "customer_id": pd.array([r[0] for r in rows], dtype="string"),
        "action_id": [r[1] for r in rows],
        "t_dat": [r[2] for r in rows],
    })


def _fit(rows, reference=CUTOFF):
    return HybridModel().fit(_events(rows), reference_date=reference, action_ids=ACTIONS)


# --------------------------------------------------------------------------
# Recency
# --------------------------------------------------------------------------
def test_recency_weight_halves_at_one_half_life():
    """Recency ~1 at the reference and ~0.5 one half-life before it."""
    hl = config.HALF_LIFE_DAYS
    rows = [
        ("recent", 0, CUTOFF - pd.Timedelta(days=1)),    # ~ at reference
        ("old", 1, CUTOFF - pd.Timedelta(days=hl)),       # one half-life before
    ]
    m = _fit(rows)
    # personal = log(1+count) * recency; count=1 -> log(2) factor cancels in ratio.
    w_recent = m.personal[m.customer_index["recent"], m.action_index[0]] / np.log(2)
    w_old = m.personal[m.customer_index["old"], m.action_index[1]] / np.log(2)
    assert w_recent == pytest.approx(0.5 ** (1 / hl), abs=1e-3)  # ~1.0
    assert w_old == pytest.approx(0.5, abs=1e-3)                  # exactly half
    assert w_recent > w_old  # decreases with age


# --------------------------------------------------------------------------
# Frequency damping
# --------------------------------------------------------------------------
def test_log_damped_frequency():
    """10 purchases yields more than 1 but far less than 10x (log damping)."""
    day = CUTOFF - pd.Timedelta(days=1)
    rows = [("c10", 0, day)] * 10 + [("c1", 1, day)]
    m = _fit(rows)
    s10 = m.personal[m.customer_index["c10"], m.action_index[0]]
    s1 = m.personal[m.customer_index["c1"], m.action_index[1]]
    assert s10 > s1                     # more purchases -> higher
    assert s10 < 10 * s1                # but nowhere near 10x
    assert s10 == pytest.approx(np.log(11) / np.log(2) * s1, rel=1e-3)  # log(11) vs log(2)


# --------------------------------------------------------------------------
# Popularity floor
# --------------------------------------------------------------------------
def test_gamma_only_reproduces_popularity_ranking():
    """alpha=beta=0, gamma=1 reproduces the popularity ranking exactly.

    All canonical actions appear in the events (as they do over the real 128
    actions), so popularity ranks the full action set and the two rankings match.
    """
    day = CUTOFF - pd.Timedelta(days=1)
    rows = ([("a", 0, day)] * 4 + [("a", 1, day)] * 3 + [("a", 2, day)] * 2
            + [("a", 3, day)] + [("a", 4, day)] + [("b", 2, day)])
    m = _fit(rows)
    pop = PopularityModel().fit(_events(rows))
    assert len(pop.ranked_actions) == len(ACTIONS)  # all actions present
    for c in ["a", "b"]:
        assert m.recommend(c, k=5, alpha=0, beta=0, gamma=1) == pop.recommend(k=5)


def test_recommend_returns_k_and_cold_start_is_popularity():
    day = CUTOFF - pd.Timedelta(days=1)
    rows = [("a", 0, day), ("a", 1, day), ("b", 2, day)]
    m = _fit(rows)
    assert len(m.recommend("a", k=3)) == 3
    cold = m.recommend("ghost", k=3)
    assert cold == m.popularity_model.recommend(k=3)


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
def test_fit_rejects_events_on_or_after_reference():
    """fit asserts every event predates the reference date (no leakage)."""
    rows = [("a", 0, CUTOFF)]  # exactly at reference -> not allowed
    with pytest.raises(AssertionError):
        _fit(rows)


# --------------------------------------------------------------------------
# Divergent-slice determinism
# --------------------------------------------------------------------------
def test_divergent_selection_deterministic():
    """The divergent-customer selection is reproducible run to run."""
    from src.models.run_hybrid_exp3 import divergent_customers
    if not (config.PROCESSED_DIR / "features_events.parquet").exists():
        pytest.skip("features_events.parquet not built yet")
    from src.models.popularity import load_feature_events
    from src.eval import evaluable
    fe = load_feature_events()
    evaluable_ids, _ = evaluable.get_evaluable()
    model = HybridModel().fit(fe, reference_date=config.CUTOFF_DATE)
    d1, t1 = divergent_customers(fe, evaluable_ids, model)
    d2, t2 = divergent_customers(fe, evaluable_ids, model)
    assert t1 == t2
    assert list(d1["customer_id"]) == list(d2["customer_id"])


# --------------------------------------------------------------------------
# Experiment A / B artifacts intact
# --------------------------------------------------------------------------
def test_experiment_a_artifacts_intact():
    """Guard: product-type action space still has exactly 128 actions."""
    path = config.PROCESSED_DIR / "actions.parquet"
    if not path.exists():
        pytest.skip("actions.parquet not built yet")
    actions = pd.read_parquet(path, engine="pyarrow")
    assert len(actions) == 128


def test_experiment_b_artifacts_intact():
    """Guard: article-level labels untouched (built in Experiment B)."""
    path = config.PROCESSED_DIR / "labels_article.parquet"
    if not path.exists():
        pytest.skip("labels_article.parquet not built yet")
    la = pd.read_parquet(path, columns=["customer_id"], engine="pyarrow")
    assert la["customer_id"].nunique() == 15246
