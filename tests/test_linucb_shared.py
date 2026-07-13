"""Tests for Phase 2c — shared-model LinUCB at article level.

The central test is the shared model's superpower: an article never seen during
learning still receives a finite score, because theta acts on its features.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.linucb_shared import SharedLinUCB


# --------------------------------------------------------------------------
# Shared model: ONE A/b, not per-arm
# --------------------------------------------------------------------------
def test_maintains_single_shared_model():
    d = 10
    m = SharedLinUCB(d=d, alpha=1.0)
    assert m.A.shape == (d, d)        # ONE d x d matrix (not n_actions x d x d)
    assert m.b.shape == (d,)
    assert m.A_inv.shape == (d, d)


def test_unseen_article_still_gets_a_finite_score():
    """The key property: score_one works for ANY feature vector, incl. an article
    never seen in training (a disjoint per-arm model could not score it)."""
    d = 6
    m = SharedLinUCB(d=d, alpha=1.0)
    # Train on some (customer, article) vectors.
    rng = np.random.default_rng(0)
    for _ in range(30):
        m.update(rng.normal(size=d), int(rng.random() < 0.2))
    # A brand-new article's feature vector (content-only, MF dims zero) — never seen.
    x_unseen = np.array([0.3, -0.1, 0.0, 0.0, 0.7, 1.0])
    s = m.score_one(x_unseen)
    assert np.isfinite(s)             # scored purely from features


# --------------------------------------------------------------------------
# Scoring / selection over candidates
# --------------------------------------------------------------------------
def test_select_over_candidate_matrix():
    d = 5
    m = SharedLinUCB(d=d, alpha=0.5)
    X = np.random.default_rng(1).normal(size=(20, d))   # 20 candidates
    i, est, bonus, ucb = m.select_index(X)
    assert 0 <= i < 20
    assert ucb.shape == (20,)
    assert np.array_equal(ucb, m.scores(X)[0])          # deterministic


def test_top_k_exploit_excludes_uncertainty_bonus():
    """hit@k ranking uses theta.x; UCB adds the bonus and can differ (regression guard)."""
    d = 2
    m = SharedLinUCB(d=d, alpha=10.0)
    # Learn a strong direction so exploitation is decisive; keep candidates with
    # different norms so the UCB bonus reorders them.
    for _ in range(30):
        m.update(np.array([1.0, 0.0]), 1)
    X = np.array([[1.0, 0.0], [0.0, 3.0]])   # cand 0 = learned dir; cand 1 = high-norm/uncertain
    exploit = list(m.top_k_exploit_index(X, 2))
    ucb_order = list(np.argsort(-m.scores(X)[0]))
    assert exploit[0] == 0                    # exploitation prefers the learned candidate
    assert ucb_order[0] == 1                  # UCB prefers the high-uncertainty candidate


# --------------------------------------------------------------------------
# Held-out evaluation must not mutate state; disjointness of splits
# --------------------------------------------------------------------------
def test_scoring_does_not_mutate_state():
    d = 4
    m = SharedLinUCB(d=d, alpha=1.0)
    for _ in range(10):
        m.update(np.random.default_rng(2).normal(size=d), 1)
    A0, Ainv0, b0 = m.A.copy(), m.A_inv.copy(), m.b.copy()
    X = np.random.default_rng(3).normal(size=(50, d))
    m.scores(X); m.select_index(X); m.top_k_exploit_index(X, 5)   # read-only ops
    assert np.array_equal(m.A, A0) and np.array_equal(m.A_inv, Ainv0) and np.array_equal(m.b, b0)


def test_learn_heldout_split_disjoint():
    rng = np.random.default_rng(config.SEED)
    ids = [f"c{i}" for i in range(500)]
    order = sorted(ids); rng.shuffle(order)
    n_learn = int(round(config.BANDIT_LEARN_FRAC * len(order)))
    learn, held = order[:n_learn], order[n_learn:]
    assert set(learn).isdisjoint(held)
    assert len(learn) + len(held) == len(ids)


def test_update_changes_state_sherman_morrison():
    """update() applies a rank-1 change; A_inv stays the true inverse of A."""
    d = 4
    m = SharedLinUCB(d=d, alpha=1.0)
    x = np.array([0.5, -0.2, 1.0, 0.3])
    m.update(x, 1)
    assert np.allclose(m.A_inv @ m.A, np.eye(d), atol=1e-8)   # incremental inverse correct


# --------------------------------------------------------------------------
# Config / spec
# --------------------------------------------------------------------------
def test_n_candidates_and_feature_dim_spec():
    """N_CANDIDATES configured; joint feature dim = context + MF(64) + contentSVD(24)
    + 4 affinity + bias (checked structurally via the model accepting d)."""
    assert config.N_CANDIDATES == 100
    d = 24 + 64 + 24 + 4 + 1     # the documented composition
    m = SharedLinUCB(d=d, alpha=1.0)
    assert m.d == d


# --------------------------------------------------------------------------
# Prior artifacts intact
# --------------------------------------------------------------------------
def test_prior_experiment_artifacts_intact():
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
    lap = config.PROCESSED_DIR / "labels_article.parquet"
    if lap.exists():
        assert pd.read_parquet(lap, columns=["customer_id"], engine="pyarrow")["customer_id"].nunique() == 15246
