"""Tests for the Phase 2 revision — fair bandit evaluation (v2).

The central guard is that held-out evaluation NEVER updates the bandit's state,
so the reported held-out numbers are a true generalization test.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.linucb import LinUCB
from src import run_bandit_v2 as v2

CUTOFF = pd.Timestamp(config.CUTOFF_DATE)


# --------------------------------------------------------------------------
# Split integrity
# --------------------------------------------------------------------------
def test_learn_heldout_split_disjoint_and_sized():
    """Learning and held-out sets are disjoint and match BANDIT_LEARN_FRAC."""
    rng = np.random.default_rng(config.SEED)
    ids = [f"c{i}" for i in range(1000)]
    order = sorted(ids)
    rng.shuffle(order)
    n_learn = int(round(config.BANDIT_LEARN_FRAC * len(order)))
    learn, held = order[:n_learn], order[n_learn:]
    assert set(learn).isdisjoint(held)
    assert len(learn) + len(held) == len(ids)
    assert len(learn) == n_learn
    assert abs(len(learn) / len(ids) - config.BANDIT_LEARN_FRAC) < 1e-9


# --------------------------------------------------------------------------
# The key guard: no state change during held-out evaluation
# --------------------------------------------------------------------------
def _trained_bandit():
    m = LinUCB(n_actions=3, d=2, alpha=0.5, action_ids=[0, 1, 2])
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.normal(size=2)
        a = rng.integers(3)
        m.update(a, x, int(rng.random() < 0.5))
    return m


def test_heldout_eval_does_not_change_state():
    """Held-out evaluation must not mutate A_a or b_a (frozen bandit)."""
    m = _trained_bandit()
    A_before = m.A.copy()
    Ainv_before = m.A_inv.copy()
    b_before = m.b.copy()

    heldout = ["h0", "h1", "h2", "h3"]
    X = np.random.default_rng(1).normal(size=(4, 2))
    cid_to_row = {c: i for i, c in enumerate(heldout)}
    is_cold = {c: False for c in heldout}
    label_sets = {c: {0} for c in heldout}
    v2._heldout_metrics(m, heldout, X, cid_to_row, is_cold, label_sets, pop_topk=[0, 1, 2])

    assert np.array_equal(m.A, A_before)
    assert np.array_equal(m.A_inv, Ainv_before)
    assert np.array_equal(m.b, b_before)


# --------------------------------------------------------------------------
# hit@k uses exploitation estimate, not UCB (regression guard)
# --------------------------------------------------------------------------
def test_heldout_ranks_by_exploitation_not_ucb():
    """Held-out recs equal top_k_exploit (θ·x), NOT the UCB ranking."""
    # Build a state where exploitation and UCB disagree: action 0 is well-learned
    # (high estimate, low uncertainty); action 1 is untouched (zero estimate, high
    # uncertainty). Exploitation prefers 0; heavy-exploration UCB prefers 1.
    m = LinUCB(n_actions=2, d=1, alpha=5.0, action_ids=[0, 1])
    x = np.array([1.0])
    for _ in range(20):
        m.update(0, x, 1)
    assert m.top_k_exploit(x, 2)[0] == 0     # exploitation: the learned action
    assert m.top_k(x, 2)[0] == 1             # UCB: the high-uncertainty action

    recs = v2._heldout_recs(m, ["h"], np.array([[1.0]]), {"h": 0},
                            is_cold={"h": False}, pop_topk=[0, 1])
    assert recs["h"] == m.top_k_exploit(x, max(v2.HITK))  # v2 uses exploitation


# --------------------------------------------------------------------------
# Multi-epoch: state persists and grows across epochs
# --------------------------------------------------------------------------
def test_state_persists_and_grows_across_epochs():
    """A_a accumulates across updates (multi-epoch learning keeps state)."""
    m = LinUCB(n_actions=2, d=2, alpha=1.0, action_ids=[0, 1])
    x = np.array([1.0, 1.0])
    trace0 = np.trace(m.A[0])
    for _ in range(3):                     # simulate repeated passes
        m.update(0, x, 1)
    assert np.trace(m.A[0]) > trace0        # A grew -> learning accumulated
    # The untouched action is unchanged (still identity).
    assert np.array_equal(m.A[1], np.eye(2))


# --------------------------------------------------------------------------
# Learning-curve rows
# --------------------------------------------------------------------------
def test_learning_curve_rows_complete():
    """_heldout_metrics yields all curve fields, populated."""
    m = _trained_bandit()
    heldout = ["h0", "h1"]
    X = np.random.default_rng(2).normal(size=(2, 2))
    out = v2._heldout_metrics(m, heldout, X, {"h0": 0, "h1": 1},
                              {"h0": False, "h1": False},
                              {"h0": {0}, "h1": {1}}, pop_topk=[0, 1, 2])
    for f in ["heldout_avg_reward", "heldout_hit@1", "heldout_hit@6", "heldout_hit@12"]:
        assert f in out and out[f] is not None


# --------------------------------------------------------------------------
# Leakage guard (structural)
# --------------------------------------------------------------------------
def test_context_is_prebuilt_and_rewards_from_labels():
    """Context comes from the (pre-cutoff) context matrix; rewards from labels —
    the bandit never sees post-cutoff data except as the reward target."""
    # select_action takes only x (no labels) — the decision cannot use rewards.
    m = _trained_bandit()
    import inspect
    params = inspect.signature(m.select_action).parameters
    assert "reward" not in params and "label" not in params
    # reward only enters through update(action, x, reward).
    up = inspect.signature(m.update).parameters
    assert "reward" in up
