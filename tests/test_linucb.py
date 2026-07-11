"""Tests for the Phase 2 LinUCB contextual bandit and its replay/audit behavior."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.linucb import LinUCB
from src.run_bandit import _replay


# --------------------------------------------------------------------------
# LinUCB update math
# --------------------------------------------------------------------------
def test_update_moves_estimate_toward_reward():
    """After updating an action with (x, reward=1), its estimate for x rises toward 1."""
    m = LinUCB(n_actions=2, d=2, alpha=1.0, action_ids=[0, 1])
    x = np.array([1.0, 0.0])
    assert m.reward_estimate(0, x) == 0.0     # starts at 0 (b=0)
    m.update(0, x, reward=1)
    e1 = m.reward_estimate(0, x)
    assert 0.0 < e1 < 1.0                      # moved toward the reward
    m.update(0, x, reward=1)
    e2 = m.reward_estimate(0, x)
    assert e2 > e1                              # keeps moving toward 1 with more evidence


def test_uncertainty_shrinks_as_action_selected():
    """The uncertainty bonus for an action shrinks as it accrues observations."""
    m = LinUCB(n_actions=2, d=2, alpha=1.0, action_ids=[0, 1])
    x = np.array([1.0, 0.0])
    u0 = m.uncertainty(0, x)
    m.update(0, x, reward=0)
    u1 = m.uncertainty(0, x)
    assert u1 < u0                              # A_a grew -> A_a^-1 shrank
    # An untouched action keeps its higher uncertainty.
    assert m.uncertainty(1, x) > u1


def test_select_action_valid_and_reproducible():
    """select_action returns a valid action; scores are deterministic given state."""
    m = LinUCB(n_actions=3, d=2, alpha=0.7, action_ids=[5, 6, 7])
    x = np.array([0.3, -0.8])
    chosen, est, bonus, scores = m.select_action(x)
    assert chosen in [5, 6, 7]
    assert scores.shape == (3,)
    # Deterministic: same state + x -> identical scores.
    s2 = m.scores(x)[0]
    assert np.array_equal(scores, s2)


# --------------------------------------------------------------------------
# Replay: reward rule, cold-start fallback, audit log
# --------------------------------------------------------------------------
def _tiny_replay(label_sets, is_cold, alpha=0.5):
    action_ids = [10, 20]
    name_by_id = {10: "a", 20: "b"}
    order = ["c1", "c2", "c3"]
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cid_to_row = {"c1": 0, "c2": 1, "c3": 2}
    return _replay(order, X, cid_to_row, label_sets, action_ids, name_by_id,
                   top_action=10, is_cold=is_cold, alpha=alpha, log=True)


def test_reward_is_one_only_when_chosen_in_labels():
    label_sets = {"c1": {20}, "c2": {10}, "c3": {10}}
    is_cold = {"c1": False, "c2": False, "c3": False}
    run = _tiny_replay(label_sets, is_cold)
    for row in run["log"]:
        expected = 1 if row["chosen_action_id"] in label_sets[row["customer_id"]] else 0
        assert row["reward_observed"] == expected


def test_cold_start_triggers_flagged_popularity_fallback():
    label_sets = {"c1": {20}, "c2": {10}, "c3": {10}}
    is_cold = {"c1": False, "c2": False, "c3": True}
    run = _tiny_replay(label_sets, is_cold)
    log = {r["customer_id"]: r for r in run["log"]}
    assert log["c3"]["is_cold_start_fallback"] is True
    assert log["c3"]["chosen_action_id"] == 10     # top_action fallback
    assert log["c1"]["is_cold_start_fallback"] is False


def test_decision_log_one_row_per_customer_all_fields():
    label_sets = {"c1": {20}, "c2": {10}, "c3": {10}}
    is_cold = {"c1": False, "c2": False, "c3": False}
    run = _tiny_replay(label_sets, is_cold)
    assert len(run["log"]) == 3
    required = {"customer_id", "chosen_action_id", "chosen_action_name",
                "reward_estimate", "uncertainty_bonus", "ucb_score",
                "reward_observed", "is_cold_start_fallback", "top3_actions_with_scores"}
    for row in run["log"]:
        assert required <= set(row.keys())
        assert all(row[f] is not None for f in required)


def test_first_decision_independent_of_labels_leakage_guard():
    """The bandit chooses BEFORE seeing reward: the first pick is identical
    regardless of the labels (rewards can't leak into the decision)."""
    is_cold = {"c1": False, "c2": False, "c3": False}
    run_a = _tiny_replay({"c1": {10}, "c2": {10}, "c3": {10}}, is_cold)
    run_b = _tiny_replay({"c1": {20}, "c2": {20}, "c3": {20}}, is_cold)
    # First customer's chosen action must match — no reward seen yet at t=0.
    assert run_a["log"][0]["chosen_action_id"] == run_b["log"][0]["chosen_action_id"]
