r"""LinUCB with customer x action features (Phase 2 v3) — the standard LinUCB.

v2 conditioned on a **customer-only** context vector, so an action's score could
not depend on the customer's affinity for *that specific action* — the bandit had
no way to encode "this customer buys trousers constantly." That is exactly the
signal Exp 3 used to beat popularity, so the v2 result was confounded: it could
not separate "contextual bandits don't help here" from "the bandit was starved of
the predictive features."

v3 uses the standard LinUCB formulation: the feature vector describes the
**(customer, action) pair**. Each action a is scored on its own feature vector
x_a = [customer state | action-specific signals for a | bias], where the
action-specific signals are Exp 3's personal recency×log-frequency affinity,
recency-weighted CF score, and normalized action popularity.

Disjoint per-action models (A_a, b_a) as in v2, d = feature dim:
    p_a = theta_a . x_a  +  alpha * sqrt( x_a . A_a^-1 . x_a )
    choose argmax_a p_a ;  update: A_a += x_a x_a^T, b_a += r x_a   (Sherman-Morrison)

The one API change from v2's LinUCB is that ``select_action`` / ``top_k_exploit``
take a per-action feature MATRIX (n_actions x d, one row per action) instead of a
single shared context vector. The learning math is identical.
"""

from __future__ import annotations

import numpy as np

from src import config


class LinUCBv3:
    def __init__(self, n_actions, d, alpha=None, action_ids=None):
        self.n_actions = int(n_actions)
        self.d = int(d)
        self.alpha = float(config.BANDIT_ALPHA if alpha is None else alpha)
        self.action_ids = list(action_ids) if action_ids is not None else list(range(n_actions))
        self._index = {a: i for i, a in enumerate(self.action_ids)}

        eye = np.eye(self.d, dtype=np.float64)
        self.A = np.repeat(eye[None], self.n_actions, axis=0)
        self.A_inv = np.repeat(eye[None], self.n_actions, axis=0)
        self.b = np.zeros((self.n_actions, self.d), dtype=np.float64)

    # -- scoring (per-action feature matrix Xc: (n_actions, d)) --------------
    def scores(self, Xc: np.ndarray):
        Xc = np.asarray(Xc, dtype=np.float64)
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)   # (n, d)
        reward_est = np.einsum("ai,ai->a", theta, Xc)          # theta_a . x_a
        Ainv_x = np.einsum("aij,aj->ai", self.A_inv, Xc)       # A_a^-1 x_a
        quad = np.einsum("ai,ai->a", Ainv_x, Xc)               # x_a A_a^-1 x_a
        bonus = self.alpha * np.sqrt(np.maximum(quad, 0.0))
        return reward_est + bonus, reward_est, bonus

    def select_action(self, Xc):
        ucb, reward_est, bonus = self.scores(Xc)
        i = int(np.argmax(ucb))
        return self.action_ids[i], float(reward_est[i]), float(bonus[i]), ucb

    def top_k_exploit(self, Xc, k):
        """Top-k action_ids by exploitation estimate theta_a . x_a (no bonus)."""
        _, reward_est, _ = self.scores(Xc)
        order = np.argsort(-reward_est, kind="stable")[:k]
        return [self.action_ids[i] for i in order]

    def top_k(self, Xc, k):
        ucb, _, _ = self.scores(Xc)
        order = np.argsort(-ucb, kind="stable")[:k]
        return [self.action_ids[i] for i in order]

    # -- learning ------------------------------------------------------------
    def update(self, action_id, x_a, reward):
        """Rank-1 update for the chosen action's model (Sherman-Morrison)."""
        i = self._index[action_id]
        x = np.asarray(x_a, dtype=np.float64).ravel()
        self.A[i] += np.outer(x, x)
        Ainv = self.A_inv[i]
        Ainv_x = Ainv @ x
        denom = 1.0 + float(x @ Ainv_x)
        self.A_inv[i] = Ainv - np.outer(Ainv_x, Ainv_x) / denom
        self.b[i] += reward * x

    def reward_estimate(self, action_id, x_a):
        i = self._index[action_id]
        x = np.asarray(x_a, dtype=np.float64).ravel()
        return float((self.A_inv[i] @ self.b[i]) @ x)

    def uncertainty(self, action_id, x_a):
        i = self._index[action_id]
        x = np.asarray(x_a, dtype=np.float64).ravel()
        return float(self.alpha * np.sqrt(max(x @ self.A_inv[i] @ x, 0.0)))
