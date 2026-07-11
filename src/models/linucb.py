r"""LinUCB contextual bandit (disjoint linear models) — Phase 2, the NBA core.

Algorithm: LinUCB with disjoint models (Li et al. 2010, "A Contextual-Bandit
Approach to Personalized News Article Recommendation"). One independent linear
reward model per action; each decision conditions on the customer's context.

Why LinUCB over Thompson Sampling: LinUCB's confidence term is **deterministic**,
so every decision has a reproducible, inspectable "why" (a point reward estimate
plus an explicit uncertainty bonus). That auditable decision trail is the
Responsible-AI requirement here, which we value over Thompson Sampling's random
posterior draws.

Per action a, maintain:
    A_a = d x d matrix (init identity),  b_a = d-vector (init zeros)
For a context vector x (d features):
    theta_a = A_a^-1 b_a                                  # learned reward weights
    p_a = theta_a . x  +  alpha * sqrt( x . A_a^-1 . x )
          \___________/    \______________________________/
          reward estimate      uncertainty bonus (exploration)
Pick argmax_a p_a. After observing reward r for the chosen action a:
    A_a += x x^T ;  b_a += r x
The uncertainty bonus shrinks as an action accrues observations in the
direction of x (A_a grows, A_a^-1 shrinks) — exploration naturally decays.

Efficiency: we keep A_a^-1 cached and update it incrementally with the
Sherman-Morrison rank-1 formula (no d x d solve per step), and score all 128
actions in one vectorized pass. ``alpha`` (config.BANDIT_ALPHA) tunes
exploration: 0 = pure exploitation, higher = more exploration.
"""

from __future__ import annotations

import numpy as np

from src import config


class LinUCB:
    def __init__(self, n_actions, d, alpha=None, action_ids=None):
        self.n_actions = int(n_actions)
        self.d = int(d)
        self.alpha = float(config.BANDIT_ALPHA if alpha is None else alpha)
        self.action_ids = list(action_ids) if action_ids is not None else list(range(n_actions))
        self._index = {a: i for i, a in enumerate(self.action_ids)}

        eye = np.eye(self.d, dtype=np.float64)
        self.A = np.repeat(eye[None], self.n_actions, axis=0)       # (n, d, d)
        self.A_inv = np.repeat(eye[None], self.n_actions, axis=0)   # cached inverse
        self.b = np.zeros((self.n_actions, self.d), dtype=np.float64)

    # -- scoring -------------------------------------------------------------
    def scores(self, x: np.ndarray):
        """Return (ucb_scores, reward_estimates, uncertainty_bonuses) for all actions."""
        x = np.asarray(x, dtype=np.float64).ravel()
        theta = np.einsum("nij,nj->ni", self.A_inv, self.b)   # (n, d)
        reward_est = theta @ x                                 # (n,)
        Ainv_x = np.einsum("nij,j->ni", self.A_inv, x)         # (n, d)
        quad = np.einsum("ni,i->n", Ainv_x, x)                 # x^T A^-1 x  (n,)
        bonus = self.alpha * np.sqrt(np.maximum(quad, 0.0))
        return reward_est + bonus, reward_est, bonus

    def select_action(self, x):
        """Choose the highest-UCB action.

        Returns (chosen_action_id, reward_estimate, uncertainty_bonus, all_scores).
        Deterministic given the current state and x.
        """
        ucb, reward_est, bonus = self.scores(x)
        i = int(np.argmax(ucb))
        return self.action_ids[i], float(reward_est[i]), float(bonus[i]), ucb

    def top_k(self, x, k):
        """Top-k action_ids by UCB score (ties broken by action order)."""
        ucb, _, _ = self.scores(x)
        order = np.argsort(-ucb, kind="stable")[:k]
        return [self.action_ids[i] for i in order]

    def top_k_exploit(self, x, k):
        """Top-k action_ids by the learned reward estimate theta_a . x only.

        Excludes the exploration bonus — this is the *recommendation* ranking of
        the trained model (what it believes is best), used for hit_rate@k so the
        bandit is judged on what it learned, not on exploration noise.
        """
        _, reward_est, _ = self.scores(x)
        order = np.argsort(-reward_est, kind="stable")[:k]
        return [self.action_ids[i] for i in order]

    # -- learning ------------------------------------------------------------
    def update(self, action_id, x, reward):
        """Observe reward r for the chosen action; rank-1 update of A^-1 and b."""
        i = self._index[action_id]
        x = np.asarray(x, dtype=np.float64).ravel()
        self.A[i] += np.outer(x, x)
        # Sherman-Morrison: (A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1)/(1 + x^T A^-1 x)
        Ainv = self.A_inv[i]
        Ainv_x = Ainv @ x
        denom = 1.0 + float(x @ Ainv_x)
        self.A_inv[i] = Ainv - np.outer(Ainv_x, Ainv_x) / denom
        self.b[i] += reward * x

    def reward_estimate(self, action_id, x):
        """theta_a . x for one action (used in tests)."""
        i = self._index[action_id]
        x = np.asarray(x, dtype=np.float64).ravel()
        theta = self.A_inv[i] @ self.b[i]
        return float(theta @ x)

    def uncertainty(self, action_id, x):
        """alpha * sqrt(x^T A_a^-1 x) for one action (used in tests)."""
        i = self._index[action_id]
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(self.alpha * np.sqrt(max(x @ self.A_inv[i] @ x, 0.0)))
