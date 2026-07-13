r"""Shared-model LinUCB at article level (Phase 2c).

At ~79k articles a disjoint (per-arm) bandit is impossible: each arm would see far
less than one observation, so its A_a/b_a never leave the prior. A **shared model**
maintains a SINGLE (A, b) over the joint (customer, article) **feature space**, so
**every** observation updates the same θ and learning **transfers across articles**
— including articles never seen in training, which still get a score because θ acts
on their features.

    θ = A^-1 b
    p_ca = θ · x_ca  +  alpha * sqrt( x_ca · A^-1 · x_ca )     (over candidate articles a)
    choose argmax_a p_ca ; on reward r:  A += x_ca x_ca^T ,  b += r x_ca   (Sherman-Morrison)

Contrast with the disjoint formulation (v3): there, an unseen action's model is the
untrained prior and cannot benefit from any other action's data. Here a single θ
generalizes over features, so the sparse ~79k-article space is dissolved into a
fixed-dimensional feature problem. ``score_one`` scores an arbitrary feature vector
(the key unseen-article property).
"""

from __future__ import annotations

import numpy as np

from src import config


class SharedLinUCB:
    def __init__(self, d, alpha=None):
        self.d = int(d)
        self.alpha = float(config.SHARED_ALPHA if alpha is None else alpha)
        self.A = np.eye(self.d, dtype=np.float64)
        self.A_inv = np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)

    # -- scoring (candidate feature matrix X: (n_candidates, d)) --------------
    def scores(self, X):
        X = np.asarray(X, dtype=np.float64)
        theta = self.A_inv @ self.b                     # (d,)
        reward_est = X @ theta                          # (n,)
        Ainv_x = X @ self.A_inv                         # (n, d)  (A_inv symmetric)
        quad = np.einsum("ni,ni->n", Ainv_x, X)         # x A^-1 x per candidate
        bonus = self.alpha * np.sqrt(np.maximum(quad, 0.0))
        return reward_est + bonus, reward_est, bonus

    def select_index(self, X):
        """Index of the argmax-UCB candidate; also returns (est, bonus, ucb)."""
        ucb, est, bonus = self.scores(X)
        i = int(np.argmax(ucb))
        return i, float(est[i]), float(bonus[i]), ucb

    def top_k_exploit_index(self, X, k):
        """Indices of the top-k candidates by exploitation estimate θ·x (no bonus)."""
        _, est, _ = self.scores(X)
        return np.argsort(-est, kind="stable")[:k]

    def score_one(self, x):
        """θ·x for a single feature vector — works for ANY article, seen or not."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return float((self.A_inv @ self.b) @ x)

    def uncertainty_one(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(self.alpha * np.sqrt(max(x @ self.A_inv @ x, 0.0)))

    # -- learning ------------------------------------------------------------
    def update(self, x, reward):
        """Rank-1 update of the single shared model (Sherman-Morrison)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self.A += np.outer(x, x)
        Ainv_x = self.A_inv @ x
        denom = 1.0 + float(x @ Ainv_x)
        self.A_inv = self.A_inv - np.outer(Ainv_x, Ainv_x) / denom
        self.b += reward * x
