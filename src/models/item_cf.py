"""Item-to-item collaborative filtering for the NBA Decisioning Engine.

The first *personalized* model. Intuition: actions that are frequently bought by
the same customers are related, so we recommend actions similar to what a
customer has already bought.

Pipeline:
  1. Binary customer-action interaction — for each customer, the SET of distinct
     action_ids bought pre-cutoff. We use **binary** "bought / didn't", not
     purchase counts: at product-type granularity, buying "trousers" ten times
     mostly reflects category volume, not ten times the affinity; binary co-buy
     is the cleaner co-occurrence signal.
  2. Action-action co-occurrence C[i,j] = # distinct customers who bought both.
  3. **Cosine-normalized similarity** sim[i,j] = C[i,j]/(sqrt(C[i,i])*sqrt(C[j,j])).
     Normalization is the critical step: raw co-occurrence is dominated by
     globally-popular actions (everyone buys trousers, so trousers co-occurs
     with everything). Dividing by each action's own popularity corrects for
     that, so item-CF learns real structure instead of collapsing back into the
     popularity baseline. Self-similarity is zeroed (an action is not its own
     recommendation).

LEAKAGE GUARANTEE: fit on feature-side events only (t_dat < CUTOFF_DATE);
``fit`` asserts this, same guard as the popularity baseline.

This is a 128 x 128 action space, so a dense similarity matrix is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.models.popularity import PopularityModel


class ItemCF:
    """Item-to-item CF over the (small, dense) action space."""

    def __init__(self, popularity_model: PopularityModel | None = None):
        self.similarity = None          # (n_actions, n_actions) cosine similarity
        self.cooccurrence = None        # (n_actions, n_actions) raw co-counts
        self.action_ids = None          # matrix index -> action_id
        self.action_index = None        # action_id -> matrix index
        self.customer_actions = None    # customer_id -> np.array of matrix indices
        self.popularity_model = popularity_model

    def fit(self, feature_events: pd.DataFrame) -> "ItemCF":
        # Leakage guard: only pre-cutoff events may inform the model.
        if "t_dat" in feature_events.columns:
            cutoff = pd.Timestamp(config.CUTOFF_DATE)
            assert feature_events["t_dat"].max() < cutoff, (
                "item-CF received events on/after the cutoff — label leakage! "
                "Fit on features_events.parquet only."
            )

        # 1. Binary customer-action pairs (distinct).
        pairs = feature_events[["customer_id", "action_id"]].drop_duplicates()

        self.action_ids = np.sort(pairs["action_id"].unique())
        self.action_index = {a: i for i, a in enumerate(self.action_ids)}
        n_actions = len(self.action_ids)

        cust_ids = pairs["customer_id"].unique()
        cust_index = {c: i for i, c in enumerate(cust_ids)}
        n_customers = len(cust_ids)

        row = pairs["customer_id"].map(cust_index).to_numpy()
        col = pairs["action_id"].map(self.action_index).to_numpy()

        # 2. Binary interaction matrix B (customers x actions) and co-occurrence.
        B = np.zeros((n_customers, n_actions), dtype=np.float32)
        B[row, col] = 1.0
        C = B.T @ B  # C[i,j] = # customers who bought both i and j
        self.cooccurrence = C

        # 3. Cosine normalization (correct for popularity), zero the diagonal.
        diag = np.diagonal(C).copy()
        norm = np.sqrt(diag)
        denom = np.outer(norm, norm)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, C / denom, 0.0)
        np.fill_diagonal(sim, 0.0)
        self.similarity = sim.astype(np.float32)

        # Per-customer purchased action indices (for scoring at recommend time).
        grouped = pairs.groupby("customer_id", sort=False)["action_id"]
        self.customer_actions = {
            cid: np.array([self.action_index[a] for a in grp], dtype=np.int64)
            for cid, grp in grouped
        }

        # Cold-start fallback model.
        if self.popularity_model is None:
            self.popularity_model = PopularityModel().fit(feature_events)
        return self

    def similar_actions(self, action_id, n: int = 5):
        """Top-n most similar actions to ``action_id`` as (action_id, score)."""
        i = self.action_index[action_id]
        sims = self.similarity[i]
        order = np.argsort(-sims, kind="stable")[:n]
        return [(int(self.action_ids[j]), float(sims[j])) for j in order]

    def _rank(self, score: np.ndarray, k: int):
        """Top-k action_ids by score, ties broken by ascending action_id."""
        order = np.lexsort((self.action_ids, -score))  # primary: -score, tie: action_id
        return [int(self.action_ids[j]) for j in order[:k]]

    def recommend(self, customer_id, k: int = 12, include_repeats: bool = False):
        """Top-k personalized actions for a customer.

        Scores each candidate action by the sum of its similarity to the
        customer's pre-cutoff purchased actions. Cold-start customers (no
        history) fall back to the popularity top-k.
        """
        idxs = self.customer_actions.get(customer_id) if self.customer_actions else None
        if idxs is None or len(idxs) == 0:
            return self.popularity_model.recommend(customer_id, k=k)  # cold-start

        score = self.similarity[idxs].sum(axis=0)
        if not include_repeats:
            score = score.copy()
            score[idxs] = -np.inf  # exclude already-bought actions
        return self._rank(score, k)

    def recommend_all(self, customer_ids, k: int = 24, include_repeats: bool = False) -> dict:
        return {
            cid: self.recommend(cid, k=k, include_repeats=include_repeats)
            for cid in customer_ids
        }
