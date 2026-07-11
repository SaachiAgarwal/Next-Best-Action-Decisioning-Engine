"""Experiment 3 — recency + frequency weighted hybrid (product-type level).

One model designed to beat the popularity baseline by combining four ingredients,
all computed from feature-side events only:

  1. FREQUENCY (how habitual) — log(1 + count) of the customer's purchases of an
     action. Log-dampening is deliberate: a 10x buyer likes an action more than a
     1x buyer, but not 10x more, and it stops volume outliers (e.g. the
     1,237-purchase customer) from dominating.
  2. RECENCY (how current) — w = 0.5 ** (Δ / HALF_LIFE_DAYS) where Δ is days from
     the customer's MOST RECENT purchase of that action to the reference date.
     HALF_LIFE_DAYS is a true half-life: w = 1 at the reference and exactly 0.5
     one half-life before it (equivalently exp(-ln2·Δ/HALF_LIFE_DAYS)).
  3. CF (what similar customers co-buy) — reuse product-type item-CF cosine
     similarity, but weight each of the customer's purchases by its recency, so
     recent co-purchases count more:
         cf_score(c, a) = Σ_j recency_w(c, j) * sim(j, a)   over purchased j.
  4. POPULARITY prior — global purchase counts, normalized. This is the floor
     that guarantees the blend can fall back to (and never underperform) the
     popularity baseline; it also handles cold-start automatically (with no
     history, personal/cf are zero so gamma dominates).

personal_score(c, a) = log(1 + count) * recency_w   (ingredients 1 x 2)

final_score = ALPHA*personal + BETA*cf + GAMMA*popularity, with each component
min-max normalized per customer to a comparable [0, 1] scale before weighting.
Repeats are INCLUDED by default (fashion rebuys; product-type repeat lift was
+0.319), so already-bought actions are never excluded.

LEAKAGE GUARANTEE: fit takes an explicit ``reference_date`` and asserts every
event predates it. Tuning fits on a train sub-window (reference = validation
cutoff); final fit uses the real CUTOFF_DATE. The post-cutoff label window is
never seen during fit/tune.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.models.item_cf import ItemCF
from src.models.popularity import PopularityModel


def _row_minmax(M: np.ndarray) -> np.ndarray:
    """Per-row min-max to [0, 1]; rows with no spread (e.g. all-zero) -> zeros."""
    lo = M.min(axis=1, keepdims=True)
    hi = M.max(axis=1, keepdims=True)
    span = hi - lo
    out = np.where(span > 0, (M - lo) / span, 0.0)
    return out.astype(np.float32)


class HybridModel:
    """Recency + frequency weighted hybrid over the 128 product-type actions."""

    def __init__(self):
        self.action_ids = None       # canonical action_id order (from actions.parquet)
        self.action_index = None
        self.customer_index = None
        self.personal = None         # (n_cust x n_actions) log-freq x recency
        self.cf = None               # (n_cust x n_actions) recency-weighted CF
        self.pop = None              # (n_actions,) global purchase counts
        self.pop_norm = None         # (n_actions,) min-max normalized popularity
        self.popularity_model = None # cold-start fallback + pop ranking
        self.reference_date = None

    # -- fit -----------------------------------------------------------------
    def _canonical_actions(self):
        actions = pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
        return np.sort(actions["action_id"].to_numpy())

    def fit(self, events: pd.DataFrame, reference_date, action_ids=None) -> "HybridModel":
        ref = pd.Timestamp(reference_date)
        assert events["t_dat"].max() < ref, (
            f"hybrid.fit received events on/after reference {ref.date()} — leakage!"
        )
        self.reference_date = ref

        self.action_ids = np.sort(action_ids) if action_ids is not None else self._canonical_actions()
        self.action_index = {int(a): i for i, a in enumerate(self.action_ids)}
        n_act = len(self.action_ids)

        cust_ids = events["customer_id"].unique()
        self.customer_index = {c: i for i, c in enumerate(cust_ids)}
        n_cust = len(cust_ids)

        # Per (customer, action): purchase count and most-recent purchase date.
        agg = (
            events.groupby(["customer_id", "action_id"], sort=False)["t_dat"]
            .agg(count="size", last="max").reset_index()
        )
        rows = agg["customer_id"].map(self.customer_index).to_numpy()
        cols = agg["action_id"].map(self.action_index).to_numpy()
        counts = agg["count"].to_numpy(dtype=np.float32)
        delta_days = (ref - agg["last"]).dt.days.to_numpy()
        # True half-life decay: weight is exactly 0.5 one HALF_LIFE_DAYS before ref.
        recency = np.power(0.5, delta_days / config.HALF_LIFE_DAYS).astype(np.float32)

        F = np.zeros((n_cust, n_act), dtype=np.float32)
        W = np.zeros((n_cust, n_act), dtype=np.float32)
        F[rows, cols] = counts
        W[rows, cols] = recency

        # Ingredient 1x2: personal repurchase propensity.
        self.personal = (np.log1p(F) * W).astype(np.float32)

        # Ingredient 3: recency-weighted CF, embedding item-CF sim into canonical order.
        itemcf = ItemCF().fit(events)
        sim = np.zeros((n_act, n_act), dtype=np.float32)
        embed = [self.action_index[int(a)] for a in itemcf.action_ids]
        sim[np.ix_(embed, embed)] = itemcf.similarity
        self.cf = (W @ sim).astype(np.float32)

        # Ingredient 4: global popularity prior.
        self.popularity_model = PopularityModel().fit(events)
        pop = np.zeros(n_act, dtype=np.float32)
        pc = events.groupby("action_id").size()
        for a, c in pc.items():
            pop[self.action_index[int(a)]] = c
        self.pop = pop
        self.pop_norm = _row_minmax(pop.reshape(1, -1)).ravel()
        return self

    # -- scoring -------------------------------------------------------------
    def score_rows(self, row_indices, alpha, beta, gamma) -> np.ndarray:
        """Blended score matrix for the given customer row indices."""
        Pn = _row_minmax(self.personal[row_indices])
        Cn = _row_minmax(self.cf[row_indices])
        return alpha * Pn + beta * Cn + gamma * self.pop_norm

    def _weights(self, alpha, beta, gamma):
        a = config.ALPHA_EXP3 if alpha is None else alpha
        b = config.BETA_EXP3 if beta is None else beta
        g = config.GAMMA_EXP3 if gamma is None else gamma
        return a, b, g

    def _topk_row(self, score_row, k):
        order = np.lexsort((self.action_ids, -score_row))  # -score primary, action_id tie
        return [int(self.action_ids[j]) for j in order[:k]]

    def recommend(self, customer_id, k=12, alpha=None, beta=None, gamma=None):
        """Top-k by final score (repeats included). Cold-start -> popularity top-k."""
        a, b, g = self._weights(alpha, beta, gamma)
        idx = self.customer_index.get(customer_id) if self.customer_index else None
        if idx is None:
            return self.popularity_model.recommend(customer_id, k=k)  # cold-start
        score = self.score_rows([idx], a, b, g)[0]
        return self._topk_row(score, k)

    def recommend_all(self, customer_ids, k=24, alpha=None, beta=None, gamma=None) -> dict:
        a, b, g = self._weights(alpha, beta, gamma)
        customer_ids = list(customer_ids)
        warm_idx, warm_ids, cold_ids = [], [], []
        for c in customer_ids:
            i = self.customer_index.get(c)
            if i is None:
                cold_ids.append(c)
            else:
                warm_idx.append(i)
                warm_ids.append(c)

        recs = {}
        if cold_ids:
            pop_topk = self.popularity_model.recommend(k=k)
            for c in cold_ids:
                recs[c] = list(pop_topk)
        if warm_idx:
            S = self.score_rows(warm_idx, a, b, g)
            for c, row in zip(warm_ids, S):
                recs[c] = self._topk_row(row, k)
        return recs
