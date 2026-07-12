"""Experiment 5 — article-level matrix factorization (implicit-feedback ALS).

A second escape from the article-level sparsity that sank neighborhood CF in
Experiment B — benchmarked head-to-head against the content-based escape (Exp 4).

Neighborhood CF (Exp B) relates two articles only if customers *co-bought* them;
with ~82k SKUs, most pairs never co-occur, so the signal collapses. Matrix
factorization takes a different route: it learns a low-dimensional **latent
factor** vector for every customer and every article such that
``dot(U_customer, V_article)`` reconstructs the interactions. Because articles
that appear in similar contexts end up near each other in factor space, MF can
*generalize* to article pairs that never literally co-occurred — the theoretical
reason it should handle sparsity better than co-occurrence counting.

We use **implicit-feedback ALS** (there are no ratings, only binary purchase
signals; explicit-rating SVD would be inappropriate). Interactions are binary
(bought / not), matching Exp B. Article order is aligned to the Exp B / Exp 4
article space so all article-level models are directly blendable.

LEAKAGE GUARANTEE: fit on feature-side events only (t_dat < CUTOFF_DATE); asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src import config
from src.models.content_based_exp4 import topk_dense
from src.models.popularity_article import ArticlePopularityModel


class MFModel:
    """Implicit-feedback matrix factorization over the article space."""

    def __init__(self, factors=None, epochs=None, reg=None, popularity_model=None):
        self.factors = int(config.MF_FACTORS if factors is None else factors)
        self.epochs = int(config.MF_EPOCHS if epochs is None else epochs)
        self.reg = float(config.MF_REG if reg is None else reg)
        self.popularity_model = popularity_model
        self.U = None                 # customer factors (n_customers x factors)
        self.V = None                 # article factors (n_articles x factors)
        self.article_ids = None
        self.article_index = None
        self.customer_index = None
        self.customer_articles = None  # cid -> np.array of article indices (for repeats)
        self.interaction_counts = None  # per-article interaction count
        self.sparsity = None

    def fit(self, feature_events, article_order=None) -> "MFModel":
        if "t_dat" in feature_events.columns:
            assert feature_events["t_dat"].max() < pd.Timestamp(config.CUTOFF_DATE), \
                "MF received events on/after the cutoff — leakage!"

        pairs = feature_events[["customer_id", "article_id"]].drop_duplicates()
        pairs["article_id"] = pairs["article_id"].astype("string")
        if article_order is None:
            self.article_ids = np.array(sorted(pairs["article_id"].unique().tolist()))
        else:
            self.article_ids = np.asarray(article_order)
        self.article_index = {a: i for i, a in enumerate(self.article_ids)}

        cust_ids = pairs["customer_id"].unique()
        self.customer_index = {c: i for i, c in enumerate(cust_ids)}
        n_cust, n_art = len(cust_ids), len(self.article_ids)

        keep = pairs["article_id"].isin(self.article_index)
        pairs = pairs[keep]
        rows = pairs["customer_id"].map(self.customer_index).to_numpy()
        cols = pairs["article_id"].map(self.article_index).to_numpy()
        B = sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rows, cols)),
                          shape=(n_cust, n_art))
        self.sparsity = 100.0 * B.nnz / (n_cust * n_art)
        self.interaction_counts = np.asarray(B.sum(axis=0)).ravel()

        model = AlternatingLeastSquares(
            factors=self.factors, regularization=self.reg,
            iterations=self.epochs, random_state=config.SEED,
        )
        model.fit(B, show_progress=False)
        self.U = np.asarray(model.user_factors)
        self.V = np.asarray(model.item_factors)

        grouped = pairs.groupby("customer_id", sort=False)["article_id"]
        self.customer_articles = {
            cid: np.array([self.article_index[a] for a in grp], dtype=np.int64)
            for cid, grp in grouped
        }
        if self.popularity_model is None:
            self.popularity_model = ArticlePopularityModel().fit(feature_events)
        return self

    # -- scoring -------------------------------------------------------------
    def score_chunk(self, warm_ids) -> np.ndarray:
        """Dense MF affinity scores (len(warm_ids) x n_articles) = U . V^T."""
        rows = [self.customer_index[c] for c in warm_ids]
        return self.U[rows] @ self.V.T

    def similar_articles(self, article_id, n=5):
        """Top-n nearest neighbors in article factor space (cosine)."""
        i = self.article_index[str(article_id)]
        v = self.V[i]
        norms = np.linalg.norm(self.V, axis=1) * (np.linalg.norm(v) + 1e-12)
        cos = (self.V @ v) / (norms + 1e-12)
        cos[i] = -np.inf
        order = np.argsort(-cos)[:n]
        return [(str(self.article_ids[j]), float(cos[j])) for j in order]

    def recommend(self, customer_id, k=12, include_repeats=False):
        if not self.customer_articles or customer_id not in self.customer_index:
            return self.popularity_model.recommend(customer_id, k=k)  # cold-start
        S = self.score_chunk([customer_id])[0]
        bought = self.customer_articles.get(customer_id, np.array([], dtype=np.int64))
        return topk_dense(S, self.article_ids, bought, k, include_repeats,
                          self.popularity_model.ranked_articles)

    def recommend_all(self, customer_ids, k=24, include_repeats=False, batch_size=1000) -> dict:
        customer_ids = list(customer_ids)
        warm = [c for c in customer_ids if c in self.customer_index]
        warm_set = set(warm)
        recs = {}
        pop_topk = self.popularity_model.recommend(k=k)
        for c in customer_ids:
            if c not in warm_set:
                recs[c] = list(pop_topk)
        pop_ranked = self.popularity_model.ranked_articles
        for start in range(0, len(warm), batch_size):
            chunk = warm[start:start + batch_size]
            S = self.score_chunk(chunk)
            for c, row in zip(chunk, S):
                bought = self.customer_articles.get(c, np.array([], dtype=np.int64))
                recs[c] = topk_dense(row, self.article_ids, bought, k, include_repeats, pop_ranked)
        return recs


class TripleHybrid:
    """Blend of content (Exp 4), neighborhood CF (Exp B), and MF (Exp 5) scores.

    final = w1*content + w2*cf + w3*mf, each component min-max normalized per
    customer. All three score the same (aligned) article space. With w3=0 this
    reproduces the Exp 4 two-signal content+CF hybrid exactly.
    """

    def __init__(self, content_model, cf_model, mf_model):
        from src.models.content_based_exp4 import _row_minmax
        from src.models.hybrid_content_cf_exp4 import cf_score_chunk
        self.content = content_model
        self.cf = cf_model
        self.mf = mf_model
        self.article_ids = content_model.article_ids
        self.popularity_model = content_model.popularity_model
        self._mm = _row_minmax
        self._cf_score = cf_score_chunk

    def _warm(self, customer_ids):
        return [c for c in customer_ids
                if c in self.content.customer_articles
                and c in self.cf.customer_articles
                and c in self.mf.customer_index]

    def blended_chunk(self, warm_ids, w1, w2, w3):
        Cn = self._mm(self.content.score_chunk(warm_ids))
        Fn = self._mm(self._cf_score(self.cf, warm_ids))
        Mn = self._mm(self.mf.score_chunk(warm_ids))
        return w1 * Cn + w2 * Fn + w3 * Mn

    def recommend(self, customer_id, k=12, w1=1.0, w2=1.0, w3=1.0, include_repeats=False):
        if customer_id not in self._warm([customer_id]):
            return self.popularity_model.recommend(customer_id, k=k)
        S = self.blended_chunk([customer_id], w1, w2, w3)[0]
        bought = self.content.customer_articles[customer_id][0]
        return topk_dense(S, self.article_ids, bought, k, include_repeats,
                          self.popularity_model.ranked_articles)

    def recommend_all(self, customer_ids, k=24, w1=1.0, w2=1.0, w3=1.0,
                      include_repeats=False, batch_size=1000):
        customer_ids = list(customer_ids)
        warm = self._warm(customer_ids)
        warm_set = set(warm)
        recs = {}
        pop_topk = self.popularity_model.recommend(k=k)
        for c in customer_ids:
            if c not in warm_set:
                recs[c] = list(pop_topk)
        pop_ranked = self.popularity_model.ranked_articles
        for start in range(0, len(warm), batch_size):
            chunk = warm[start:start + batch_size]
            S = self.blended_chunk(chunk, w1, w2, w3)
            for c, row in zip(chunk, S):
                bought = self.content.customer_articles[c][0]
                recs[c] = topk_dense(row, self.article_ids, bought, k, include_repeats, pop_ranked)
        return recs


def load_articles():
    return pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")


def load_feature_events():
    return pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
