"""Experiment 4 — content-based recommendation at article level.

Directly contrasts Experiment B, where article-level collaborative filtering
collapsed from sparsity: CF can only relate articles that co-occur, so ~82k
rarely-co-bought SKUs get almost no signal. Content-based recommendation sidesteps
that entirely — it scores articles by their **attributes** (product type, colour,
department, appearance, and a TF-IDF of the text description), so it can relate
articles that never appear in the same basket.

Why TF-IDF, not embeddings: the descriptions are short, factual, small-vocabulary
product copy; TF-IDF captures the salient terms cheaply, is interpretable, and
that interpretability feeds the later explanation layer. Embeddings would add cost
and opacity for little gain on this kind of text.

Pipeline (all sparse — never densify the article feature matrix):
  1. Item profiles: one-hot categorical attributes + TF-IDF(detail_desc),
     hstacked and L2-normalized (so cosine == dot product).
  2. Customer profile: recency-weighted average (reuse Exp 3 half-life) of the
     profiles of the articles the customer bought pre-cutoff.
  3. Score = cosine(customer profile, article profile); recommend top-k.
  Cold-start (no history) -> article popularity.

Article order is aligned to the Exp B article-level CF model so the two are
blendable in the content+CF hybrid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

from src import config
from src.models.popularity_article import ArticlePopularityModel

CATEGORICAL_ATTRS = [
    "product_type_name", "product_group_name", "colour_group_name",
    "department_name", "graphical_appearance_name",
]
TFIDF_MAX_FEATURES = 800


def _row_minmax(M: np.ndarray) -> np.ndarray:
    lo = M.min(axis=1, keepdims=True)
    hi = M.max(axis=1, keepdims=True)
    span = hi - lo
    safe = np.where(span > 0, span, 1.0)  # avoid 0/0 on flat rows
    out = (M - lo) / safe
    out[np.broadcast_to(span, M.shape) == 0] = 0.0
    return out.astype(np.float32)


def topk_dense(score_row, article_ids, bought_idx, k, include_repeats, pop_ranked):
    """Top-k article_ids from a dense score row: exclude bought + pad popularity."""
    row = score_row
    if not include_repeats and len(bought_idx):
        row = row.copy()
        row[bought_idx] = -np.inf
    # Take a generous top slice, then order exactly with an action_id tie-break.
    m = min(len(row), max(k * 4, k))
    cand = np.argpartition(-row, m - 1)[:m]
    cand = cand[np.isfinite(row[cand])]
    order = np.lexsort((article_ids[cand], -row[cand]))
    recs = [str(article_ids[cand[j]]) for j in order[:k]]
    if len(recs) < k:
        chosen = set(recs)
        bset = set() if include_repeats else set(article_ids[bought_idx].tolist())
        for a in pop_ranked:
            if a not in chosen and a not in bset:
                recs.append(a)
                chosen.add(a)
                if len(recs) == k:
                    break
    return recs[:k]


class ContentModel:
    """Content-based article recommender over article attribute profiles."""

    def __init__(self, popularity_model: ArticlePopularityModel | None = None):
        self.article_ids = None        # index -> article_id (string array), aligned to CF
        self.article_index = None
        self.item_matrix = None        # (n_articles x n_features) sparse, L2-normalized
        self.vectorizer = None
        self.encoder = None
        self.customer_articles = None  # cid -> (idx array, recency array)
        self.popularity_model = popularity_model
        self.n_features = None

    def fit(self, articles_df, feature_events, article_order=None,
            reference_date=None) -> "ContentModel":
        ref = pd.Timestamp(reference_date or config.CUTOFF_DATE)
        assert feature_events["t_dat"].max() < ref, "content fit saw post-reference events — leakage!"

        # Article set/order: align to the CF article space if given, else feature-side.
        if article_order is None:
            article_order = np.sort(feature_events["article_id"].astype("string").unique())
        self.article_ids = np.asarray(article_order)
        self.article_index = {a: i for i, a in enumerate(self.article_ids)}

        # Reindex the article metadata to the canonical order.
        adf = articles_df.copy()
        adf["article_id"] = adf["article_id"].astype("string")
        adf = adf.set_index("article_id").reindex(self.article_ids)
        for c in CATEGORICAL_ATTRS:
            adf[c] = adf[c].astype("string").fillna("unknown")
        adf["detail_desc"] = adf["detail_desc"].astype("string").fillna("unknown")

        # 1. Item profiles: one-hot categorical + TF-IDF(text), hstacked, L2-normalized.
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        cat = self.encoder.fit_transform(adf[CATEGORICAL_ATTRS].to_numpy())
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=TFIDF_MAX_FEATURES
        )
        txt = self.vectorizer.fit_transform(adf["detail_desc"].tolist())
        M = sp.hstack([cat, txt]).tocsr()
        self.item_matrix = normalize(M, norm="l2", axis=1)  # rows unit-norm
        self.n_features = self.item_matrix.shape[1]

        # 2. Per-customer purchased article indices + recency (half-life to ref).
        pairs = feature_events[["customer_id", "article_id", "t_dat"]].copy()
        pairs["article_id"] = pairs["article_id"].astype("string")
        last = pairs.groupby(["customer_id", "article_id"], sort=False)["t_dat"].max().reset_index()
        last = last[last["article_id"].isin(self.article_index)]
        last["idx"] = last["article_id"].map(self.article_index).astype("int64")
        delta = (ref - last["t_dat"]).dt.days.to_numpy()
        last["rec"] = np.power(0.5, delta / config.HALF_LIFE_DAYS).astype(np.float32)
        self.customer_articles = {
            cid: (grp["idx"].to_numpy(), grp["rec"].to_numpy(dtype=np.float32))
            for cid, grp in last.groupby("customer_id", sort=False)
        }

        if self.popularity_model is None:
            self.popularity_model = ArticlePopularityModel().fit(feature_events)
        return self

    # -- scoring -------------------------------------------------------------
    def _profiles(self, warm_ids):
        """Recency-weighted customer profiles (n x n_features) as a sparse matrix."""
        rows, cols, data = [], [], []
        n_art = len(self.article_ids)
        for i, c in enumerate(warm_ids):
            idx, rec = self.customer_articles[c]
            rows.append(np.full(len(idx), i, dtype=np.int64))
            cols.append(idx)
            data.append(rec)
        W = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(warm_ids), n_art),
        )
        return W @ self.item_matrix  # (n x n_features), weighted sum of item profiles

    def score_chunk(self, warm_ids) -> np.ndarray:
        """Dense content-similarity scores (len(warm_ids) x n_articles)."""
        P = self._profiles(warm_ids)              # n x n_features (sparse)
        return (P @ self.item_matrix.T).toarray()  # n x n_articles (cosine, rows unit-norm)

    def similar_articles(self, article_id, n=5):
        """Top-n content-similar articles to a given article as (article_id, cos)."""
        i = self.article_index[str(article_id)]
        sims = (self.item_matrix @ self.item_matrix[i].T).toarray().ravel()
        sims[i] = -np.inf
        order = np.argsort(-sims)[:n]
        return [(str(self.article_ids[j]), float(sims[j])) for j in order]

    def recommend(self, customer_id, k=12, include_repeats=False):
        if not self.customer_articles or customer_id not in self.customer_articles:
            return self.popularity_model.recommend(customer_id, k=k)  # cold-start
        S = self.score_chunk([customer_id])[0]
        bought = self.customer_articles[customer_id][0]
        return topk_dense(S, self.article_ids, bought, k, include_repeats,
                          self.popularity_model.ranked_articles)

    def recommend_all(self, customer_ids, k=24, include_repeats=False, batch_size=1000) -> dict:
        customer_ids = list(customer_ids)
        warm = [c for c in customer_ids if c in self.customer_articles]
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
                bought = self.customer_articles[c][0]
                recs[c] = topk_dense(row, self.article_ids, bought, k,
                                     include_repeats, pop_ranked)
        return recs


def load_articles():
    return pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")


def load_feature_events():
    return pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
