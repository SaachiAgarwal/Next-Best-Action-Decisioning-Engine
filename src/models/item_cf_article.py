"""Sparse article-level item-to-item collaborative filtering (Experiment B).

At article granularity there are ~79k feature-side articles. A dense 79k x 79k
similarity matrix would be ~6.3B cells (~25 GB float32) — impossible. The core
engineering move is to stay **sparse** end to end:

  1. Binary customer x article interaction as a scipy.sparse CSR matrix A
     (1 = customer bought the article pre-cutoff, distinct).
  2. Article-article co-occurrence C = A^T A (sparse). C has nonzeros only for
     article pairs that were actually co-bought — the full dense matrix is never
     materialized.
  3. Cosine normalization: sim = D^-1/2 C D^-1/2 with D = diag(C) (per-article
     customer counts). Self-similarity is zeroed. sim stays sparse.
  4. Scoring: a customer's score vector is (indicator @ sim), a sparse
     vector-matrix product that touches only articles reachable via
     co-occurrence (neighbors), never all 79k.

LEAKAGE GUARANTEE: fit on feature-side events only; ``fit`` asserts it.
Cold-start customers (no history / no reachable neighbors) fall back to article
popularity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config
from src.models.popularity_article import ArticlePopularityModel


def _mb(*arrays) -> float:
    return sum(a.nbytes for a in arrays) / 1e6


class ArticleItemCF:
    """Sparse item-to-item CF over the article action space."""

    def __init__(self, popularity_model: ArticlePopularityModel | None = None):
        self.similarity = None        # scipy.sparse CSR (n_articles x n_articles)
        self.article_ids = None       # index -> article_id (np array)
        self.article_index = None     # article_id -> index
        self.customer_rows = None     # customer_id -> CSR row (1 x n_articles)
        self.customer_articles = None # customer_id -> np.array of article indices
        self.popularity_model = popularity_model
        self.memory = {}

    def fit(self, feature_events: pd.DataFrame, verbose: bool = True) -> "ArticleItemCF":
        if "t_dat" in feature_events.columns:
            cutoff = pd.Timestamp(config.CUTOFF_DATE)
            assert feature_events["t_dat"].max() < cutoff, (
                "article item-CF received events on/after the cutoff — leakage!"
            )

        # 1. Sparse binary customer x article matrix.
        # article_id is a STRING (leading zeros matter) — keep it as a string
        # array throughout; never coerce to int or joins/labels silently break.
        pairs = feature_events[["customer_id", "article_id"]].drop_duplicates()
        self.article_ids = np.array(sorted(pairs["article_id"].astype("string").unique().tolist()))
        self.article_index = {a: i for i, a in enumerate(self.article_ids)}
        cust_ids = pairs["customer_id"].unique()
        cust_index = {c: i for i, c in enumerate(cust_ids)}

        rows = pairs["customer_id"].map(cust_index).to_numpy()
        cols = pairs["article_id"].map(self.article_index).to_numpy()
        data = np.ones(len(pairs), dtype=np.float32)
        n_cust, n_art = len(cust_ids), len(self.article_ids)
        A = sp.csr_matrix((data, (rows, cols)), shape=(n_cust, n_art), dtype=np.float32)

        # 2. Sparse co-occurrence C = A^T A (only co-bought pairs are nonzero).
        C = (A.T @ A).tocsr()

        # 3. Cosine normalization; zero the diagonal.
        diag = C.diagonal().copy()          # per-article customer counts
        C.setdiag(0)
        C.eliminate_zeros()
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(diag > 0, 1.0 / np.sqrt(diag), 0.0).astype(np.float32)
        Dinv = sp.diags(inv_sqrt)
        self.similarity = (Dinv @ C @ Dinv).tocsr()

        # Per-customer purchased article indices + indicator rows (for scoring).
        grouped = pairs.groupby("customer_id", sort=False)["article_id"]
        self.customer_articles = {
            cid: np.array([self.article_index[a] for a in grp], dtype=np.int64)
            for cid, grp in grouped
        }

        if self.popularity_model is None:
            self.popularity_model = ArticlePopularityModel().fit(feature_events)

        # Memory footprint of the sparse structures.
        self.memory = {
            "n_customers": n_cust,
            "n_articles": n_art,
            "A_nnz": A.nnz,
            "A_mb": _mb(A.data, A.indices, A.indptr),
            "C_nnz": int(self.similarity.nnz),
            "sim_mb": _mb(self.similarity.data, self.similarity.indices, self.similarity.indptr),
            "dense_would_be_gb": (n_art * n_art * 4) / 1e9,
        }
        if verbose:
            self.print_memory()
        return self

    def print_memory(self) -> None:
        m = self.memory
        print("=" * 60)
        print("SPARSE MEMORY FOOTPRINT (article item-CF)")
        print("=" * 60)
        print(f"  customers x articles      : {m['n_customers']:,} x {m['n_articles']:,}")
        print(f"  interaction A: nnz={m['A_nnz']:,}  ({m['A_mb']:.1f} MB)")
        print(f"  similarity   : nnz={m['C_nnz']:,}  ({m['sim_mb']:.1f} MB)")
        print(f"  a DENSE {m['n_articles']:,}^2 float32 matrix would be "
              f"{m['dense_would_be_gb']:.1f} GB (never materialized)")

    def similar_articles(self, article_id, n: int = 5):
        """Top-n most similar articles as (article_id, score)."""
        i = self.article_index[article_id]
        row = self.similarity.getrow(i).tocoo()
        order = np.argsort(-row.data)[:n]
        return [(str(self.article_ids[row.col[j]]), float(row.data[j])) for j in order]

    def recommend(self, customer_id, k: int = 12, include_repeats: bool = False):
        """Top-k personalized articles, scored sparsely over reachable neighbors."""
        idxs = self.customer_articles.get(customer_id) if self.customer_articles else None
        if idxs is None or len(idxs) == 0:
            return self.popularity_model.recommend(customer_id, k=k)  # cold-start

        # Sparse indicator @ similarity -> scores only over co-occurring neighbors.
        n = len(self.article_ids)
        u = sp.csr_matrix(
            (np.ones(len(idxs), dtype=np.float32), (np.zeros(len(idxs)), idxs)),
            shape=(1, n),
        )
        scores = (u @ self.similarity).tocoo()
        cand_idx, cand_score = scores.col, scores.data

        if not include_repeats and len(cand_idx):
            bought = set(idxs.tolist())
            keep = np.array([c not in bought for c in cand_idx], dtype=bool)
            cand_idx, cand_score = cand_idx[keep], cand_score[keep]

        # Top-k neighbors by score (tie-break by article_id for determinism).
        recs = []
        if len(cand_idx):
            order = np.lexsort((self.article_ids[cand_idx], -cand_score))
            recs = [str(self.article_ids[cand_idx[j]]) for j in order[:k]]

        # Pad to exactly k from popularity if too few reachable neighbors.
        if len(recs) < k:
            chosen = set(recs)
            bought = set() if include_repeats else set(self.article_ids[idxs].tolist())
            for a in self.popularity_model.ranked_articles:
                if a not in chosen and a not in bought:
                    recs.append(a)
                    chosen.add(a)
                    if len(recs) == k:
                        break
        return recs[:k]

    def _topk_from_row(self, cand_idx, cand_score, bought_idx, k, include_repeats):
        """Top-k article_ids for one scored row, excluding bought + padding."""
        if not include_repeats and len(cand_idx):
            mask = ~np.isin(cand_idx, bought_idx)
            cand_idx, cand_score = cand_idx[mask], cand_score[mask]
        recs = []
        if len(cand_idx):
            order = np.lexsort((self.article_ids[cand_idx], -cand_score))
            recs = [str(self.article_ids[cand_idx[j]]) for j in order[:k]]
        if len(recs) < k:  # pad from popularity if too few reachable neighbors
            chosen = set(recs)
            bset = set() if include_repeats else set(self.article_ids[bought_idx].tolist())
            for a in self.popularity_model.ranked_articles:
                if a not in chosen and a not in bset:
                    recs.append(a)
                    chosen.add(a)
                    if len(recs) == k:
                        break
        return recs[:k]

    def recommend_all(self, customer_ids, k: int = 24, include_repeats: bool = False,
                      batch_size: int = 2000) -> dict:
        """Batched scoring: one sparse (indicator @ similarity) matmul per chunk.

        Far faster than per-customer scoring for large evaluable sets; results are
        identical to calling ``recommend`` per customer. Cold-start customers
        (no history) get the popularity top-k.
        """
        customer_ids = list(customer_ids)
        warm = [c for c in customer_ids
                if self.customer_articles.get(c) is not None and len(self.customer_articles[c])]
        warm_set = set(warm)
        pop_topk = self.popularity_model.recommend(k=k)

        recs = {c: list(pop_topk) for c in customer_ids if c not in warm_set}  # cold-start

        n = len(self.article_ids)
        for start in range(0, len(warm), batch_size):
            chunk = warm[start:start + batch_size]
            rows, cols = [], []
            for i, c in enumerate(chunk):
                a = self.customer_articles[c]
                rows.append(np.full(len(a), i, dtype=np.int64))
                cols.append(a)
            r = np.concatenate(rows); cl = np.concatenate(cols)
            U = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, cl)),
                              shape=(len(chunk), n))
            S = (U @ self.similarity).tocsr()
            for i, c in enumerate(chunk):
                lo, hi = S.indptr[i], S.indptr[i + 1]
                recs[c] = self._topk_from_row(
                    S.indices[lo:hi], S.data[lo:hi], self.customer_articles[c], k, include_repeats
                )
        return recs


def load_feature_events() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
