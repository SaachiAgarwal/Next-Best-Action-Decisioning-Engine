"""Experiment 4 — content + collaborative-filtering hybrid (article level).

Content-based recommendation (Exp 4 Task 1-2) and article-level CF (Exp B) have
complementary blind spots:

  - **Content** relates articles by shared attributes, so it works on sparse SKUs
    that never co-occur — but it lives in a filter bubble: it only ever suggests
    more of the same type/colour/department the customer already bought.
  - **CF** captures cross-attribute serendipity (customers who bought X also
    bought the unrelated Y) — but it collapses on sparse articles that lack
    co-occurrence signal (Experiment B).

Blending them lets each cover the other's gap:

    final = CONTENT_ALPHA * content_score + CONTENT_BETA * cf_score

with both components min-max normalized per customer to a comparable scale before
weighting. Both score the same ~79k article space (content is built aligned to the
Exp B CF article order), so the blend is row-comparable. Repeats flag and
cold-start->popularity behave as in the components.

This module REUSES a fitted Exp B ``ArticleItemCF`` via its public attributes
(``similarity``, ``customer_articles``, ``article_ids``); it does not modify it.

Documented extension (NOT built): a *hierarchical* variant could broadcast the
product-type-level CF similarity (Exp A, dense 128x128) down to the articles that
map to each product type, giving sparse SKUs a stable third signal borrowed from
their category. Noted as future work in the report.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src import config
from src.models.content_based_exp4 import ContentModel, _row_minmax, topk_dense
from src.models.item_cf_article import ArticleItemCF


def cf_score_chunk(cf: ArticleItemCF, warm_ids) -> np.ndarray:
    """Dense article-level CF scores (len(warm_ids) x n_articles) from a fitted
    Exp B ArticleItemCF, using its public attributes only."""
    n_art = len(cf.article_ids)
    rows, cols = [], []
    for i, c in enumerate(warm_ids):
        a = cf.customer_articles[c]
        rows.append(np.full(len(a), i, dtype=np.int64))
        cols.append(a)
    U = sp.csr_matrix(
        (np.ones(sum(len(r) for r in rows), dtype=np.float32),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(warm_ids), n_art),
    )
    return (U @ cf.similarity).toarray()


class ContentCFHybrid:
    """Blend of content-based and article-level CF scores."""

    def __init__(self, content_model: ContentModel, cf_model: ArticleItemCF):
        self.content = content_model
        self.cf = cf_model
        # Both models share the same article order (content built on cf.article_ids).
        assert np.array_equal(self.content.article_ids, self.cf.article_ids), \
            "content and CF article orders differ — cannot blend"
        self.article_ids = self.content.article_ids
        self.popularity_model = content_model.popularity_model

    def _weights(self, alpha, beta):
        a = config.CONTENT_ALPHA_EXP4 if alpha is None else alpha
        b = config.CONTENT_BETA_EXP4 if beta is None else beta
        return a, b

    def _warm(self, customer_ids):
        return [c for c in customer_ids
                if c in self.content.customer_articles and c in self.cf.customer_articles]

    def blended_chunk(self, warm_ids, alpha, beta) -> np.ndarray:
        """Normalized, weighted blend scores for a chunk of warm customers."""
        Cn = _row_minmax(self.content.score_chunk(warm_ids))
        Fn = _row_minmax(cf_score_chunk(self.cf, warm_ids))
        return alpha * Cn + beta * Fn

    def recommend(self, customer_id, k=12, alpha=None, beta=None, include_repeats=False):
        a, b = self._weights(alpha, beta)
        if customer_id not in self.content.customer_articles or \
                customer_id not in self.cf.customer_articles:
            return self.popularity_model.recommend(customer_id, k=k)  # cold-start
        S = self.blended_chunk([customer_id], a, b)[0]
        bought = self.content.customer_articles[customer_id][0]
        return topk_dense(S, self.article_ids, bought, k, include_repeats,
                          self.popularity_model.ranked_articles)

    def recommend_all(self, customer_ids, k=24, alpha=None, beta=None,
                      include_repeats=False, batch_size=1000) -> dict:
        a, b = self._weights(alpha, beta)
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
            S = self.blended_chunk(chunk, a, b)
            for c, row in zip(chunk, S):
                bought = self.content.customer_articles[c][0]
                recs[c] = topk_dense(row, self.article_ids, bought, k,
                                     include_repeats, pop_ranked)
        return recs
