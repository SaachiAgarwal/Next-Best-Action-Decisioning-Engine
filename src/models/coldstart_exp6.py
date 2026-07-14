"""Experiment 6b — attribute-based cold-start recommender.

For a customer with NO purchase history, attributes are the only signal. We score
articles by the customer's segment lift (Exp 6a) blended with a global popularity
prior (a thin/unusual segment falls back safely to popularity):

    score(customer, article) = v1 * segment_lift(customer, article) + v2 * popularity(article)

Both components are min-max normalized. Customers with unknown/missing attributes
degrade gracefully to pure popularity (their segment lift is the 'unknown' row,
≈ global, so it adds no distinctive signal). ``v1/v2`` are tuned by *simulating*
cold-start on warm customers (masking their history) — documented in the runner.
"""

from __future__ import annotations

import numpy as np

from src.models.content_based_exp4 import _row_minmax, topk_dense
from src.models.hybrid_attrs_exp6 import ATTRS


class ColdStartModel:
    def __init__(self, seglift, popularity_model):
        self.seglift = seglift
        self.popularity_model = popularity_model
        self.article_ids = seglift.article_ids
        # Normalized popularity vector aligned to the article order.
        counts = dict(zip(popularity_model.popularity["article_id"],
                          popularity_model.popularity["purchase_count"]))
        pop = np.array([counts.get(a, 0) for a in self.article_ids], dtype=np.float64)
        self.pop_norm = _row_minmax(pop.reshape(1, -1)).ravel().astype(np.float32)

    def has_known_attributes(self, customer_id) -> bool:
        attrs = self.seglift.customer_attrs.get(customer_id)
        if not attrs:
            return False
        # Known if at least one raw attribute value is not 'unknown'.
        return any(str(v) != "unknown" for v in attrs.values())

    def score_chunk(self, customer_ids, v1, v2) -> np.ndarray:
        attr = _row_minmax(self.seglift.score_chunk(customer_ids))
        return v1 * attr + v2 * self.pop_norm

    def recommend(self, customer_id, k=12, v1=1.0, v2=1.0):
        if not self.has_known_attributes(customer_id):
            return self.popularity_model.recommend(customer_id, k=k)  # pure popularity fallback
        S = self.score_chunk([customer_id], v1, v2)[0]
        return topk_dense(S, self.article_ids, np.array([], dtype=np.int64), k,
                          include_repeats=True, pop_ranked=self.popularity_model.ranked_articles)

    def recommend_all(self, customer_ids, k=24, v1=1.0, v2=1.0, batch_size=1000):
        customer_ids = list(customer_ids)
        known = [c for c in customer_ids if self.has_known_attributes(c)]
        known_set = set(known)
        recs = {}
        pop_topk = self.popularity_model.recommend(k=k)
        for c in customer_ids:
            if c not in known_set:
                recs[c] = list(pop_topk)
        pop_ranked = self.popularity_model.ranked_articles
        empty = np.array([], dtype=np.int64)
        for s in range(0, len(known), batch_size):
            chunk = known[s:s + batch_size]
            S = self.score_chunk(chunk, v1, v2)
            for c, row in zip(chunk, S):
                recs[c] = topk_dense(row, self.article_ids, empty, k, True, pop_ranked)
        return recs
