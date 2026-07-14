"""Experiment 6a — customer attributes as a segment-lift signal in the hybrid.

The core design problem: **attributes describe the customer, but the model scores
articles.** We bridge it with **segment lift** — which articles a customer's
demographic segment buys *disproportionately* relative to global demand.

    p_seg(a)  = (count(a in segment) + K·p_global(a)) / (segment purchases + K)   [smoothed]
    lift(a,s) = p_seg(a) / p_global(a)            (>1: over-bought by the segment)

Using **lift, not raw counts** is essential: raw segment popularity would just
re-derive global popularity (every segment buys the popular articles most). Lift
isolates what is *distinctive* about a segment. The smoothing K shrinks thin
segments toward global (lift→1), so noise doesn't masquerade as signal.

A customer's ``attr_score`` for an article is the mean lift across their three
attribute segments (age_band, club_member_status, fashion_news_frequency).
Everything is computed from pre-cutoff data only (leakage guard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.models.content_based_exp4 import _row_minmax, topk_dense
from src.models.hybrid_content_cf_exp4 import cf_score_chunk

ATTRS = ["age_band", "club_member_status", "fashion_news_frequency"]
SMOOTHING = 20.0


class SegmentLift:
    """Per-(customer, article) attribute affinity via smoothed segment lift."""

    def __init__(self):
        self.article_ids = None
        self.article_index = None
        self.lift = {}            # attr -> (n_segments x n_articles) lift matrix
        self.seg_index = {}       # attr -> {segment_value: row}
        self.customer_seg = {}    # customer_id -> {attr: row index}

    def fit(self, feature_events, context, article_order, smoothing=SMOOTHING):
        if "t_dat" in feature_events.columns:
            assert feature_events["t_dat"].max() < pd.Timestamp(config.CUTOFF_DATE), \
                "segment lift saw post-cutoff events — leakage!"
        self.article_ids = np.asarray(article_order)
        self.article_index = {a: i for i, a in enumerate(self.article_ids)}
        n_art = len(self.article_ids)

        ctx = context[["customer_id"] + ATTRS].copy()
        for a in ATTRS:
            ctx[a] = ctx[a].astype("string").fillna("unknown")

        pairs = feature_events[["customer_id", "article_id"]].copy()
        pairs["article_id"] = pairs["article_id"].astype("string")
        pairs = pairs[pairs["article_id"].isin(self.article_index)]
        pairs["ai"] = pairs["article_id"].map(self.article_index).to_numpy()
        pairs = pairs.merge(ctx, on="customer_id", how="left")
        for a in ATTRS:
            pairs[a] = pairs[a].fillna("unknown")

        total = len(pairs)
        gcount = np.zeros(n_art)
        vc = pairs["ai"].value_counts()
        gcount[vc.index.to_numpy()] = vc.to_numpy()
        p_global = gcount / total
        p_global[p_global == 0] = 1.0 / total   # articles present always have >=1

        for attr in ATTRS:
            segs = sorted(pairs[attr].unique())
            self.seg_index[attr] = {s: i for i, s in enumerate(segs)}
            mat = np.zeros((len(segs), n_art))
            g = pairs.groupby([attr, "ai"], sort=False).size().reset_index(name="c")
            mat[g[attr].map(self.seg_index[attr]).to_numpy(), g["ai"].to_numpy()] = g["c"].to_numpy()
            seg_total = mat.sum(axis=1, keepdims=True)
            p_seg = (mat + smoothing * p_global[None, :]) / (seg_total + smoothing)
            self.lift[attr] = (p_seg / p_global[None, :]).astype(np.float32)

        # Customer -> segment row per attribute, and the raw attribute values (so
        # cold-start can tell a genuine value from 'unknown' even when 'unknown'
        # never occurs among purchasers and so is not a fitted segment).
        self.customer_seg = {}
        self.customer_attrs = {}
        for r in ctx.itertuples(index=False):
            cid = r.customer_id
            self.customer_attrs[cid] = {attr: getattr(r, attr, "unknown") for attr in ATTRS}
            self.customer_seg[cid] = {
                attr: self.seg_index[attr].get(getattr(r, attr, "unknown"),
                                               self.seg_index[attr].get("unknown", 0))
                for attr in ATTRS
            }
        return self

    def score_chunk(self, customer_ids) -> np.ndarray:
        """Mean segment lift across the 3 attributes, per (customer, article)."""
        n = len(customer_ids)
        acc = np.zeros((n, len(self.article_ids)), dtype=np.float32)
        for attr in ATTRS:
            idx = np.array([self.customer_seg.get(c, {}).get(attr, 0) for c in customer_ids])
            acc += self.lift[attr][idx]
        return acc / len(ATTRS)

    def distinctive_articles(self, attr, segment, n=5):
        """Top-n highest-lift (most distinctive) articles for a segment."""
        i = self.seg_index[attr][segment]
        order = np.argsort(-self.lift[attr][i])[:n]
        return [(str(self.article_ids[j]), float(self.lift[attr][i][j])) for j in order]


class FourSignalHybrid:
    """content + CF + MF + attribute segment-lift. w4=0 == the Exp 5 triple hybrid."""

    def __init__(self, content_model, cf_model, mf_model, seglift):
        self.content = content_model
        self.cf = cf_model
        self.mf = mf_model
        self.seglift = seglift
        self.article_ids = content_model.article_ids
        self.popularity_model = content_model.popularity_model

    def _warm(self, customer_ids):
        return [c for c in customer_ids
                if c in self.content.customer_articles
                and c in self.cf.customer_articles
                and c in self.mf.customer_index]

    def blended_chunk(self, warm_ids, w1, w2, w3, w4):
        Cn = _row_minmax(self.content.score_chunk(warm_ids))
        Fn = _row_minmax(cf_score_chunk(self.cf, warm_ids))
        Mn = _row_minmax(self.mf.score_chunk(warm_ids))
        S = w1 * Cn + w2 * Fn + w3 * Mn
        if w4 != 0:
            S = S + w4 * _row_minmax(self.seglift.score_chunk(warm_ids))
        return S

    def recommend(self, customer_id, k=12, w1=1.0, w2=1.0, w3=1.0, w4=0.0, include_repeats=False):
        if customer_id not in self._warm([customer_id]):
            return self.popularity_model.recommend(customer_id, k=k)
        S = self.blended_chunk([customer_id], w1, w2, w3, w4)[0]
        bought = self.content.customer_articles[customer_id][0]
        return topk_dense(S, self.article_ids, bought, k, include_repeats,
                          self.popularity_model.ranked_articles)

    def recommend_all(self, customer_ids, k=24, w1=1.0, w2=1.0, w3=1.0, w4=0.0,
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
        for s in range(0, len(warm), batch_size):
            chunk = warm[s:s + batch_size]
            S = self.blended_chunk(chunk, w1, w2, w3, w4)
            for c, row in zip(chunk, S):
                bought = self.content.customer_articles[c][0]
                recs[c] = topk_dense(row, self.article_ids, bought, k, include_repeats, pop_ranked)
        return recs
