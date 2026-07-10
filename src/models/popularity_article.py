"""Article-level popularity baseline (Experiment B).

Identical logic to the product-type popularity baseline, but the "action" is now
``article_id`` directly (~79k feature-side articles instead of 128 product types).
This is the non-personalized bar and cold-start fallback for the article-level
granularity experiment.

LEAKAGE GUARANTEE: computed from feature-side events only
(``features_events.parquet``, all t_dat < CUTOFF_DATE); ``fit`` asserts it.
"""

from __future__ import annotations

import pandas as pd

from src import config


class ArticlePopularityModel:
    """Global article popularity ranking; non-personalized recommend()."""

    def __init__(self):
        self.ranked_articles = []   # article_id, most popular first
        self.popularity = None      # per-article purchase_count

    def fit(self, feature_events: pd.DataFrame) -> "ArticlePopularityModel":
        if "t_dat" in feature_events.columns:
            cutoff = pd.Timestamp(config.CUTOFF_DATE)
            assert feature_events["t_dat"].max() < cutoff, (
                "article popularity received events on/after the cutoff — leakage!"
            )
        counts = feature_events.groupby("article_id").size().rename("purchase_count")
        pop = counts.reset_index()
        # Rank by volume, then article_id for a stable, reproducible tie-break.
        pop = pop.sort_values(["purchase_count", "article_id"], ascending=[False, True])
        self.popularity = pop
        self.ranked_articles = pop["article_id"].tolist()
        return self

    def recommend(self, customer_id=None, k: int = 12):
        """Top-k popular articles — identical for every customer (cold-start safe)."""
        return list(self.ranked_articles[:k])

    def recommend_all(self, customer_ids, k: int = 24) -> dict:
        topk = self.recommend(k=k)
        return {cid: topk for cid in customer_ids}


def load_feature_events() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
