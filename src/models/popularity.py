"""Non-personalized popularity baseline for the NBA Decisioning Engine.

The bar every later model must beat, and the cold-start fallback. Popularity is
a single global ranking of actions; every customer — including cold-start
customers with no history — receives the same top-k.

LEAKAGE GUARANTEE: popularity is computed from **feature-side events only**
(``features_events.parquet``, all t_dat < CUTOFF_DATE). It never reads the label
window. ``PopularityModel.fit`` asserts the events it is handed end before the
cutoff, so label data cannot leak into the ranking.

Two popularity signals are computed:
    - purchase_count    : number of purchase events per action (canonical)
    - distinct_customers: number of distinct customers who bought the action
Ties are broken by action_id for a stable, reproducible ranking.
"""

from __future__ import annotations

import pandas as pd

from src import config


class PopularityModel:
    """Global popularity ranking with a non-personalized recommend()."""

    def __init__(self):
        self.ranked_actions = []            # canonical ranking (purchase_count)
        self.ranked_by_customers = []       # alt ranking (distinct_customers)
        self.popularity = None              # per-action stats DataFrame

    def fit(self, feature_events: pd.DataFrame) -> "PopularityModel":
        """Fit popularity on feature-side events. Asserts no label leakage."""
        if "t_dat" in feature_events.columns:
            cutoff = pd.Timestamp(config.CUTOFF_DATE)
            assert feature_events["t_dat"].max() < cutoff, (
                "popularity received events on/after the cutoff — label leakage! "
                "Fit on features_events.parquet only."
            )

        purchase_count = feature_events.groupby("action_id").size().rename("purchase_count")
        distinct_customers = (
            feature_events.groupby("action_id")["customer_id"].nunique().rename("distinct_customers")
        )
        pop = pd.concat([purchase_count, distinct_customers], axis=1).reset_index()

        # Canonical ranking: purchase volume, then action_id for stable ties.
        self.popularity = pop
        self.ranked_actions = (
            pop.sort_values(["purchase_count", "action_id"], ascending=[False, True])
            ["action_id"].tolist()
        )
        self.ranked_by_customers = (
            pop.sort_values(["distinct_customers", "action_id"], ascending=[False, True])
            ["action_id"].tolist()
        )
        return self

    def recommend(self, customer_id=None, k: int = 12):
        """Top-k popular actions — identical for every customer (non-personalized).

        Works for cold-start customers (customer_id is ignored by design).
        """
        return list(self.ranked_actions[:k])

    def recommend_all(self, customer_ids, k: int = 24) -> dict:
        """Map every customer to the same global top-k list."""
        topk = self.recommend(k=k)
        return {cid: topk for cid in customer_ids}


def load_feature_events() -> pd.DataFrame:
    """Load the feature-side event log (pre-cutoff only)."""
    return pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
