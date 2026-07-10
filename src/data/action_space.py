"""Action space definition for the NBA Decisioning Engine.

Defines the set of actions (candidate recommendations) the engine can take,
aggregated at ``config.ACTION_GRANULARITY`` (``product_type_name``). Product-type
granularity trades recommendation precision against statistical learnability:
it avoids article-level sparsity while staying more actionable than broad
product groups.

Outputs (both saved to data/processed/ as parquet):
  - actions.parquet            : one row per action, with volume/reach stats
  - article_action_map.parquet : article_id -> action, for later drill-down

Any article whose ``product_type_name`` is null is assigned an explicit
"unknown" action bucket rather than being dropped.
"""

from __future__ import annotations

import pandas as pd

from src import config

UNKNOWN_ACTION = "unknown"
LONG_TAIL_THRESHOLD = 100  # actions with fewer purchases than this are "long tail"


def build_action_space(transactions, articles):
    """Build the actions table and the article->action mapping.

    Returns
    -------
    (actions, article_action_map, stats) : tuple
        actions              -- action_id, product_type_name, article_count,
                                 total_purchases, distinct_customers
        article_action_map   -- article_id, product_type_name, action_id
        stats                -- dict with action_space_size, top15, long_tail
    """
    gran = config.ACTION_GRANULARITY  # "product_type_name"

    # 1. article -> action label. Null product types get an explicit bucket.
    art = articles[["article_id", gran]].copy()
    art[gran] = art[gran].astype("string").fillna(UNKNOWN_ACTION)
    art = art.rename(columns={gran: "product_type_name"})

    # 2. Ensure every article that appears in transactions has a mapping.
    #    (Sampling guarantees this, but assign "unknown" to any stray id rather
    #    than silently dropping it.)
    tx_article_ids = pd.Index(transactions["article_id"].unique())
    missing = tx_article_ids.difference(pd.Index(art["article_id"]))
    if len(missing) > 0:
        art = pd.concat([
            art,
            pd.DataFrame({
                "article_id": pd.array(missing, dtype="string"),
                "product_type_name": UNKNOWN_ACTION,
            }),
        ], ignore_index=True)

    # 3. Stable integer action_id: sort action names, assign 0..n-1.
    action_names = sorted(art["product_type_name"].dropna().unique())
    action_id_map = {name: i for i, name in enumerate(action_names)}
    art["action_id"] = art["product_type_name"].map(action_id_map).astype("int64")

    article_action_map = art[["article_id", "product_type_name", "action_id"]].copy()
    article_action_map["article_id"] = article_action_map["article_id"].astype("string")

    # 4. Aggregate transaction volume / reach per action.
    tx = transactions.merge(
        article_action_map[["article_id", "action_id"]], on="article_id", how="left"
    )
    total_purchases = tx.groupby("action_id").size().rename("total_purchases")
    distinct_customers = (
        tx.groupby("action_id")["customer_id"].nunique().rename("distinct_customers")
    )
    article_count = (
        article_action_map.groupby("action_id").size().rename("article_count")
    )

    # 5. Assemble the actions table.
    actions = pd.DataFrame({
        "action_id": [action_id_map[n] for n in action_names],
        "product_type_name": action_names,
    })
    actions = (
        actions
        .merge(article_count, on="action_id", how="left")
        .merge(total_purchases, on="action_id", how="left")
        .merge(distinct_customers, on="action_id", how="left")
    )
    for col in ("article_count", "total_purchases", "distinct_customers"):
        actions[col] = actions[col].fillna(0).astype("int64")
    actions["product_type_name"] = actions["product_type_name"].astype("string")
    actions = actions.sort_values("total_purchases", ascending=False).reset_index(drop=True)

    # 6. Stats for reporting.
    top15 = actions.head(15)[
        ["action_id", "product_type_name", "article_count", "total_purchases", "distinct_customers"]
    ]
    long_tail = actions[actions["total_purchases"] < LONG_TAIL_THRESHOLD]
    stats = {
        "action_space_size": len(actions),
        "top15": top15,
        "long_tail_threshold": LONG_TAIL_THRESHOLD,
        "long_tail_count": len(long_tail),
    }
    return actions, article_action_map, stats


def save_action_space(actions, article_action_map) -> None:
    """Persist the actions table and article->action map as parquet."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    actions.to_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
    article_action_map.to_parquet(
        config.PROCESSED_DIR / "article_action_map.parquet", engine="pyarrow"
    )


def print_action_space(stats) -> None:
    """Print the action-space size and the top 15 actions by purchase volume."""
    print("=" * 70)
    print("ACTION SPACE (granularity = product_type_name)")
    print("=" * 70)
    print(f"  Total actions (action-space size): {stats['action_space_size']}")
    print(f"  Long-tail actions (<{stats['long_tail_threshold']} purchases): "
          f"{stats['long_tail_count']}")
    print("\n  Top 15 actions by total_purchases:")
    print(stats["top15"].to_string(index=False))
