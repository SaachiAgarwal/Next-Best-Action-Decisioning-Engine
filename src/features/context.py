"""Customer context / feature layer (Phase 1).

Builds the per-customer **context vector** the contextual bandit conditions on:
one row per customer, computed STRICTLY from pre-cutoff (feature-side) data.

Design note (deliberate architecture decision): the context is **aggregate
customer-state** — "who is this customer" (how recent/frequent/valuable, how
broad their taste, their attributes) — and is intentionally DISTINCT from the
recommender's per-action affinity scores ("which action is good"). Keeping them
separate means the bandit's context complements the candidate scorer instead of
duplicating its signal: the scorer proposes actions, the context describes the
person the policy is deciding for.

Everything here derives only from events with ``t_dat < CUTOFF_DATE``; ``build``
asserts this (leakage guard). ``price`` is normalized to [0, 1] in the source, so
the monetary features are a *relative engagement* signal, not currency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

AGE_BINS = [-np.inf, 25, 35, 45, 55, np.inf]
AGE_LABELS = ["<=25", "26-35", "36-45", "46-55", "56+"]
UNKNOWN = "unknown"
COLD_DOMINANT_ACTION = -1  # sentinel: cold-start customer has no dominant action

CATEGORICAL_CONTEXT = ["age_band", "club_member_status", "fashion_news_frequency"]
NUMERIC_CONTEXT = [
    "recency_days", "frequency", "monetary_total", "monetary_avg",
    "distinct_actions", "tenure_days", "avg_repurchase_gap_days",
    "dominant_action_share", "age",
]


def _age_band(age: pd.Series) -> pd.Series:
    band = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)
    return band.astype("object").where(age.notna(), UNKNOWN).astype("string")


def build(feature_events: pd.DataFrame, customers: pd.DataFrame,
          reference_date=None) -> pd.DataFrame:
    """Assemble the raw (human-readable) customer context table (one row/customer).

    Includes every customer in ``customers`` — those with pre-cutoff history get
    computed features; cold-start customers (no history) get explicit safe
    defaults and ``is_cold_start=True`` rather than being dropped.
    """
    ref = pd.Timestamp(reference_date or config.CUTOFF_DATE)
    assert feature_events["t_dat"].max() < ref, (
        f"context received events on/after {ref.date()} — leakage! "
        "Build from features_events.parquet (pre-cutoff) only."
    )

    fe = feature_events
    g = fe.groupby("customer_id", sort=False)

    # --- Task 1: RFM (customer-level) ---
    agg = g.agg(
        last_dat=("t_dat", "max"),
        first_dat=("t_dat", "min"),
        frequency=("t_dat", "size"),
        monetary_total=("price", "sum"),
        monetary_avg=("price", "mean"),
        distinct_actions=("action_id", "nunique"),
    )
    agg["recency_days"] = (ref - agg["last_dat"]).dt.days
    agg["tenure_days"] = (ref - agg["first_dat"]).dt.days
    # Mean consecutive-purchase gap telescopes to (last-first)/(n-1); 0 if single.
    span = (agg["last_dat"] - agg["first_dat"]).dt.days
    agg["avg_repurchase_gap_days"] = np.where(
        agg["frequency"] >= 2, span / (agg["frequency"] - 1), 0.0)

    # --- Task 3: dominant action (most-purchased; deterministic tie-break) ---
    ac = fe.groupby(["customer_id", "action_id"], sort=False).size().reset_index(name="n")
    ac = ac.sort_values(["customer_id", "n", "action_id"], ascending=[True, False, True])
    dom = ac.drop_duplicates("customer_id", keep="first").set_index("customer_id")
    agg["dominant_action_id"] = dom["action_id"]              # aligns on customer_id index
    agg["dominant_action_share"] = dom["n"] / agg["frequency"]

    agg = agg.drop(columns=["last_dat", "first_dat"]).reset_index()

    # --- Base = every customer (so cold-start customers are retained) ---
    ctx = customers[["customer_id", "age", "club_member_status",
                     "fashion_news_frequency"]].copy()
    ctx = ctx.merge(agg, on="customer_id", how="left")

    # Cold-start flag: customers with no pre-cutoff history.
    ctx["is_cold_start"] = ctx["frequency"].isna()

    # --- Task 2: attributes (nulls -> "unknown", never dropped) ---
    ctx["age"] = ctx["age"].astype("float64")
    ctx["age_band"] = _age_band(ctx["age"])
    for c in ("club_member_status", "fashion_news_frequency"):
        ctx[c] = ctx[c].astype("string").fillna(UNKNOWN)

    # --- Cold-start-safe defaults (recency/tenure stay NaN, flagged) ---
    zero_fill = {
        "frequency": 0, "monetary_total": 0.0, "monetary_avg": 0.0,
        "distinct_actions": 0, "avg_repurchase_gap_days": 0.0,
        "dominant_action_share": 0.0,
    }
    ctx = ctx.fillna(zero_fill)
    ctx["dominant_action_id"] = ctx["dominant_action_id"].fillna(COLD_DOMINANT_ACTION).astype("int64")
    ctx["frequency"] = ctx["frequency"].astype("int64")
    ctx["distinct_actions"] = ctx["distinct_actions"].astype("int64")
    ctx["customer_id"] = ctx["customer_id"].astype("string")

    col_order = ["customer_id", "is_cold_start",
                 "recency_days", "frequency", "monetary_total", "monetary_avg",
                 "distinct_actions", "tenure_days", "avg_repurchase_gap_days",
                 "dominant_action_id", "dominant_action_share",
                 "age", "age_band", "club_member_status", "fashion_news_frequency"]
    return ctx[col_order]


def build_model_ready(context: pd.DataFrame):
    """Deterministically encode the raw context into a model-ready numeric matrix.

    Numeric features are median-imputed (cold-start recency/tenure/age have NaN)
    then standardized; categoricals are one-hot encoded. Returns (matrix, scaler,
    feature_names). No NaN leaks into the encoded matrix.
    """
    df = context.copy()

    # Impute numeric NaNs (cold-start recency/tenure, missing age). The
    # is_cold_start flag preserves the "no history" signal explicitly.
    num = df[NUMERIC_CONTEXT].astype("float64")
    medians = num.median(numeric_only=True)
    num = num.fillna(medians)

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(num), columns=NUMERIC_CONTEXT, index=df.index)

    cats = pd.get_dummies(df[CATEGORICAL_CONTEXT].astype("string"),
                          prefix=CATEGORICAL_CONTEXT, dtype="float64")

    matrix = pd.concat([
        df[["customer_id"]].reset_index(drop=True),
        df[["is_cold_start"]].astype("float64").reset_index(drop=True),
        scaled.reset_index(drop=True),
        cats.reset_index(drop=True),
    ], axis=1)
    feature_names = ["is_cold_start"] + NUMERIC_CONTEXT + list(cats.columns)
    return matrix, scaler, feature_names


def load_inputs():
    fe = pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    customers = pd.read_parquet(config.PROCESSED_DIR / "customers.parquet", engine="pyarrow")
    return fe, customers
