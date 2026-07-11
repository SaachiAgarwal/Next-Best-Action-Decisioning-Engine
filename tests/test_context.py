"""Tests for the Phase 1 customer context / feature layer.

A small synthetic case (customers with history, a cold-start customer, and
null attributes) lets us check the RFM math, cold-start retention, null
handling, the leakage guard, and the model-ready encoding exactly.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import context as ctx_mod

CUTOFF = pd.Timestamp(config.CUTOFF_DATE)


def _feature_events():
    """c1: 3 buys (action 5 x2, action 7 x1); c2: 1 buy (action 5). c3: none."""
    rows = [
        ("c1", 5, 0.1, CUTOFF - pd.Timedelta(days=40)),
        ("c1", 5, 0.2, CUTOFF - pd.Timedelta(days=25)),
        ("c1", 7, 0.3, CUTOFF - pd.Timedelta(days=10)),
        ("c2", 5, 0.5, CUTOFF - pd.Timedelta(days=5)),
    ]
    return pd.DataFrame({
        "customer_id": pd.array([r[0] for r in rows], dtype="string"),
        "action_id": [r[1] for r in rows],
        "price": np.array([r[2] for r in rows], dtype="float32"),
        "t_dat": [r[3] for r in rows],
    })


def _customers():
    return pd.DataFrame({
        "customer_id": pd.array(["c1", "c2", "c3"], dtype="string"),
        "age": [30.0, np.nan, 60.0],                 # c2 age null
        "club_member_status": pd.array(["ACTIVE", None, "ACTIVE"], dtype="string"),
        "fashion_news_frequency": pd.array(["Regularly", None, "NONE"], dtype="string"),
    })


@pytest.fixture
def ctx():
    return ctx_mod.build(_feature_events(), _customers())


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def test_one_row_per_customer_no_duplicates(ctx):
    assert len(ctx) == 3
    assert ctx["customer_id"].is_unique


# --------------------------------------------------------------------------
# RFM correctness
# --------------------------------------------------------------------------
def test_rfm_computed_correctly(ctx):
    c1 = ctx.set_index("customer_id").loc["c1"]
    assert c1["frequency"] == 3
    assert c1["monetary_total"] == pytest.approx(0.6, abs=1e-5)
    assert c1["monetary_avg"] == pytest.approx(0.2, abs=1e-5)
    assert c1["recency_days"] == 10          # last buy 10d before cutoff
    assert c1["tenure_days"] == 40           # first buy 40d before cutoff
    assert c1["distinct_actions"] == 2       # actions 5 and 7
    assert c1["avg_repurchase_gap_days"] == pytest.approx(15.0)  # span 30 / (3-1)
    assert c1["dominant_action_id"] == 5     # bought twice
    assert c1["dominant_action_share"] == pytest.approx(2 / 3)


def test_single_purchase_gap_is_zero(ctx):
    c2 = ctx.set_index("customer_id").loc["c2"]
    assert c2["frequency"] == 1
    assert c2["avg_repurchase_gap_days"] == 0.0  # convention for single purchase
    assert c2["dominant_action_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Cold-start & null handling
# --------------------------------------------------------------------------
def test_cold_start_retained_with_safe_defaults(ctx):
    c3 = ctx.set_index("customer_id").loc["c3"]
    assert c3["is_cold_start"]
    assert c3["frequency"] == 0
    assert c3["monetary_total"] == 0.0
    assert c3["distinct_actions"] == 0
    assert c3["dominant_action_id"] == -1
    assert pd.isna(c3["recency_days"])  # flagged, not fabricated


def test_null_attributes_become_unknown_not_dropped(ctx):
    c2 = ctx.set_index("customer_id").loc["c2"]
    assert c2["club_member_status"] == "unknown"
    assert c2["fashion_news_frequency"] == "unknown"
    assert c2["age_band"] == "unknown"  # age was null


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
def test_build_rejects_post_cutoff_events():
    fe = _feature_events()
    fe.loc[len(fe)] = ("c1", 5, np.float32(0.1), CUTOFF)  # event AT cutoff
    with pytest.raises(AssertionError):
        ctx_mod.build(fe, _customers())


# --------------------------------------------------------------------------
# Model-ready encoding
# --------------------------------------------------------------------------
def test_model_ready_no_nans_scaled_and_encoded(ctx):
    matrix, scaler, feature_names = ctx_mod.build_model_ready(ctx)
    assert len(matrix) == len(ctx)
    # No NaNs anywhere in the encoded matrix (cold-start defaults applied).
    assert not matrix.drop(columns=["customer_id"]).isna().any().any()
    # Categoricals one-hot encoded (unknown bands present as columns).
    assert any(col.startswith("club_member_status_") for col in feature_names)
    assert any(col.startswith("age_band_") for col in feature_names)
    # Numerics standardized: near-zero mean across rows.
    assert abs(matrix["frequency"].mean()) < 1e-6
