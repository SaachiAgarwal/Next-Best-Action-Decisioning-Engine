"""Tests for the consolidated metrics summary."""

import pandas as pd
import pytest

from src import config
from src.eval import metrics

SUMMARY = config.PROCESSED_DIR / "metrics_summary.parquet"

ARTICLE_STATIC = [
    "article popularity", "neighborhood CF (Exp B)", "content (Exp 4)",
    "content+CF hybrid (Exp 4)", "MF (Exp 5)", "triple hybrid (Exp 5, production)",
]
DIAG_MODELS = [
    "article popularity", "neighborhood CF (Exp B)", "content (Exp 4)",
    "MF (Exp 5)", "triple hybrid (Exp 5, production)",
]


def _load():
    if not SUMMARY.exists():
        pytest.skip("metrics_summary.parquet not built yet — run src.run_metrics_table")
    return pd.read_parquet(SUMMARY, engine="pyarrow")


# --------------------------------------------------------------------------
# Recall / precision math (synthetic, independent of the run)
# --------------------------------------------------------------------------
def test_recall_precision_math_synthetic():
    recs = {"a": [1, 2, 3, 4]}
    labels = {"a": {1, 2}}          # bought 2 of the 4 recommended
    assert metrics.recall_at_k(recs, labels, 4) == pytest.approx(2 / 2)   # both covered
    assert metrics.precision_at_k(recs, labels, 4) == pytest.approx(2 / 4)
    # A miss beyond k reduces recall.
    recs2 = {"a": [1, 9, 9, 9]}
    assert metrics.recall_at_k(recs2, {"a": {1, 2}}, 4) == pytest.approx(1 / 2)


# --------------------------------------------------------------------------
# Table completeness
# --------------------------------------------------------------------------
def test_all_article_models_present():
    df = _load()
    art = df[df["regime"] == "article"]
    for m in ARTICLE_STATIC:
        assert m in set(art["model"]), f"missing article model: {m}"
    assert any("shared bandit" in m for m in art["model"])   # bandit row present + flagged


def test_all_product_type_models_present():
    df = _load()
    pt = df[df["regime"] == "product-type"]
    for m in ["popularity (Exp A)", "item-CF (Exp A)", "recency+freq hybrid (Exp 3, production)"]:
        assert m in set(pt["model"])
    assert any("LinUCB v3" in m for m in pt["model"])


def test_full_accuracy_metrics_populated_for_static_models():
    """Every static (non-bandit) article model has hit/recall/precision @ all k."""
    df = _load()
    art = df[(df["regime"] == "article") & (df["model"].isin(ARTICLE_STATIC))]
    for k in (6, 12, 24):
        for col in (f"hit@{k}", f"recall@{k}", f"precision@{k}"):
            assert art[col].notna().all(), f"{col} has gaps for static models"


# --------------------------------------------------------------------------
# Diagnostics join kept every row (no drops)
# --------------------------------------------------------------------------
def test_diagnostics_join_no_dropped_rows():
    df = _load()
    art = df[df["regime"] == "article"]
    # Left join must not drop accuracy rows.
    assert len(art) >= len(ARTICLE_STATIC)
    # The 5 diagnostics-covered models have coverage populated.
    for m in DIAG_MODELS:
        cov = art.loc[art["model"] == m, "coverage_pct@12"].iloc[0]
        assert pd.notna(cov), f"{m} missing coverage from the diagnostics join"


# --------------------------------------------------------------------------
# Same evaluable set (except flagged bandits)
# --------------------------------------------------------------------------
def test_static_models_share_evaluable_set_bandits_flagged():
    df = _load()
    for _, r in df.iterrows():
        if "†" in r["model"]:
            assert r["n"] == 4574          # bandits: held-out subset, flagged
        else:
            assert r["n"] == 15246         # everything else: the core set


# --------------------------------------------------------------------------
# Prior artifacts intact
# --------------------------------------------------------------------------
def test_prior_experiment_artifacts_intact():
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
