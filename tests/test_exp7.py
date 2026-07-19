"""Tests for Experiment 7 — temporal signals.

Fast, synthetic where possible. The central guard: the temporal hybrid with
w4=w5=0 and no due modifier reproduces the Exp 5 triple hybrid ranking EXACTLY.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.temporal_exp7 import (
    TemporalSignals, TemporalHybrid, compute_due, assign_band)


def _articles(ids):
    return np.array([f"{i:07d}" for i in ids])


# --------------------------------------------------------------------------
# Signal A — trend uses the last TREND_WINDOW_DAYS before ref only (leakage)
# --------------------------------------------------------------------------
def test_trend_uses_recent_window_only():
    ref = pd.Timestamp("2020-08-26")
    aids = _articles([0, 1, 2])
    rows = [
        # article 0: only OLD purchases (outside 30d window) -> trend 0
        {"article_id": aids[0], "t_dat": ref - pd.Timedelta(days=200)},
        {"article_id": aids[0], "t_dat": ref - pd.Timedelta(days=100)},
        # article 1: only RECENT purchases -> highest trend
        {"article_id": aids[1], "t_dat": ref - pd.Timedelta(days=5)},
        {"article_id": aids[1], "t_dat": ref - pd.Timedelta(days=10)},
        # article 2: one recent
        {"article_id": aids[2], "t_dat": ref - pd.Timedelta(days=2)},
        # a POST-ref event that must be ignored entirely (leakage guard)
        {"article_id": aids[0], "t_dat": ref + pd.Timedelta(days=1)},
    ]
    fe = pd.DataFrame(rows)
    sig = TemporalSignals(aids).fit(fe, reference_date=ref)
    assert sig.recent_counts[0] == 0            # old-only article: no recent demand
    assert sig.recent_counts[1] == 2
    assert sig.trend_norm[1] == 1.0             # most recent-popular normalizes to 1
    assert sig.alltime_counts[0] == 2           # post-ref event excluded from all-time too


# --------------------------------------------------------------------------
# Signal B — seasonal profile uses pre-cutoff months only; smoothing works
# --------------------------------------------------------------------------
def test_seasonal_profile_precutoff_and_smoothing():
    ref = pd.Timestamp("2020-08-26")
    aids = _articles([0, 1])
    rows = []
    # article 0: many purchases, all in September (month 9) -> spiked profile
    for _ in range(50):
        rows.append({"article_id": aids[0], "t_dat": pd.Timestamp("2019-09-15")})
    # article 1: a SINGLE June purchase -> profile must shrink toward global
    rows.append({"article_id": aids[1], "t_dat": pd.Timestamp("2019-06-15")})
    # leakage bait: a September purchase AFTER ref must not be counted
    rows.append({"article_id": aids[1], "t_dat": ref + pd.Timedelta(days=10)})
    fe = pd.DataFrame(rows)
    sig = TemporalSignals(aids).fit(fe, reference_date=ref, smooth_k=20)

    # article 0 (50 Sept buys) is strongly Sept-skewed
    assert sig.article_month[0, 8] > 0.9
    # article 1 (1 June buy) is shrunk: its June mass is far from a spike (1.0)
    assert sig.article_month[1, 5] < 0.5
    # and it sits close to the global month distribution (dominated by Sept here)
    assert abs(sig.article_month[1].argmax() - 8) == 0 or sig.article_month[1, 5] < 0.3
    # leakage guard: article 1 has exactly ONE counted purchase (post-ref ignored)
    assert sig.alltime_counts[1] == 1


# --------------------------------------------------------------------------
# Signal C — due_ratio math + single-purchase fallback
# --------------------------------------------------------------------------
def test_due_ratio_math_and_single_purchase_fallback():
    ref = pd.Timestamp("2020-08-26")
    # customer A: buys every 10 days, last buy 10 days before ref -> due_ratio ~1
    rowsA = [{"customer_id": "A", "t_dat": ref - pd.Timedelta(days=d)} for d in (10, 20, 30, 40)]
    # customer B: single purchase -> no gap -> population-median-gap fallback, flagged
    rowsB = [{"customer_id": "B", "t_dat": ref - pd.Timedelta(days=15)}]
    el = pd.DataFrame(rowsA + rowsB)
    el["article_id"] = "x"
    df = compute_due(el, ["A", "B"], ref)
    a = df[df["customer_id"] == "A"].iloc[0]
    b = df[df["customer_id"] == "B"].iloc[0]
    assert a["typical_gap"] == 10.0 and a["days_since_last"] == 10
    assert abs(a["due_ratio"] - 1.0) < 1e-6 and not a["single_purchase"]
    assert b["single_purchase"]                       # flagged
    assert b["typical_gap"] == df.attrs["population_gap"] == 10.0   # pop median gap fallback


# --------------------------------------------------------------------------
# Contact bands — boundaries + one band per customer
# --------------------------------------------------------------------------
def test_band_boundaries():
    assert assign_band(0.0) == "just purchased"
    assert assign_band(0.39) == "just purchased"
    assert assign_band(0.4) == "approaching"
    assert assign_band(0.8) == "due now"
    assert assign_band(1.49) == "due now"
    assert assign_band(1.5) == "overdue"
    assert assign_band(3.0) == "lapsed"
    assert assign_band(99.0) == "lapsed"


def test_every_customer_gets_one_band():
    ref = pd.Timestamp("2020-08-26")
    rng = np.random.default_rng(0)
    rows = []
    for c in range(50):
        for _ in range(rng.integers(1, 5)):
            rows.append({"customer_id": f"C{c}", "article_id": "x",
                         "t_dat": ref - pd.Timedelta(days=int(rng.integers(1, 200)))})
    el = pd.DataFrame(rows)
    ids = [f"C{c}" for c in range(50)]
    df = compute_due(el, ids, ref)
    assert len(df) == len(ids)                        # one row per customer
    assert df["customer_id"].nunique() == len(ids)
    valid = {b[0] for b in config.CONTACT_BANDS}
    assert set(df["band"]).issubset(valid)
    assert df["band"].notna().all()


# --------------------------------------------------------------------------
# THE KEY REGRESSION GUARD: w4=w5=0, no due modifier == triple hybrid exactly
# --------------------------------------------------------------------------
class _FakeTriple:
    """Stand-in triple hybrid with a deterministic blended_chunk."""
    def __init__(self, article_ids):
        self.article_ids = np.asarray(article_ids)
        self.popularity_model = None
        self.content = None
        self._blend = {}

    def _warm(self, ids):
        return list(ids)

    def blended_chunk(self, warm_ids, w1, w2, w3):
        rng = np.random.default_rng(7)
        return np.vstack([rng.random(len(self.article_ids)) for _ in warm_ids])


def test_temporal_zero_weights_reproduces_triple():
    aids = _articles(range(10))
    triple = _FakeTriple(aids)
    sig = TemporalSignals(aids)
    # give the signals arbitrary non-zero content to prove they're truly unused
    sig.trend_norm = np.linspace(0, 1, 10)
    sig.momentum_norm = np.linspace(1, 0, 10)
    sig.season_norm = np.ones(10)
    sig.pop_norm = np.linspace(0, 1, 10)
    th = TemporalHybrid(triple, sig, due_ratios={})
    warm = ["a", "b", "c"]
    base = triple.blended_chunk(warm, 1.0, 0.5, 1.0)
    got = th.blended_chunk(warm, 1.0, 0.5, 1.0, w4=0.0, w5=0.0, due_modifier=False)
    assert np.array_equal(base, got)                  # EXACT, byte-for-byte
    # and any non-zero temporal weight DOES change the scores
    changed = th.blended_chunk(warm, 1.0, 0.5, 1.0, w4=0.5, w5=0.0, due_modifier=False)
    assert not np.array_equal(base, changed)


# --------------------------------------------------------------------------
# Prior artifacts intact (guard)
# --------------------------------------------------------------------------
def test_prior_artifacts_intact():
    pd_ = config.PROCESSED_DIR
    checks = {
        "actions.parquet": lambda d: len(d) == 128,
        "labels_article.parquet": lambda d: d["customer_id"].nunique() == 15246,
    }
    for f, ok in checks.items():
        p = pd_ / f
        if p.exists():
            assert ok(pd.read_parquet(p, engine="pyarrow"))
    # demo exports untouched
    demo = pd_.parent / "demo"
    if demo.exists():
        import json
        assert len(json.loads((demo / "customers.json").read_text())) >= 30
