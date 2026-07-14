"""Tests for Experiment 6 — attributes in the hybrid (6a) + attribute cold-start (6b)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.item_cf_article import ArticleItemCF
from src.models.content_based_exp4 import ContentModel
from src.models.mf_exp5 import MFModel, TripleHybrid
from src.models.hybrid_attrs_exp6 import SegmentLift, FourSignalHybrid, ATTRS
from src.models.coldstart_exp6 import ColdStartModel

CUTOFF = pd.Timestamp(config.CUTOFF_DATE)


def _articles():
    ids = ["0000000", "0000001", "0000002", "0000003"]
    df = pd.DataFrame({
        "article_id": pd.array(ids, dtype="string"),
        "product_type_name": ["trousers", "dress", "hat", "socks"],
        "product_group_name": "g", "colour_group_name": "black",
        "department_name": "d", "graphical_appearance_name": "solid",
        "detail_desc": ["black cotton trousers", "floral summer dress",
                        "warm winter hat", "wool socks"],
    })
    return df, ids


def _events():
    day = CUTOFF - pd.Timedelta(days=1)
    # Young customers (u1,u2) buy the 'hat' (0000002); older (u3,u4) buy 'dress'.
    # Everyone buys the popular 'trousers' (0000000).
    rows = [
        ("u1", "0000000"), ("u1", "0000002"), ("u2", "0000000"), ("u2", "0000002"),
        ("u3", "0000000"), ("u3", "0000001"), ("u4", "0000000"), ("u4", "0000001"),
        ("u5", "0000000"), ("u5", "0000003"),
    ]
    return pd.DataFrame({
        "customer_id": pd.array([c for c, _ in rows], dtype="string"),
        "article_id": pd.array([a for _, a in rows], dtype="string"),
        "t_dat": [day] * len(rows),
    })


def _context():
    return pd.DataFrame({
        "customer_id": pd.array(["u1", "u2", "u3", "u4", "u5", "u6"], dtype="string"),
        "age_band": pd.array(["<=25", "<=25", "56+", "56+", "26-35", "unknown"], dtype="string"),
        "club_member_status": pd.array(["ACTIVE"] * 5 + ["unknown"], dtype="string"),
        "fashion_news_frequency": pd.array(["NONE"] * 5 + ["unknown"], dtype="string"),
        "frequency": [2, 2, 2, 2, 2, 0], "recency_days": [1.0] * 5 + [np.nan],
    })


@pytest.fixture
def fitted():
    fe = _events(); arts, ids = _articles(); ctx = _context()
    cf = ArticleItemCF().fit(fe, verbose=False)
    content = ContentModel().fit(arts, fe, article_order=cf.article_ids)
    mf = MFModel(factors=4).fit(fe, article_order=cf.article_ids)
    sl = SegmentLift().fit(fe, ctx, cf.article_ids)
    return fe, arts, ctx, cf, content, mf, sl


# --------------------------------------------------------------------------
# Segment lift: leakage + lift-not-count
# --------------------------------------------------------------------------
def test_segment_lift_rejects_post_cutoff(fitted):
    fe, arts, ctx, cf, *_ = fitted
    bad = fe.copy(); bad["t_dat"] = CUTOFF
    with pytest.raises(AssertionError):
        SegmentLift().fit(bad, ctx, cf.article_ids)


def test_lift_not_raw_count(fitted):
    """A globally-popular article NOT disproportionately bought by a segment gets
    a LOW lift for that segment (lift isolates distinctiveness, not popularity)."""
    _, _, _, _, _, _, sl = fitted
    ai = sl.article_index
    young = sl.seg_index["age_band"]["<=25"]
    # 'trousers' (0000000) is bought by everyone -> not distinctive -> lift ~1.
    lift_trousers = sl.lift["age_band"][young][ai["0000000"]]
    # 'hat' (0000002) is bought only by the young -> distinctive -> lift > trousers.
    lift_hat = sl.lift["age_band"][young][ai["0000002"]]
    assert lift_hat > lift_trousers
    assert lift_trousers < 1.5      # the universal item is not distinctive


# --------------------------------------------------------------------------
# Four-signal hybrid: w4=0 reproduces the triple hybrid (key regression guard)
# --------------------------------------------------------------------------
def test_four_signal_w4_zero_equals_triple(fitted):
    fe, arts, ctx, cf, content, mf, sl = fitted
    four = FourSignalHybrid(content, cf, mf, sl)
    triple = TripleHybrid(content, cf, mf)
    for c in ["u1", "u3", "u5"]:
        assert (four.recommend(c, k=3, w1=1.0, w2=0.5, w3=1.0, w4=0.0, include_repeats=True)
                == triple.recommend(c, k=3, w1=1.0, w2=0.5, w3=1.0, include_repeats=True))


# --------------------------------------------------------------------------
# Cold-start model: unknown attributes -> popularity fallback
# --------------------------------------------------------------------------
def test_coldstart_unknown_attrs_falls_back_to_popularity(fitted):
    _, _, _, _, _, _, sl = fitted
    from src.models.popularity_article import ArticlePopularityModel
    pop = ArticlePopularityModel().fit(_events())
    cold = ColdStartModel(sl, pop)
    # u6 has all-unknown attributes -> not "known" -> pure popularity.
    assert not cold.has_known_attributes("u6")
    assert cold.recommend("u6", k=3) == pop.recommend(k=3)
    # u1 has known attributes.
    assert cold.has_known_attributes("u1")


def test_coldstart_recommend_returns_k(fitted):
    _, _, _, _, _, _, sl = fitted
    from src.models.popularity_article import ArticlePopularityModel
    pop = ArticlePopularityModel().fit(_events())
    cold = ColdStartModel(sl, pop)
    assert len(cold.recommend("u1", k=3, v1=1.0, v2=1.0)) == 3


# --------------------------------------------------------------------------
# Prior artifacts intact
# --------------------------------------------------------------------------
def test_prior_experiment_artifacts_intact():
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
    lap = config.PROCESSED_DIR / "labels_article.parquet"
    if lap.exists():
        assert pd.read_parquet(lap, columns=["customer_id"], engine="pyarrow")["customer_id"].nunique() == 15246
