"""Tests for Experiment 5 — article-level matrix factorization + triple hybrid."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.item_cf_article import ArticleItemCF
from src.models.content_based_exp4 import ContentModel
from src.models.hybrid_content_cf_exp4 import ContentCFHybrid
from src.models.mf_exp5 import MFModel, TripleHybrid

CUTOFF = pd.Timestamp(config.CUTOFF_DATE)

_TYPES = {"0000000": "trousers", "0000001": "trousers", "0000002": "dress",
          "0000003": "dress", "0000004": "top", "0000005": "top"}
_DESC = {"trousers": "black cotton trousers", "dress": "floral summer dress",
         "top": "white cotton top"}


def _articles():
    rows = [{"article_id": a, "product_type_name": t, "product_group_name": "garment",
             "colour_group_name": "black" if t == "trousers" else "white",
             "department_name": t + " d", "graphical_appearance_name": "solid",
             "detail_desc": _DESC[t]} for a, t in _TYPES.items()]
    df = pd.DataFrame(rows); df["article_id"] = df["article_id"].astype("string")
    return df


def _events(baskets):
    day = CUTOFF - pd.Timedelta(days=1)
    rows = [(c, a) for c, acts in baskets.items() for a in acts]
    return pd.DataFrame({
        "customer_id": pd.array([c for c, _ in rows], dtype="string"),
        "article_id": pd.array([a for _, a in rows], dtype="string"),
        "t_dat": [day] * len(rows),
    })


_BASKETS = {"c1": ["0000000", "0000002"], "c2": ["0000000", "0000002"],
            "c3": ["0000001", "0000003"], "c4": ["0000004", "0000001"],
            "c5": ["0000005", "0000002"], "c6": ["0000003", "0000004"]}


@pytest.fixture
def mf():
    fe = _events(_BASKETS)
    cf = ArticleItemCF().fit(fe, verbose=False)
    return MFModel(factors=8).fit(fe, article_order=cf.article_ids)


# --------------------------------------------------------------------------
# Interaction matrix / embeddings
# --------------------------------------------------------------------------
def test_embeddings_have_expected_shape(mf):
    n_cust = len(mf.customer_index)
    n_art = len(mf.article_ids)
    assert mf.U.shape == (n_cust, 8)
    assert mf.V.shape == (n_art, 8)
    assert mf.sparsity > 0


def test_fit_rejects_post_cutoff_events():
    fe = _events({"c1": ["0000000"]})
    fe["t_dat"] = CUTOFF  # at cutoff -> not allowed
    with pytest.raises(AssertionError):
        MFModel(factors=8).fit(fe)


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------
def test_recommend_k_excludes_bought_and_cold_start(mf):
    recs = mf.recommend("c1", k=2, include_repeats=False)  # c1 bought 0,2
    assert len(recs) == 2
    assert set(recs).isdisjoint({"0000000", "0000002"})
    cold = mf.recommend("ghost", k=3)
    assert cold == mf.popularity_model.recommend(k=3)


def test_recommend_repeats_returns_k(mf):
    recs = mf.recommend("c1", k=4, include_repeats=True)
    assert len(recs) == 4
    assert all(isinstance(a, str) for a in recs)  # string ids, leading zeros kept


# --------------------------------------------------------------------------
# Triple hybrid degenerates to Exp 4 two-signal when w3=0
# --------------------------------------------------------------------------
def test_triple_w3_zero_reproduces_exp4_hybrid():
    fe = _events(_BASKETS)
    cf = ArticleItemCF().fit(fe, verbose=False)
    content = ContentModel().fit(_articles(), fe, article_order=cf.article_ids)
    mf = MFModel(factors=8).fit(fe, article_order=cf.article_ids)
    triple = TripleHybrid(content, cf, mf)
    cch = ContentCFHybrid(content, cf)
    for c in ["c1", "c3", "c5"]:
        assert (triple.recommend(c, k=3, w1=1.0, w2=1.0, w3=0.0, include_repeats=False)
                == cch.recommend(c, k=3, alpha=1.0, beta=1.0, include_repeats=False))


# --------------------------------------------------------------------------
# Prior experiments intact
# --------------------------------------------------------------------------
def test_prior_experiment_artifacts_intact():
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
    lap = config.PROCESSED_DIR / "labels_article.parquet"
    if lap.exists():
        la = pd.read_parquet(lap, columns=["customer_id"], engine="pyarrow")
        assert la["customer_id"].nunique() == 15246
