"""Tests for Experiment 4 — content-based recommender and content/CF hybrid.

Synthetic articles carry leading-zero ids and clean per-type attributes so the
content-similarity and blend behaviors are checkable exactly. Data-backed tests
guard that Experiments A / B / 3 artifacts are intact.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import config
from src.models.content_based_exp4 import ContentModel
from src.models.item_cf_article import ArticleItemCF
from src.models.hybrid_content_cf_exp4 import ContentCFHybrid

CUTOFF = pd.Timestamp(config.CUTOFF_DATE)

# 6 articles, 2 per product type; same-type articles share all attributes.
_TYPES = {
    "0000000": "trousers", "0000001": "trousers",
    "0000002": "dress", "0000003": "dress",
    "0000004": "top", "0000005": "top",
}
_DESC = {"trousers": "black cotton trousers slim leg",
         "dress": "floral summer dress midi",
         "top": "white cotton short sleeve top"}


def _articles():
    rows = []
    for aid, t in _TYPES.items():
        rows.append({
            "article_id": aid,
            "product_type_name": t,
            "product_group_name": "garment",
            "colour_group_name": "black" if t == "trousers" else "white",
            "department_name": t + " dept",
            "graphical_appearance_name": "solid",
            "detail_desc": _DESC[t],
        })
    df = pd.DataFrame(rows)
    df["article_id"] = df["article_id"].astype("string")
    return df


def _events(baskets):
    day = CUTOFF - pd.Timedelta(days=1)
    rows = [(c, a) for c, acts in baskets.items() for a in acts]
    return pd.DataFrame({
        "customer_id": pd.array([c for c, _ in rows], dtype="string"),
        "article_id": pd.array([a for _, a in rows], dtype="string"),
        "t_dat": [day] * len(rows),
    })


_BASKETS = {
    "c1": ["0000000", "0000002"], "c2": ["0000000", "0000002"],
    "c3": ["0000001", "0000003"], "c4": ["0000004"], "c5": ["0000005", "0000001"],
}


@pytest.fixture
def content():
    fe = _events(_BASKETS)
    cf = ArticleItemCF().fit(fe, verbose=False)
    return ContentModel().fit(_articles(), fe, article_order=cf.article_ids)


@pytest.fixture
def models():
    fe = _events(_BASKETS)
    cf = ArticleItemCF().fit(fe, verbose=False)
    cm = ContentModel().fit(_articles(), fe, article_order=cf.article_ids)
    return cm, cf, ContentCFHybrid(cm, cf)


# --------------------------------------------------------------------------
# Item profiles & similarity
# --------------------------------------------------------------------------
def test_item_matrix_sparse_one_row_per_article(content):
    assert sp.issparse(content.item_matrix)
    assert content.item_matrix.shape[0] == len(content.article_ids)


def test_top_content_neighbor_shares_product_type(content):
    """An article's most content-similar neighbor shares its product_type."""
    names = dict(_TYPES)
    for aid in _TYPES:
        top = content.similar_articles(aid, 1)[0][0]
        assert names[top] == names[aid], f"{aid} top neighbor {top} differs in type"


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------
def test_content_recommend_k_excludes_bought_cold_start(content):
    recs = content.recommend("c1", k=2, include_repeats=False)  # bought 0,2
    assert len(recs) == 2
    assert set(recs).isdisjoint({"0000000", "0000002"})
    cold = content.recommend("ghost", k=3)
    assert cold == content.popularity_model.recommend(k=3)


# --------------------------------------------------------------------------
# Hybrid degenerates to each component
# --------------------------------------------------------------------------
def test_hybrid_beta0_is_pure_content(models):
    cm, cf, hyb = models
    for c in ["c1", "c3", "c5"]:
        assert (hyb.recommend(c, k=3, alpha=1, beta=0, include_repeats=False)
                == cm.recommend(c, k=3, include_repeats=False))


def test_hybrid_alpha0_is_pure_cf(models):
    cm, cf, hyb = models
    for c in ["c1", "c3", "c5"]:
        assert (hyb.recommend(c, k=3, alpha=0, beta=1, include_repeats=False)
                == cf.recommend(c, k=3, include_repeats=False))


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
def test_content_fit_rejects_post_reference_events():
    bad = _events({"c1": ["0000000"]})
    bad["t_dat"] = CUTOFF  # exactly at reference -> not allowed
    with pytest.raises(AssertionError):
        ContentModel().fit(_articles(), bad, article_order=np.array(list(_TYPES), dtype=object))


# --------------------------------------------------------------------------
# Prior experiments intact
# --------------------------------------------------------------------------
def test_experiment_artifacts_intact():
    """Exp A (128 product-type actions) and Exp B (article labels) untouched."""
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
    lap = config.PROCESSED_DIR / "labels_article.parquet"
    if lap.exists():
        la = pd.read_parquet(lap, columns=["customer_id"], engine="pyarrow")
        assert la["customer_id"].nunique() == 15246
