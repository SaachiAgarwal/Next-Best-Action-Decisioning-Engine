"""Tests for the Phase 3a beyond-accuracy diagnostics."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src import config
from src.eval import diagnostics as dg


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------
def test_coverage_on_synthetic():
    recs = {"c1": ["a", "b", "c"], "c2": ["a", "b", "d"]}
    top10 = {"a"}
    out = dg.coverage(recs, n_catalog=10, k=3, top10_set=top10)
    assert out["coverage_count"] == 4                 # {a,b,c,d}
    assert out["coverage_pct"] == pytest.approx(40.0)
    assert out["aggregate_diversity"] == pytest.approx(4 / (2 * 3))
    # 6 recs total, 2 are 'a' (top-10%), 4 are long tail -> 66.67%
    assert out["long_tail_share_pct"] == pytest.approx(100 * 4 / 6)


# --------------------------------------------------------------------------
# Popularity rank / Gini
# --------------------------------------------------------------------------
def test_gini_extremes():
    assert dg.gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)   # perfectly even
    assert dg.gini([0, 0, 0, 10]) > 0.7                            # concentrated


def test_popularity_bias_on_synthetic():
    # ranked most->least: a(1), b(2), c(3), d(4)
    pop_rank = dg.popularity_ranks(["a", "b", "c", "d"])
    recs = {"c1": ["a", "a", "d"]}   # two head, one tail
    out = dg.popularity_bias(recs, pop_rank, n_ranked=4, top1_cut=1, top10_cut=2)
    assert out["mean_pop_rank"] == pytest.approx((1 + 1 + 4) / 3)
    assert out["pct_top1"] == pytest.approx(100 * 2 / 3)   # two 'a' at rank 1
    assert out["median_pop_rank"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Intra-list diversity
# --------------------------------------------------------------------------
def _tiny_content():
    # a,b identical; c orthogonal. Rows L2-normalized already.
    M = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    idx = {"a": 0, "b": 1, "c": 2}
    ptype = {"a": "x", "b": "x", "c": "y"}
    return M, idx, ptype


def test_intra_list_diversity_identical_vs_varied():
    M, idx, ptype = _tiny_content()
    # All-identical list -> dissimilarity 0, one distinct type.
    d0 = dg.intra_list_diversity({"c": ["a", "b"]}, M, idx, ptype, k=2)
    assert d0["intra_list_dissimilarity"] == pytest.approx(0.0, abs=1e-9)
    assert d0["avg_distinct_types"] == pytest.approx(1.0)
    # Varied list -> higher dissimilarity, two distinct types.
    d1 = dg.intra_list_diversity({"c": ["a", "c"]}, M, idx, ptype, k=2)
    assert d1["intra_list_dissimilarity"] > d0["intra_list_dissimilarity"]
    assert d1["avg_distinct_types"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Cold-start article scoring asymmetry (the central finding)
# --------------------------------------------------------------------------
def test_cold_article_scoring_asymmetry():
    """MF cannot score a zero-interaction article (no embedding); content can
    (attributes exist). This is the empirical case for content features."""
    from src.models.item_cf_article import ArticleItemCF
    from src.models.content_based_exp4 import ContentModel, CATEGORICAL_ATTRS
    from src.models.mf_exp5 import MFModel
    from sklearn.preprocessing import normalize

    day = pd.Timestamp(config.CUTOFF_DATE) - pd.Timedelta(days=1)
    seen = ["0000000", "0000001", "0000002"]
    fe = pd.DataFrame({
        "customer_id": pd.array(["u1", "u1", "u2", "u2", "u3"], dtype="string"),
        "article_id": pd.array([seen[0], seen[1], seen[0], seen[2], seen[1]], dtype="string"),
        "t_dat": [day] * 5,
    })
    arts = pd.DataFrame({
        "article_id": pd.array(seen + ["9999999"], dtype="string"),  # last = cold, unseen
        "product_type_name": ["trousers", "trousers", "dress", "hat"],
        "product_group_name": "g", "colour_group_name": "black",
        "department_name": "d", "graphical_appearance_name": "solid",
        "detail_desc": ["black cotton trousers", "blue denim trousers",
                        "floral summer dress", "warm winter hat"],
    })
    cf = ArticleItemCF().fit(fe, verbose=False)
    mf = MFModel(factors=4).fit(fe, article_order=cf.article_ids)
    content = ContentModel().fit(arts, fe, article_order=cf.article_ids)

    cold = "9999999"
    # MF: the cold article is not in its item space at all -> cannot score.
    assert cold not in mf.article_index
    # Content: its attributes can be transformed into a feature vector -> scorable.
    sub = arts[arts["article_id"] == cold]
    cat = content.encoder.transform(sub[CATEGORICAL_ATTRS].astype("string").fillna("unknown").to_numpy())
    txt = content.vectorizer.transform(sub["detail_desc"].astype("string").fillna("unknown").tolist())
    vec = normalize(sp.hstack([cat, txt]).tocsr(), norm="l2", axis=1)
    assert vec.shape[0] == 1 and vec.nnz > 0     # content produced a real vector


# --------------------------------------------------------------------------
# Segment breakdown covers every customer exactly once
# --------------------------------------------------------------------------
def test_segment_breakdown_partitions_customers():
    label_sets = {f"c{i}": {"a"} for i in range(10)}
    segment_of = {f"c{i}": ("even" if i % 2 == 0 else "odd") for i in range(10)}
    recs = {f"c{i}": ["a"] for i in range(10)}
    out = dg.hit_by_segment(recs, label_sets, segment_of, k=1)
    total = sum(v["n"] for kk, v in out.items() if kk != "_spread")
    assert total == 10                      # no double counting, no drops
    assert out["even"]["hit"] == 1.0 and out["odd"]["hit"] == 1.0
    assert out["_spread"] == pytest.approx(0.0)


def test_quartile_labels_balanced():
    labs = dg.quartile_labels(list(range(100)))
    counts = pd.Series(labs).value_counts()
    assert set(counts.index) == {"Q1", "Q2", "Q3", "Q4"}
    assert counts.min() >= 24 and counts.max() <= 26   # ~balanced


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
