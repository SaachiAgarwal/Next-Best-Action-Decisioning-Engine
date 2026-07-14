"""Tests for the Phase 3b diversity + constraint re-ranker."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.rerank.reranker import ReRanker


def _ranker(product_types, pop_scores):
    """A ReRanker over a synthetic catalog of len(product_types) articles."""
    n = len(product_types)
    article_ids = np.array([f"{i:07d}" for i in range(n)])
    return ReRanker(article_ids, np.array(product_types, dtype=object), np.array(pop_scores))


def _sim_from_vectors(vecs, order):
    """Content cosine sim among candidates given unit-norm feature vectors."""
    V = np.array([vecs[i] for i in order], dtype=float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V @ V.T


# --------------------------------------------------------------------------
# The key regression guard: LAMBDA=1, POP_PENALTY=0 == stage-1 top-k
# --------------------------------------------------------------------------
def test_lambda1_pop0_reproduces_stage1_topk():
    rr = _ranker(["a", "b", "c", "d", "e"], [0.9, 0.1, 0.5, 0.2, 0.3])
    cand = np.array([0, 1, 2, 3, 4])              # stage-1 order
    rel = np.array([5.0, 4.0, 3.0, 2.0, 1.0])     # descending relevance
    sim = np.eye(5)                                # sim irrelevant at λ=1
    sel, blocks = rr.rerank(cand, rel, sim, k=3, lam=1.0, pop_penalty=0.0)
    assert sel == [0, 1, 2]                        # exactly the stage-1 top-3
    assert blocks == []


# --------------------------------------------------------------------------
# MMR penalizes redundancy
# --------------------------------------------------------------------------
def test_mmr_penalizes_identical_candidate():
    # Candidates 0 and 1 are identical in content; 2 is different.
    vecs = {0: [1, 0], 1: [1, 0], 2: [0, 1]}
    rr = _ranker(["x", "x", "y"], [0.0, 0.0, 0.0])
    cand = np.array([0, 1, 2]); rel = np.array([1.0, 0.99, 0.5])
    sim = _sim_from_vectors(vecs, cand)
    # With diversity weight, after picking 0 the identical 1 is penalized; 2 comes next.
    sel, _ = rr.rerank(cand, rel, sim, k=2, lam=0.5, pop_penalty=0.0)
    assert sel[0] == 0 and sel[1] == 2           # the redundant twin is skipped


# --------------------------------------------------------------------------
# Popularity penalty lowers mean popularity rank of the output
# --------------------------------------------------------------------------
def test_pop_penalty_pushes_toward_tail():
    # article 0 = very popular (pop_score 1) with top relevance; article 1 = tail
    # (pop_score 0) with near-top relevance. A third low-relevance item keeps the
    # min-max scale from collapsing. The penalty should flip the top pick to tail.
    rr = _ranker(["a", "b", "c"], [1.0, 0.0, 0.0])
    cand = np.array([0, 1, 2]); rel = np.array([1.0, 0.9, 0.0]); sim = np.eye(3)
    top_no_pen = rr.rerank(cand, rel, sim, k=1, lam=1.0, pop_penalty=0.0)[0]
    top_pen = rr.rerank(cand, rel, sim, k=1, lam=1.0, pop_penalty=0.3)[0]
    assert top_no_pen == [0]                      # popular item without penalty
    assert top_pen == [1]                         # tail item once penalized


# --------------------------------------------------------------------------
# Category cap
# --------------------------------------------------------------------------
def test_category_cap_limits_one_type():
    rr = _ranker(["trousers"] * 4 + ["dress"], [0.0] * 5)
    cand = np.array([0, 1, 2, 3, 4]); rel = np.array([5, 4, 3, 2, 1.0])
    sim = np.eye(5)
    sel, blocks = rr.rerank(cand, rel, sim, k=4, lam=1.0, pop_penalty=0.0, cap=2)
    types = ["trousers", "trousers", "trousers", "trousers", "dress"]
    chosen_types = [types[np.where(cand == a)[0][0]] for a in sel]
    assert chosen_types.count("trousers") <= 2    # cap respected
    assert any(r == "category_cap" for _, r in blocks)


# --------------------------------------------------------------------------
# Fatigue + block logging
# --------------------------------------------------------------------------
def test_fatigue_blocks_recent_type_and_logs():
    rr = _ranker(["trousers", "dress"], [0.0, 0.0])
    cand = np.array([0, 1]); rel = np.array([5.0, 1.0]); sim = np.eye(2)
    # Customer bought 'trousers' recently -> trousers candidate is blocked.
    sel, blocks = rr.rerank(cand, rel, sim, k=2, lam=1.0, pop_penalty=0.0,
                            fatigue_types={"trousers"})
    assert 0 not in sel                            # trousers filtered
    assert (0, "fatigue") in blocks                # logged with a reason
    # With no fatigue window, trousers is selected.
    sel2, _ = rr.rerank(cand, rel, sim, k=2, lam=1.0, pop_penalty=0.0, fatigue_types=set())
    assert 0 in sel2


def test_out_of_stock_filtered_and_logged():
    rr = _ranker(["a", "b"], [0.0, 0.0])
    cand = np.array([0, 1]); rel = np.array([5.0, 1.0]); sim = np.eye(2)
    sel, blocks = rr.rerank(cand, rel, sim, k=2, lam=1.0, pop_penalty=0.0, oos={0})
    assert 0 not in sel
    assert (0, "out_of_stock") in blocks


# --------------------------------------------------------------------------
# Cannot exceed the retrieved set
# --------------------------------------------------------------------------
def test_selects_only_from_retrieved():
    rr = _ranker(["a", "b", "c"], [0.0, 0.0, 0.0])
    cand = np.array([2, 0, 1]); rel = np.array([3.0, 2.0, 1.0]); sim = np.eye(3)
    sel, _ = rr.rerank(cand, rel, sim, k=3, lam=0.7, pop_penalty=0.2)
    assert set(sel).issubset({0, 1, 2})           # only retrieved candidates


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
