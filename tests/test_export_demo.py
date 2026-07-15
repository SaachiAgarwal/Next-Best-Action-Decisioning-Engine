"""Tests for the Phase 5a demo JSON export.

These run against the COMMITTED files in ``data/demo/`` (produced by
``python -m src.run_export_demo``). They validate structure, the 5-variant
contract, the pre-cutoff/label-window leakage guard, and that the export stays
consistent with the source parquet. No models are built here.
"""

import json
import subprocess

import pandas as pd
import pytest

from src import config

DEMO = config.PROCESSED_DIR.parent / "demo"
pytestmark = pytest.mark.skipif(
    not (DEMO / "customers.json").exists(),
    reason="demo export not generated (run: python -m src.run_export_demo)")


def _load(name):
    return json.loads((DEMO / name).read_text())


@pytest.fixture(scope="module")
def demo():
    return {n: _load(f"{n}.json") for n in
            ["customers", "recommendations", "explanations", "frontier", "diagnostics"]}


# --------------------------------------------------------------------------
# Task 7 — every demo customer appears in all three per-customer files
# --------------------------------------------------------------------------
def test_every_customer_in_all_files(demo):
    handles = [c["id"] for c in demo["customers"]]
    assert len(handles) == len(set(handles)) and len(handles) >= 30
    assert set(handles) == set(demo["recommendations"].keys())
    assert set(handles) == set(demo["explanations"].keys())


# --------------------------------------------------------------------------
# Task 7 — 5 model variants per customer, 12 items each
# --------------------------------------------------------------------------
def test_five_variants_twelve_items(demo):
    variants = {"hybrid", "mf", "content", "rerank_div", "rerank_cov"}
    for h, rec in demo["recommendations"].items():
        assert set(rec.keys()) == variants, h
        for v, items in rec.items():
            assert len(items) == 12, f"{h}/{v} has {len(items)}"
            for it in items:
                assert set(it) >= {"aid", "name", "type", "colour", "dept",
                                   "sc", "rank", "hit"}
                assert isinstance(it["hit"], bool)


# --------------------------------------------------------------------------
# Task 7 — leakage guard: history is pre-cutoff, ground_truth is label window
# --------------------------------------------------------------------------
def test_no_leakage_history_precutoff_truth_labelwindow(demo):
    cids = [c["cid"] for c in demo["customers"]]
    cutoff = pd.Timestamp(config.CUTOFF_DATE)
    el = pd.read_parquet(config.PROCESSED_DIR / "event_log.parquet",
                         columns=["customer_id", "t_dat", "article_id"], engine="pyarrow")
    el["customer_id"] = el["customer_id"].astype("string")
    el["article_id"] = el["article_id"].astype("string")
    el["t_dat"] = pd.to_datetime(el["t_dat"])
    el = el[el["customer_id"].isin(set(cids))]
    pre = {c: set(g[g["t_dat"] < cutoff]["article_id"])
           for c, g in el.groupby("customer_id", sort=False)}
    la = pd.read_parquet(config.PROCESSED_DIR / "labels_article.parquet", engine="pyarrow")
    la["customer_id"] = la["customer_id"].astype("string")
    la["article_id"] = la["article_id"].astype("string")
    labels = {c: set(g) for c, g in la.groupby("customer_id", sort=False)["article_id"]}

    for c in demo["customers"]:
        cid = c["cid"]
        hist_aids = {h["aid"] for h in c["history"]}
        assert hist_aids <= pre.get(cid, set()), f"{c['id']} history leaks post-cutoff"
        gt_aids = {g["aid"] for g in c["ground_truth"]["articles"]}
        assert gt_aids <= labels.get(cid, set()), f"{c['id']} ground_truth not from labels"
        # cold-start honesty: zero history => (possibly) empty ground truth is fine
        if c["profile"]["cold"]:
            assert len(c["history"]) == 0


# --------------------------------------------------------------------------
# Task 7 — every file parses and matches the documented schema
# --------------------------------------------------------------------------
def test_customers_schema(demo):
    for c in demo["customers"]:
        assert set(c) >= {"id", "cid", "seg", "profile", "history", "ground_truth"}
        assert set(c["profile"]) >= {"age_band", "club", "cold", "freq",
                                     "recency_days", "distinct_types", "dominant_types"}
        assert set(c["ground_truth"]) == {"n", "articles"}


def test_explanations_schema(demo):
    for h, e in demo["explanations"].items():
        assert set(e) >= {"top_article", "why", "fidelity", "bundle"}
        assert e["fidelity"] in {"passed", "blocked"}
        assert "article_facts" in e["bundle"] and "recommendation_context" in e["bundle"]
        # leakage: the slimmed bundle must not carry the raw purchased-id list
        assert "purchased_article_ids" not in e["bundle"]["customer_history"]
    # at least one adversarial showcase present, and it is a genuine block
    adv = [e["adversarial"] for e in demo["explanations"].values() if "adversarial" in e]
    assert adv, "expected >=1 adversarial example"
    assert all(a["blocked"] and a["gate"] == "rule" for a in adv)


def test_diagnostics_schema_opposite_ranking(demo):
    d = {r["model"]: r for r in demo["diagnostics"]}
    assert {"triple hybrid", "MF", "content", "neighborhood CF"} <= set(d)
    # the headline story: MF is far more accurate than CF but covers far less
    assert d["MF"]["hit12"] > d["neighborhood CF"]["hit12"]
    assert d["MF"]["cov12"] < d["neighborhood CF"]["cov12"]


# --------------------------------------------------------------------------
# Task 7 — frontier.json points match rerank_frontier.parquet
# --------------------------------------------------------------------------
def test_frontier_matches_parquet(demo):
    fp = pd.read_parquet(config.PROCESSED_DIR / "rerank_frontier.parquet", engine="pyarrow")
    assert len(demo["frontier"]) == len(fp)
    by = {(round(r["lambda"], 2), round(r["pop"], 2)): r for r in demo["frontier"]}
    for _, row in fp.iterrows():
        j = by[(round(row["lambda"], 2), round(row["pop_penalty"], 2))]
        assert abs(j["recall12"] - row["recall@12"]) < 1e-4
        assert abs(j["cov12"] - row["coverage@12"]) < 1e-1


# --------------------------------------------------------------------------
# Task 7 — data/demo is committed (not gitignored); prior artifacts intact
# --------------------------------------------------------------------------
def test_demo_dir_not_gitignored():
    r = subprocess.run(["git", "check-ignore", "data/demo/customers.json"],
                       cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode != 0, "data/demo/customers.json is gitignored but must be committed"


def test_prior_artifacts_intact():
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
    lap = config.PROCESSED_DIR / "labels_article.parquet"
    if lap.exists():
        assert pd.read_parquet(lap, columns=["customer_id"], engine="pyarrow")["customer_id"].nunique() == 15246
