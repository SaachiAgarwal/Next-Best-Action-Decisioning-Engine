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


# ==========================================================================
# Phase 5c — contact-timing layer (contact_timing.json). Additive; the tests
# above are unchanged. This block also guards the 6 prior files byte-identical.
# ==========================================================================
import hashlib  # noqa: E402

_CT = DEMO / "contact_timing.json"
_ctmark = pytest.mark.skipif(not _CT.exists(),
                             reason="run: python -m src.run_export_contact_timing")
_VALID_BANDS = {b[0] for b in config.CONTACT_BANDS} | {"no history"}
# SHA256 of the six Phase 5a/5b demo files, captured before Phase 5c ran.
_PRIOR_HASHES = {
    "customers.json": "07906b6913aeb8c72222169701355fb1d757c1725b63c42509f93d9b918721c5",
    "recommendations.json": "120c365ec499fbdd35eb009e8c8a26f273b61e3021369ef899b86e8f308d5248",
    "explanations.json": "6e7db190ac8eb4548e10e596f4d9b32de78c82a10fa0c9541e8a9f329cae8b8b",
    "frontier.json": "a2b6d9597a5446b44cdaf313199bff93507d29ce18ae89f4f51cefa5b2c70013",
    "diagnostics.json": "a8927029a133f0dfa1014db9e8393da3ba56f2a3d1219a6ead3f0c2661a0ef59",
    "producttype.json": "7f184ef70c70ca0523c7fd8f3aa80ea1c37320cb8cf3a7203d5e05deecabefae",
}


@pytest.fixture(scope="module")
def ct():
    return json.loads(_CT.read_text())


@_ctmark
def test_ct_all_38_customers_present(ct):
    handles = {c["id"] for c in _load("customers.json")}
    assert set(ct["customers"].keys()) == handles
    assert len(ct["customers"]) == 38


@_ctmark
def test_ct_every_band_valid(ct):
    for h, r in ct["customers"].items():
        assert r["band"] in _VALID_BANDS, f"{h}: {r['band']}"


@_ctmark
def test_ct_cold_start_is_no_history(ct):
    cold = {c["id"] for c in _load("customers.json") if c["profile"].get("cold")}
    assert cold, "expected some cold-start demo customers"
    for h in cold:
        r = ct["customers"][h]
        assert r["band"] == "no history"
        assert r["due_ratio"] is None and r["days_since_last"] is None and r["typical_gap"] is None


@_ctmark
def test_ct_band_counts_sum_to_evaluable(ct):
    assert sum(b["customers"] for b in ct["bands"]) == 15246
    assert {b["band"] for b in ct["bands"]} == {b[0] for b in config.CONTACT_BANDS}


@_ctmark
def test_ct_band_hit12_matches_exp7_artifacts(ct):
    # source of truth: the Exp 7 report band table
    report = (config.REPORTS_DIR / "exp7_temporal.md").read_text()
    import re
    for b in ct["bands"]:
        m = re.search(rf"\|\s*{re.escape(b['band'])}\s*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|", report)
        assert m, b["band"]
        assert abs(b["hit12"] - round(float(m.group(1)), 4)) < 1e-9


@_ctmark
def test_ct_schema_and_size(ct):
    assert set(ct) == {"bands", "customers", "finding"}
    assert set(ct["finding"]) == {"headline", "detail", "caveat"}
    for b in ct["bands"]:
        assert set(b) == {"band", "range", "customers", "share", "hit12"}
    for r in ct["customers"].values():
        assert set(r) == {"band", "due_ratio", "days_since_last", "typical_gap"}
    assert _CT.stat().st_size < 50 * 1024


@_ctmark
def test_ct_committed_not_gitignored():
    r = subprocess.run(["git", "check-ignore", "data/demo/contact_timing.json"],
                       cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode != 0, "contact_timing.json must be committed, not gitignored"


def test_ct_regression_prior_demo_files_byte_identical():
    """Phase 5c must ONLY add a file — the six prior exports stay byte-for-byte."""
    for name, want in _PRIOR_HASHES.items():
        got = hashlib.sha256((DEMO / name).read_bytes()).hexdigest()
        assert got == want, f"{name} changed! Phase 5c must not modify existing demo files."


# ==========================================================================
# Phase 5d — portfolio segment statistics (segments.json). Additive.
# ==========================================================================
_SEG = DEMO / "segments.json"
_segmark = pytest.mark.skipif(not _SEG.exists(),
                              reason="run: python -m src.run_export_segments")
_VALID_STATUS = {"Above average", "At average", "Below average", "No ground truth"}
# 7 prior demo files, byte-identical before Phase 5d ran.
_PRIOR7 = dict(_PRIOR_HASHES)
_PRIOR7["contact_timing.json"] = "4346ae4eba02d72f57be2d26811bea6394f4c179f6f0b4e7a7abe4be9f4e3928"


@pytest.fixture(scope="module")
def seg():
    return json.loads(_SEG.read_text())


@_segmark
def test_seg_schema(seg):
    assert set(seg) == {"overall", "segments", "cold_start", "biggest_gap", "caveat"}
    assert set(seg["overall"]) == {"customers", "hit12", "recall12", "avg_distinct_types", "coverage12"}
    for s in seg["segments"]:
        assert set(s) == {"segment", "definition", "customers", "share", "hit12", "recall12",
                          "avg_distinct_types", "coverage12", "mean_rec_pop_rank",
                          "mean_true_pop_rank", "status"}
    assert set(seg["cold_start"]) == {"customers", "note"}
    assert set(seg["biggest_gap"]) == {"segment", "statement"}
    assert _SEG.stat().st_size < 20 * 1024


@_segmark
def test_seg_frequency_terciles_partition_evaluable(seg):
    # frequency terciles are mutually exclusive and partition the 15,246 evaluable
    freq = [s for s in seg["segments"] if "frequency" in s["segment"]]
    assert len(freq) == 3
    assert sum(s["customers"] for s in freq) == 15246
    # divergent is an OVERLAPPING view -> not part of the partition sum
    div = [s for s in seg["segments"] if s["segment"] == "divergent taste"][0]
    assert 0 < div["customers"] < 15246


@_segmark
def test_seg_hit12_nonnull_except_cold_and_never_zero(seg):
    for s in seg["segments"]:
        assert s["hit12"] is not None, s["segment"]
        assert s["hit12"] > 0, f"{s['segment']} hit12=0 — investigate"
    assert seg["cold_start"]["customers"] > 0
    assert "no ground truth" in seg["cold_start"]["note"].lower()


@_segmark
def test_seg_overall_matches_production(seg):
    assert abs(seg["overall"]["hit12"] - 0.0628) < 0.003    # known production value


@_segmark
def test_seg_status_labels_follow_thresholds(seg):
    oh = seg["overall"]["hit12"]
    for s in seg["segments"]:
        h = s["hit12"]
        if h >= 1.1 * oh:
            want = "Above average"
        elif h <= 0.9 * oh:
            want = "Below average"
        else:
            want = "At average"
        assert s["status"] == want, f"{s['segment']}: {s['status']} != {want}"
        assert s["status"] in _VALID_STATUS


@_segmark
def test_seg_committed_not_gitignored():
    r = subprocess.run(["git", "check-ignore", "data/demo/segments.json"],
                       cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode != 0


def test_seg_regression_seven_prior_files_byte_identical():
    """Phase 5d must ONLY add a file — the seven prior exports stay byte-for-byte."""
    for name, want in _PRIOR7.items():
        got = hashlib.sha256((DEMO / name).read_bytes()).hexdigest()
        assert got == want, f"{name} changed! Phase 5d must not modify existing demo files."
