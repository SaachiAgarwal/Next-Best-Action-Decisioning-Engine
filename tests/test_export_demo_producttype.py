"""Tests for the Phase 5b product-type demo layer (data/demo/producttype.json)."""

import json
import subprocess

import pandas as pd
import pytest

from src import config

DEMO = config.PROCESSED_DIR.parent / "demo"
PT = DEMO / "producttype.json"
pytestmark = pytest.mark.skipif(
    not PT.exists(),
    reason="product-type export not generated (run: python -m src.run_export_demo_producttype)")


@pytest.fixture(scope="module")
def pt():
    return json.loads(PT.read_text())


@pytest.fixture(scope="module")
def handles():
    return [c["id"] for c in json.loads((DEMO / "customers.json").read_text())]


def test_every_customer_both_models_twelve_items(pt, handles):
    assert set(pt["customers"].keys()) == set(handles)
    for h, rec in pt["customers"].items():
        assert set(rec) == {"exp3_hybrid", "popularity", "ground_truth"}
        for model in ("exp3_hybrid", "popularity"):
            assert len(rec[model]) == 12, f"{h}/{model}"
            for it in rec[model]:
                assert set(it) == {"action_id", "type", "rank", "hit"}
                assert isinstance(it["hit"], bool)


def test_hit_flags_match_producttype_labels(pt):
    """Spot-check hit flags + ground truth against labels.parquet (the source)."""
    cohort = {c["id"]: c["cid"] for c in json.loads((DEMO / "customers.json").read_text())}
    lab = pd.read_parquet(config.PROCESSED_DIR / "labels.parquet", engine="pyarrow")
    lab["customer_id"] = lab["customer_id"].astype("string")
    label_sets = {c: set(int(x) for x in g) for c, g in
                  lab.groupby("customer_id", sort=False)["action_id"]}
    actions = pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
    aname = dict(zip(actions["action_id"].astype(int), actions["product_type_name"]))

    for h in list(pt["customers"])[:8]:      # spot-check a subset
        cid = cohort[h]
        ls = label_sets.get(cid, set())
        rec = pt["customers"][h]
        for model in ("exp3_hybrid", "popularity"):
            for it in rec[model]:
                assert it["hit"] == (it["action_id"] in ls), (h, model, it)
                assert it["type"] == aname.get(it["action_id"])
        # ground_truth is exactly the label-window product types (leakage guard)
        assert rec["ground_truth"]["n"] == len(ls)
        assert set(rec["ground_truth"]["types"]) == {aname[a] for a in ls if a in aname}


def test_comparison_object_shape(pt):
    comp = pt["comparison"]
    assert comp["product_type"]["n_actions"] == 128
    assert comp["article"]["n_articles"] > 70000
    # the honest headline: category task barely beats popularity; SKU task doubles it
    assert comp["product_type"]["exp3_hit@12"] > comp["product_type"]["popularity_hit@12"]
    assert comp["article"]["triple_hybrid_hit@12"] > 1.8 * comp["article"]["popularity_hit@12"]
    assert "not comparable" in comp["note"].lower()
    assert "not implemented end-to-end" in comp["note"].lower()


def test_file_small_and_parses():
    assert PT.stat().st_size < 500 * 1024      # comfortably small
    json.loads(PT.read_text())


def test_committed_and_prior_artifacts_intact():
    r = subprocess.run(["git", "check-ignore", "data/demo/producttype.json"],
                       cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode != 0, "producttype.json is gitignored but must be committed"
    assert len(pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")) == 128
    assert pd.read_parquet(config.PROCESSED_DIR / "labels_article.parquet",
                           columns=["customer_id"], engine="pyarrow")["customer_id"].nunique() == 15246
