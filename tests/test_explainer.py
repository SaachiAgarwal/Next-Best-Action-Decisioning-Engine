"""Tests for the Phase 4 agentic RAG explanation layer.

The LLM is always MOCKED — the suite makes NO live API calls (Definition of Done).
Generators and verifiers are injected as deterministic objects, so every test is
free and reproducible. The rule gate is deterministic by design and needs no mock.
"""

import types

import numpy as np
import pandas as pd
import pytest

from src import config
from src.explain import agent as A
from src.explain.verifier import (RuleGate, SoftVerifier, LLMSoftVerifier,
                                   heuristic_soft_verify, parse_verdicts)
from src.run_explain import explain_and_verify

PRE = pd.Timestamp(config.CUTOFF_DATE) - pd.Timedelta(days=20)   # pre-cutoff
PRE2 = pd.Timestamp(config.CUTOFF_DATE) - pd.Timedelta(days=5)
POST = pd.Timestamp(config.CUTOFF_DATE) + pd.Timedelta(days=6)   # post-cutoff (leak bait)

ART_COLS = ["article_id", "product_type_name", "product_group_name", "colour_group_name",
            "department_name", "graphical_appearance_name", "detail_desc", "prod_name",
            "garment_group_name"]


def _articles():
    rows = [
        ("0000000001", "vest top", "garment upper body", "black", "jersey basic",
         "solid", "jersey top with straps.", "SENTINEL_PRODNAME_X", "jersey basic"),
        ("0000000002", "trousers", "garment lower body", "white", "trouser dept",
         "solid", "cotton trousers.", "trouser a", "trousers"),
        ("0000000003", "hat", "accessories", "beige", "acc dept",
         "solid", None, "hat a", "accessories"),   # null detail_desc
    ]
    return pd.DataFrame(rows, columns=ART_COLS)


def _events():
    # C1 bought the vest top (art 1) twice pre-cutoff, plus one POST-cutoff trousers
    # purchase that must NOT leak into the bundle.
    return pd.DataFrame({
        "customer_id": ["C1", "C1", "C1"],
        "t_dat": [PRE, PRE2, POST],
        "article_id": ["0000000001", "0000000001", "0000000002"],
    })


def _decision_log():
    return pd.DataFrame({
        "customer_id": ["C1"],
        "article_id": ["0000000001"],
        "stage1_relevance": [0.9],
        "mmr_score": [0.7],
        "popularity_rank": [42],
        "final_position": [0],
    })


def _block_log():
    return pd.DataFrame({
        "customer_id": ["C1"],
        "article_id": ["0000000002"],
        "rule_violated": ["fatigue"],
    })


def _store():
    return A.EvidenceStore(_articles(), _events(), _decision_log(), _block_log())


def _gate():
    return RuleGate(_articles(), _events())


class _MockGen:
    """Injected mock LLM generator returning a fixed generation."""
    name = "mock"

    def __init__(self, explanation, claims):
        self._o = {"explanation": explanation, "claims": claims}

    def generate(self, bundle):
        return {"explanation": self._o["explanation"], "claims": list(self._o["claims"])}


# --------------------------------------------------------------------------
# Task 1 — each tool returns only real data from its source (no fabrication)
# --------------------------------------------------------------------------
def test_article_facts_are_real_record():
    s = _store()
    f = s.get_article_facts("0000000001")
    assert f["colour_group_name"] == "black"
    assert f["product_type_name"] == "vest top"
    assert f["department_name"] == "jersey basic"
    # A field that isn't a real article attribute is never invented.
    assert "material" not in f
    assert s.get_article_facts("9999999999") is None  # unknown → None, not fabricated


def test_customer_history_summarises_real_purchases():
    s = _store()
    h = s.get_customer_history("C1")
    assert h["total_purchases"] == 2                        # POST-cutoff excluded
    assert {d["product_type"] for d in h["top_product_types"]} == {"vest top"}
    assert "black" in h["colours_bought"]
    assert s.get_customer_history("NOBODY")["total_purchases"] == 0


def test_recommendation_and_constraint_tools_read_logs():
    s = _store()
    rc = s.get_recommendation_context("C1", "0000000001")
    assert rc["found"] and rc["final_position"] == 0 and rc["popularity_rank"] == 42
    cons = s.get_constraint_decisions("C1")
    assert cons["blocked_counts"]["fatigue"] == 1 and cons["total_blocked"] == 1


# --------------------------------------------------------------------------
# Task 2 — evidence bundle contains only pre-cutoff data (leakage guard)
# --------------------------------------------------------------------------
def test_bundle_is_pre_cutoff_only():
    s = _store()
    b = A.build_bundle(s, "C1", "0000000001")
    ids = b["customer_history"]["purchased_article_ids"]
    assert "0000000001" in ids                 # pre-cutoff purchase present
    assert "0000000002" not in ids             # POST-cutoff purchase absent (no leak)
    assert b["customer_history"]["total_purchases"] == 2
    assert b["recommendation_context"]["shares_product_type_with_history"] is True


# --------------------------------------------------------------------------
# Task 4 — rule gate: the central tests
# --------------------------------------------------------------------------
def test_rule_gate_blocks_wrong_colour_claim():
    """THE central test — a colour the article does not have is HARD-BLOCKED."""
    s, g = _store(), _gate()
    b = A.build_bundle(s, "C1", "0000000001")   # article is black
    claims = [{"text": "This is a red item.", "type": "attribute",
               "field": "colour_group_name", "value": "red"}]
    res = g.check(b, claims)
    assert res["passed"] is False
    v = res["violations"][0]
    assert v["field"] == "colour_group_name"
    assert v["true_value"] == "black" and v["claimed"] == "red"


def test_rule_gate_blocks_fabricated_purchase():
    s, g = _store(), _gate()
    b = A.build_bundle(s, "C1", "0000000001")
    claims = [{"text": "You bought shoes 3 times.", "type": "history",
               "product_type": "shoes", "count": 3, "window_days": 60}]
    res = g.check(b, claims)
    assert res["passed"] is False
    assert res["violations"][0]["reason"] == "history_purchase_not_found"


def test_rule_gate_blocks_overstated_count():
    s, g = _store(), _gate()
    b = A.build_bundle(s, "C1", "0000000001")
    claims = [{"text": "You bought vest top 9 times.", "type": "history",
               "product_type": "vest top", "count": 9, "window_days": None}]
    res = g.check(b, claims)
    assert res["passed"] is False
    assert res["violations"][0]["reason"] == "history_count_overstated"


def test_rule_gate_blocks_wrong_material_descriptor():
    s, g = _store(), _gate()
    b = A.build_bundle(s, "C1", "0000000001")   # jersey, not linen
    claims = [{"text": "This is linen.", "type": "descriptor", "term": "linen"}]
    res = g.check(b, claims)
    assert res["passed"] is False
    assert res["violations"][0]["field"] == "material/descriptor"


def test_rule_gate_blocks_invented_detail_on_null_desc():
    s, g = _store(), _gate()
    b = A.build_bundle(s, "C1", "0000000003")   # detail_desc is null
    claims = [{"text": "It has Italian leather trim.", "type": "attribute",
               "field": "detail_desc", "value": "italian leather trim"}]
    res = g.check(b, claims)
    assert res["passed"] is False


def test_rule_gate_passes_fully_grounded_explanation():
    s, g = _store(), _gate()
    b = A.build_bundle(s, "C1", "0000000001")
    claims = [
        {"text": "This is a black item.", "type": "attribute",
         "field": "colour_group_name", "value": "black"},
        {"text": "It is a vest top.", "type": "attribute",
         "field": "product_type_name", "value": "vest top"},
        {"text": "You bought vest top 2 times.", "type": "history",
         "product_type": "vest top", "count": 2, "window_days": None},
    ]
    res = g.check(b, claims)
    assert res["passed"] is True and res["violations"] == []


# --------------------------------------------------------------------------
# Task 5 — soft verifier boundary + behaviour
# --------------------------------------------------------------------------
def test_soft_verifier_flags_speculation():
    sv = SoftVerifier()  # offline heuristic
    claims = [{"text": "Perfect for summer parties.", "type": "reasoning"}]
    verdicts = sv.verify("Perfect for summer parties.", {}, claims)
    assert sv.any_unsupported(verdicts)


def test_soft_verifier_supports_grounded_claims():
    sv = SoftVerifier()
    claims = [{"text": "This is a black vest top.", "type": "attribute",
               "field": "colour_group_name", "value": "black"}]
    verdicts = sv.verify("This is a black vest top.", {}, claims)
    assert not sv.any_unsupported(verdicts)


def test_llm_soft_verifier_sees_explanation_and_bundle_only():
    """The verifier boundary: only the explanation + bundle cross it (no leaks)."""
    class _FakeResp:
        def __init__(self, text):
            self.content = [types.SimpleNamespace(type="text", text=text)]

    class _FakeMessages:
        def __init__(self, spy):
            self.spy = spy

        def create(self, **kw):
            self.spy.append(kw)
            return _FakeResp('{"verdicts": []}')

    class _FakeClient:
        def __init__(self, spy):
            self.messages = _FakeMessages(spy)

    spy = []
    v = LLMSoftVerifier(client=_FakeClient(spy))
    s = _store()
    bundle = A.build_bundle(s, "C1", "0000000001")
    v("This is a black vest top.", bundle, claims=[{"secret": "leaked?"}])
    sent = spy[0]["messages"][0]["content"]
    assert "This is a black vest top." in sent          # explanation present
    assert '"article_id"' in sent                        # bundle present
    # prod_name is NOT part of the bundle → must not leak into the verifier prompt.
    assert "SENTINEL_PRODNAME_X" not in sent
    # the raw claims list is not forwarded to the LLM verifier
    assert "leaked?" not in sent


# --------------------------------------------------------------------------
# Task 6/7 — pipeline: blocks are logged with a reason; adversarial cases caught
# --------------------------------------------------------------------------
def test_blocked_explanation_logged_with_reason():
    s, g = _store(), _gate()
    gen = _MockGen("This is a red item.",
                   [{"text": "This is a red item.", "type": "attribute",
                     "field": "colour_group_name", "value": "red"}])
    row = explain_and_verify(s, "C1", "0000000001", gen, g, SoftVerifier())
    assert row["blocked"] is True
    assert row["block_reason"] == "rule_violation"
    assert row["evidence_bundle"] and row["explanation_text"]


def test_grounded_explanation_passes_pipeline():
    s, g = _store(), _gate()
    gen = _MockGen("This is a black vest top from jersey basic.",
                   [{"text": "This is a black item.", "type": "attribute",
                     "field": "colour_group_name", "value": "black"},
                    {"text": "It is a vest top.", "type": "attribute",
                     "field": "product_type_name", "value": "vest top"}])
    row = explain_and_verify(s, "C1", "0000000001", gen, g, SoftVerifier())
    assert row["blocked"] is False and row["block_reason"] == ""


@pytest.mark.parametrize("expl,claim,exp_reason,exp_gate", [
    ("This is a purple item.",
     {"text": "purple", "type": "attribute", "field": "colour_group_name", "value": "purple"},
     "rule_violation", "rule"),
    ("This is linen.",
     {"text": "linen", "type": "descriptor", "term": "linen"},
     "rule_violation", "rule"),
    ("You bought ski suits 5 times.",
     {"text": "ski suit x5", "type": "history", "product_type": "ski suit",
      "count": 5, "window_days": 60},
     "rule_violation", "rule"),
    ("Perfect for summer parties, you'll love it.",
     {"text": "Perfect for summer parties.", "type": "reasoning"},
     "llm_unsupported", "llm"),
])
def test_adversarial_cases_are_caught(expl, claim, exp_reason, exp_gate):
    s, g = _store(), _gate()
    gen = _MockGen(expl, [claim])
    row = explain_and_verify(s, "C1", "0000000001", gen, g, SoftVerifier(),
                             regenerate=False)
    assert row["blocked"] is True
    assert row["block_reason"] == exp_reason


def test_template_generator_is_faithful_by_construction():
    """The offline generator must pass the rule gate on real evidence."""
    s, g = _store(), _gate()
    gen = A.TemplateGenerator()
    b = A.build_bundle(s, "C1", "0000000001")
    out = gen.generate(b)
    assert g.check(b, out["claims"])["passed"] is True


def test_parse_helpers_tolerate_fenced_json():
    out = A.parse_generation('```json\n{"explanation":"hi","claims":[]}\n```')
    assert out["explanation"] == "hi"
    assert parse_verdicts('{"verdicts":[{"claim":"c","verdict":"SUPPORTED"}]}')[0]["verdict"] == "SUPPORTED"


# --------------------------------------------------------------------------
# Prior artifacts intact (do not modify Phase 1-3 outputs)
# --------------------------------------------------------------------------
def test_prior_experiment_artifacts_intact():
    ap = config.PROCESSED_DIR / "actions.parquet"
    if ap.exists():
        assert len(pd.read_parquet(ap, engine="pyarrow")) == 128
    lap = config.PROCESSED_DIR / "labels_article.parquet"
    if lap.exists():
        assert pd.read_parquet(lap, columns=["customer_id"], engine="pyarrow")["customer_id"].nunique() == 15246
