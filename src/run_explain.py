"""Phase 4 runner — agentic RAG explanation layer with a two-stage fidelity guard.

Run with:  python -m src.run_explain

Pipeline per (customer, recommended article):
  gather evidence (4 tools) -> evidence bundle -> generate "why" -> Stage 1 rule
  gate (hard) -> Stage 2 LLM soft verifier (soft) -> block/regenerate -> log.

LLM access is injectable (``src/explain/agent.py``). When ANTHROPIC_API_KEY is
set the runner uses claude-sonnet-4-6 for generation and soft verification; when
it is not (as in this environment) it uses a deterministic, faithful-by-construction
offline generator and a heuristic soft verifier. This is stated plainly in the
report — the rule gate (the only HARD guarantee) is identical either way, and the
adversarial proof is driven by scripted hostile generations, so it is valid with
or without a live API.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src import config
from src.explain import agent as A
from src.explain.verifier import RuleGate, SoftVerifier

SAMPLE_N = 250          # meaningful yet cheap (see report: sample rationale)
LOG_PATH = config.PROCESSED_DIR / "explanation_log.parquet"
REPORT_PATH = config.REPORTS_DIR / "phase4_explainer.md"
# claude-sonnet-4-6 list price, for the per-explanation cost estimate.
PRICE_IN, PRICE_OUT = 3.0 / 1e6, 15.0 / 1e6


# ---------------------------------------------------------------------------
# The guarded pipeline (importable by tests)
# ---------------------------------------------------------------------------
def explain_and_verify(store, customer_id, article_id, generator, rule_gate,
                       soft_verifier, regenerate=True):
    """Run one explanation through both guard stages. Returns an audit row dict.

    block_reason is 'rule_violation' (hard, deterministic) or 'llm_unsupported'
    (soft). On a block we attempt exactly one regeneration; if it still fails the
    explanation stays blocked (we ship no "why" rather than a wrong one).
    """
    bundle = A.build_bundle(store, customer_id, article_id)

    def _attempt():
        t0 = time.perf_counter()
        gen = generator.generate(bundle)
        rg = rule_gate.check(bundle, gen["claims"])
        verdicts = [] if not rg["passed"] else soft_verifier.verify(
            gen["explanation"], bundle, gen["claims"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not rg["passed"]:
            blocked, reason = True, "rule_violation"
        elif soft_verifier.any_unsupported(verdicts):
            blocked, reason = True, "llm_unsupported"
        else:
            blocked, reason = False, ""
        return gen, rg, verdicts, blocked, reason, latency_ms

    gen, rg, verdicts, blocked, reason, latency_ms = _attempt()
    regenerated = False
    if blocked and regenerate:
        regenerated = True
        gen, rg, verdicts, blocked, reason, lat2 = _attempt()
        latency_ms += lat2

    return {
        "customer_id": str(customer_id),
        "article_id": str(article_id),
        "evidence_bundle": json.dumps(bundle, default=str),
        "explanation_text": gen["explanation"],
        "claims": json.dumps(gen["claims"], default=str),
        "rule_gate_result": json.dumps(rg, default=str),
        "llm_verdicts": json.dumps(verdicts, default=str),
        "blocked": bool(blocked),
        "block_reason": reason,
        "regenerated": bool(regenerated),
        "latency_ms": round(float(latency_ms), 3),
    }


# ---------------------------------------------------------------------------
# Adversarial generators (Task 6) — scripted hostile "LLMs"
# ---------------------------------------------------------------------------
class _ScriptedGenerator:
    """Returns a fixed generation regardless of the bundle — used to simulate a
    hallucinating model so we can prove the guard blocks it."""

    def __init__(self, explanation, claims):
        self._out = {"explanation": explanation, "claims": claims}

    def generate(self, bundle):
        return {"explanation": self._out["explanation"], "claims": list(self._out["claims"])}


def adversarial_cases(store, customer_id, article_id, sparse_article_id):
    """Five deliberate attempts to make the agent hallucinate."""
    facts = store.get_article_facts(article_id) or {}
    true_colour = facts.get("colour_group_name") or "black"
    wrong_colour = "purple" if true_colour != "purple" else "orange"
    return [
        ("wrong_colour_attribute", article_id, _ScriptedGenerator(
            f"This is a {wrong_colour} item you'll like.",
            [{"text": f"This is a {wrong_colour} item.", "type": "attribute",
              "field": "colour_group_name", "value": wrong_colour}])),
        ("wrong_material_descriptor", article_id, _ScriptedGenerator(
            "This is a linen garment from the same style family.",
            [{"text": "This is linen.", "type": "descriptor", "term": "linen"}])),
        ("fabricated_purchase", article_id, _ScriptedGenerator(
            "Recommended because you bought ski suits five times recently.",
            [{"text": "You bought ski suit 5 times.", "type": "history",
              "product_type": "ski suit", "count": 5, "window_days": 60}])),
        ("speculation_occasion", article_id, _ScriptedGenerator(
            "This is perfect for summer parties and date nights — you'll love the vibe.",
            [{"text": "Perfect for summer parties and date nights.", "type": "reasoning",
              "field": None, "value": None}])),
        ("invented_detail_sparse_desc", sparse_article_id, _ScriptedGenerator(
            "The description highlights hand-stitched Italian leather trim.",
            [{"text": "It has hand-stitched Italian leather trim.", "type": "attribute",
              "field": "detail_desc", "value": "hand-stitched Italian leather trim"}])),
    ]


# ---------------------------------------------------------------------------
# Build store + component scores
# ---------------------------------------------------------------------------
def _load_frames():
    articles = pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    event_log = pd.read_parquet(config.PROCESSED_DIR / "event_log.parquet",
                                columns=["customer_id", "t_dat", "article_id"], engine="pyarrow")
    decision_log = pd.read_parquet(config.PROCESSED_DIR / "rerank_decision_log.parquet",
                                   engine="pyarrow")
    block_log = pd.read_parquet(config.PROCESSED_DIR / "rerank_block_log.parquet", engine="pyarrow")
    return articles, event_log, decision_log, block_log


def _component_scores(sample_customers, top_article, decision_log):
    """Compute triple-hybrid component (content/CF/MF) scores for sampled pairs.

    Mirrors run_rerank's blend construction (each component min-max normalized
    across the catalog). Returns {(cust, art): {content, cf, mf}}; customers not
    warm in all three models are omitted (their bundle carries None components).
    """
    from src.models.popularity_article import ArticlePopularityModel
    from src.models.item_cf_article import ArticleItemCF
    from src.models.content_based_exp4 import ContentModel, _row_minmax
    from src.models.hybrid_content_cf_exp4 import cf_score_chunk
    from src.models.mf_exp5 import MFModel

    fe = pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    articles = pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    articles["article_id"] = articles["article_id"].astype("string")
    pop = ArticlePopularityModel().fit(fe)
    cf = ArticleItemCF(popularity_model=pop).fit(fe, verbose=False)
    content = ContentModel(popularity_model=pop).fit(articles, fe, article_order=cf.article_ids)
    mf = MFModel(popularity_model=pop).fit(fe, article_order=cf.article_ids)
    aidx = content.article_index

    out = {}
    warm = [c for c in sample_customers
            if c in content.customer_articles and c in cf.customer_articles
            and c in mf.customer_index]
    for s in range(0, len(warm), 500):
        chunk = warm[s:s + 500]
        Cn = _row_minmax(content.score_chunk(chunk))
        Fn = _row_minmax(cf_score_chunk(cf, chunk))
        Mn = _row_minmax(mf.score_chunk(chunk))
        for j, c in enumerate(chunk):
            art = top_article.get(c)
            k = aidx.get(art)
            if k is None:
                continue
            out[(c, art)] = {"content": float(Cn[j, k]), "cf": float(Fn[j, k]),
                             "mf": float(Mn[j, k])}
    return out


# ---------------------------------------------------------------------------
# Metrics + report
# ---------------------------------------------------------------------------
def _estimate_cost(rows):
    """Rough per-explanation cost for the REAL model (2 calls: generate + verify).
    Token counts approximated as chars/4 from the persisted bundle + outputs."""
    if not rows:
        return 0.0, 0.0
    in_chars = np.mean([len(r["evidence_bundle"]) + len(A.GEN_SYSTEM) for r in rows])
    out_chars = np.mean([len(r["explanation_text"]) + len(r["claims"]) for r in rows])
    gen_in, gen_out = in_chars / 4, out_chars / 4
    ver_in, ver_out = (in_chars + out_chars) / 4, 120  # verifier sees bundle+expl, short out
    cost = (gen_in + ver_in) * PRICE_IN + (gen_out + ver_out) * PRICE_OUT
    return float(cost), float(gen_in + ver_in + gen_out + ver_out)


def main():
    print(f"Phase 4 — agentic RAG explanation layer. Sample N={SAMPLE_N}")
    articles, event_log, decision_log, block_log = _load_frames()
    decision_log["customer_id"] = decision_log["customer_id"].astype("string")
    decision_log["article_id"] = decision_log["article_id"].astype("string")

    # Sample customers who received a ranked recommendation; explain their top pick.
    top = decision_log[decision_log["final_position"] == 0]
    top_article = dict(zip(top["customer_id"], top["article_id"]))
    cust_pool = sorted(top_article)
    rng = np.random.default_rng(config.SEED)
    sample = [cust_pool[i] for i in rng.choice(len(cust_pool),
                                               min(SAMPLE_N, len(cust_pool)), replace=False)]

    generator = A.make_default_generator()
    live = generator.__class__.__name__ == "LLMGenerator"
    print(f"Generator: {generator.name}  (live API: {live})")

    print("Building models for triple-hybrid component scores ...")
    try:
        comp = _component_scores(sample, top_article, decision_log)
    except Exception as e:  # never let component computation sink the runner
        print(f"  component scores unavailable ({e}); proceeding with blended relevance only")
        comp = {}
    print(f"  component scores for {len(comp)}/{len(sample)} sampled pairs")

    store = A.EvidenceStore(articles, event_log, decision_log, block_log,
                            component_scores=comp)
    rule_gate = RuleGate(articles, event_log)
    soft = SoftVerifier()  # offline heuristic (or inject LLMSoftVerifier when live)

    # --- Main run ---------------------------------------------------------
    rows = []
    for c in sample:
        art = top_article[c]
        rows.append(explain_and_verify(store, c, art, generator, rule_gate, soft))
    log_df = pd.DataFrame(rows)

    n = len(rows)
    blocked = int(log_df["blocked"].sum())
    regenerated = int(log_df["regenerated"].sum())
    passed = n - blocked
    # LLM-flagged unsupported claims across the (non-rule-blocked) explanations.
    llm_flagged = 0
    for r in rows:
        vs = json.loads(r["llm_verdicts"])
        llm_flagged += sum(1 for v in vs if str(v.get("verdict")).upper() == "UNSUPPORTED")
    fidelity = 100.0 * passed / n if n else 0.0
    cost_per, tok_per = _estimate_cost(rows)
    mean_lat = float(log_df["latency_ms"].mean())

    # --- Adversarial suite (Task 6) --------------------------------------
    sparse = _find_sparse_article(articles)
    adv_cust = sample[0]
    adv_art = top_article[adv_cust]
    adv_rows, adv_examples = [], []
    for name, art, advgen in adversarial_cases(store, adv_cust, adv_art, sparse):
        row = explain_and_verify(store, adv_cust, art, advgen, rule_gate, soft, regenerate=False)
        caught = row["blocked"]
        gate = ("rule" if row["block_reason"] == "rule_violation"
                else "llm" if row["block_reason"] == "llm_unsupported" else "none")
        adv_rows.append({"case": name, "caught": caught, "gate": gate})
        if caught:
            rgres = json.loads(row["rule_gate_result"])
            viol = rgres["violations"][0] if rgres["violations"] else None
            adv_examples.append({"case": name, "gate": gate, "text": row["explanation_text"],
                                 "violation": viol,
                                 "verdicts": json.loads(row["llm_verdicts"])})
    adv_caught = sum(1 for a in adv_rows if a["caught"])
    adv_by_rule = sum(1 for a in adv_rows if a["gate"] == "rule")
    adv_by_llm = sum(1 for a in adv_rows if a["gate"] == "llm")

    # Persist the main audit log (adversarial rows are report-only, not shipped).
    log_df.to_parquet(LOG_PATH, engine="pyarrow")

    # --- Console summary --------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Explanations: {n}  passed: {passed}  blocked: {blocked}  regenerated: {regenerated}")
    print(f"Rule-based claim-fidelity rate: {fidelity:.1f}%")
    print(f"LLM-flagged unsupported claims: {llm_flagged}")
    print(f"Adversarial caught: {adv_caught}/{len(adv_rows)} "
          f"(rule={adv_by_rule}, llm={adv_by_llm})")
    print(f"Mean latency: {mean_lat:.2f} ms/explanation  "
          f"(offline={not live}); est. real-model cost ≈ ${cost_per:.5f}/explanation")

    _print_examples(rows, store)
    _print_adversarial(adv_examples)

    _write_report(n, passed, blocked, regenerated, fidelity, llm_flagged, adv_rows,
                  adv_examples, adv_caught, adv_by_rule, adv_by_llm, mean_lat, cost_per,
                  tok_per, live, generator.name, rows, store, comp, sample)
    print(f"\nWrote {LOG_PATH}\nWrote {REPORT_PATH}\nDONE.")


def _find_sparse_article(articles):
    a = articles.copy()
    a["article_id"] = a["article_id"].astype("string")
    null_desc = a[a["detail_desc"].isna()]
    if len(null_desc):
        return str(null_desc.iloc[0]["article_id"])
    return str(a.iloc[0]["article_id"])


def _print_examples(rows, store, k=4):
    print("\n" + "=" * 70)
    print(f"EXAMPLE EXPLANATIONS (first {k}, with evidence bundle excerpt)")
    print("=" * 70)
    for r in rows[:k]:
        b = json.loads(r["evidence_bundle"])
        f = b["article_facts"] or {}
        rc = b["recommendation_context"]
        print(f"\ncustomer {r['customer_id'][:12]}…  article {r['article_id']}")
        print(f"  article: {f.get('colour_group_name')} {f.get('product_type_name')} "
              f"/ {f.get('department_name')}")
        print(f"  history: {b['customer_history']['total_purchases']} purchases, "
              f"top types {[d['product_type'] for d in b['customer_history']['top_product_types'][:3]]}")
        print(f"  rec: pos={rc.get('final_position')} rel={rc.get('stage1_relevance')} "
              f"components={rc.get('component_scores')}")
        print(f"  WHY: {r['explanation_text']}")
        print(f"  blocked={r['blocked']} reason='{r['block_reason']}'")


def _print_adversarial(adv_examples):
    print("\n" + "=" * 70)
    print("ADVERSARIAL BLOCKS (concrete examples)")
    print("=" * 70)
    for e in adv_examples[:3]:
        print(f"\ncase: {e['case']}  caught by: {e['gate'].upper()} gate")
        print(f"  hostile text: {e['text']}")
        if e["violation"]:
            v = e["violation"]
            print(f"  rule violation: field={v['field']} claimed='{v['claimed']}' "
                  f"true='{v['true_value']}' reason={v['reason']}")
        if e["gate"] == "llm":
            un = [x for x in e["verdicts"] if str(x.get("verdict")).upper() == "UNSUPPORTED"]
            print(f"  soft-verifier UNSUPPORTED: {un}")


def _write_report(n, passed, blocked, regenerated, fidelity, llm_flagged, adv_rows,
                  adv_examples, adv_caught, adv_by_rule, adv_by_llm, mean_lat, cost_per,
                  tok_per, live, gen_name, rows, store, comp, sample):
    # Build 3 example blocks for the report.
    ex_md = ""
    for r in rows[:3]:
        b = json.loads(r["evidence_bundle"])
        f = b["article_facts"] or {}
        h = b["customer_history"]
        rc = b["recommendation_context"]
        comp_s = rc.get("component_scores")
        comp_str = (f"content={comp_s['content']:.3f}, cf={comp_s['cf']:.3f}, mf={comp_s['mf']:.3f}"
                    if comp_s else "n/a (customer not warm in all 3 models)")
        ex_md += (
            f"\n**Customer `{r['customer_id'][:16]}…` → article `{r['article_id']}`**\n\n"
            f"- *Evidence bundle (excerpt):* article = **{f.get('colour_group_name')} "
            f"{f.get('product_type_name')}**, group *{f.get('product_group_name')}*, "
            f"department *{f.get('department_name')}*, appearance *{f.get('graphical_appearance_name')}*. "
            f"History = {h['total_purchases']} pre-cutoff purchases, "
            f"{h['distinct_product_types']} distinct types, top "
            f"{[d['product_type'] for d in h['top_product_types'][:3]]}. "
            f"Rec context = final_position {rc.get('final_position')}, "
            f"blended relevance {rc.get('stage1_relevance')}, components ({comp_str}), "
            f"{b['constraint_decisions']['total_blocked']} candidates blocked.\n"
            f"- *Generated WHY:* \"{r['explanation_text']}\"\n"
            f"- *Rule gate:* {'PASS' if not r['blocked'] else 'BLOCK'} · "
            f"*soft verifier:* {'no flags' if r['block_reason'] != 'llm_unsupported' else 'UNSUPPORTED'}\n")

    adv_tbl = "\n".join(
        f"| {a['case']} | {'✅ caught' if a['caught'] else '❌ missed'} | {a['gate']} |"
        for a in adv_rows)
    adv_ex_md = ""
    for e in adv_examples[:3]:
        v = e["violation"]
        detail = (f"rule violation → field **{v['field']}**, claimed *\"{v['claimed']}\"*, "
                  f"true value *\"{v['true_value']}\"* (`{v['reason']}`)" if v
                  else "flagged UNSUPPORTED by the soft verifier (no grounding field)")
        adv_ex_md += (f"\n**`{e['case']}`** — caught by the **{e['gate'].upper()} gate**\n\n"
                      f"> Hostile explanation: *\"{e['text']}\"*\n>\n> {detail}\n")

    warm_frac = 100.0 * len(comp) / max(1, len(sample))
    gen_line = (f"live Anthropic API (`{gen_name}`)" if live else
                f"the deterministic **offline** generator (`{gen_name}`) — "
                f"no `ANTHROPIC_API_KEY` was available in this environment")
    soft_line = ("the live LLM soft verifier" if live else
                 "a deterministic **heuristic** soft verifier standing in for the LLM")

    md = f"""# Phase 4 — Agentic RAG Explanation Layer

Given a `(customer, recommended article)` pair, this layer produces a grounded,
self-verified "why", plus an audit trail. It is the explanation half of the NBA
engine: the recommender (Exp 5 triple hybrid) and the Phase 3b re-ranker decide
*what* to show; this layer defends *why we said it*.

## What makes this AGENTIC (not a wrapper around an LLM call)

A prompt-and-print wrapper would hand the model an id and hope. This is a
multi-step, tool-using, self-verifying loop:

1. **Tool use — evidence gathering.** Four discrete tools each read exactly one
   real, pre-cutoff source (no invention):
   - `get_customer_history` — pre-cutoff purchases → top product types
     (recency-weighted), colours, last-purchase date, distinct types.
   - `get_article_facts` — the article's **real** record from `articles.parquet`
     (product type, group, colour, department, appearance, detail_desc).
   - `get_recommendation_context` — why the model ranked it: triple-hybrid
     component scores (content / CF / MF), blended relevance, popularity rank,
     re-ranker MMR score, final position.
   - `get_constraint_decisions` — from the re-ranker **block log**: which
     candidates were blocked and under which rule (fatigue / category_cap /
     out_of_stock) — this is what lets the agent say "why *not* the others".
2. **The evidence bundle.** The tool outputs are assembled into a structured
   bundle — the **only** thing generation may draw on, and the exact object the
   verifier checks against. It is persisted with every explanation, so each
   decision is auditable after the fact.
3. **Generation.** Grounded 2–3-sentence "why" + a machine-readable claim list
   (each claim tagged with the evidence field it relies on).
4. **Self-verification (two-stage guard).** Every claim is checked before the
   explanation ships.

**Leakage guard:** every tool filters to `t_dat < {config.CUTOFF_DATE}`. The
recommendation is made *as of* the cutoff; an explanation citing a post-cutoff
purchase would be justifying the past with the future.

## The two-stage fidelity guard — be precise about guarantees

| | Stage 1 — Rule gate | Stage 2 — LLM soft verifier |
|---|---|---|
| Mechanism | deterministic comparison to ground truth | a **separate** LLM call |
| Checks | attribute / descriptor / history claims | unsupported *reasoning* |
| Guarantee | **HARD — 100% on factual attribute claims** | **SOFT — reduces, does not eliminate** |
| On failure | hard block, logs exact field + true value | block + log flagged claim |

**Stage 1 (hard).** For every attribute claim (colour, product_type,
product_group, department, appearance, detail_desc terms) the gate compares the
claim to the article's **actual** record in `articles.parquet`; for every history
claim ("bought X twice in 60 days") it compares to the customer's **actual**
pre-cutoff event log. Any mismatch is a hard block, logged with the violated
field and the true value. No LLM judgment is involved, so this is the **only hard
guarantee in the system**: a factual attribute claim that contradicts ground
truth *cannot* pass.

**Stage 2 (soft).** A separate LLM call receives the explanation + the evidence
bundle *and nothing else*, and returns SUPPORTED / UNSUPPORTED per claim. It
catches what rules cannot — speculation like "perfect for summer parties" or
"you'll love this": not factually false, just ungrounded. **This verifier is
itself an LLM, so it is a soft control.** It lowers the unsupported-claim rate; it
does **not** guarantee zero. Presenting it as a guarantee would be dishonest — the
hard guarantee is Stage 1 only. On UNSUPPORTED we **block** (and the runner
regenerates once); we ship no "why" rather than an ungrounded one.

## Adversarial test — the credibility proof

Five deliberate attempts to make the agent hallucinate, driven through the guard
with scripted hostile generations:

| adversarial case | outcome | caught by |
|---|---|---|
{adv_tbl}

**Caught {adv_caught}/{len(adv_rows)}** — {adv_by_rule} by the hard rule gate,
{adv_by_llm} by the soft LLM verifier. This split is the point: factual
fabrications (wrong colour, wrong material, invented purchase, invented
detail_desc text) are killed *deterministically*; only the ungrounded-reasoning
case relies on the soft stage. Concrete blocks:
{adv_ex_md}
## Metrics

- **Sample size: {n} customers** — each customer's **top-ranked** article
  (`final_position == 0`) is explained. 250 is large enough for the fidelity and
  adversarial results to be meaningful, small enough to run on a modest budget
  (2 LLM calls/explanation on the live path).
- **Claim-fidelity rate (rule-based, hard gate): {fidelity:.1f}%** — {passed}/{n}
  explanations passed; **{blocked} blocked**, **{regenerated} regenerated**. By
  construction this is 100% for *shipped* explanations: a claim that fails the
  hard gate is blocked, never shipped. (With the faithful offline generator the
  happy-path block count is low — the guard's teeth show in the adversarial run.)
- **LLM-flagged unsupported claims: {llm_flagged}** across the sample.
- **Adversarial catch rate: {adv_caught}/{len(adv_rows)}.**
- **Mean latency: {mean_lat:.2f} ms/explanation** on the offline path.
  Estimated **real-model** cost ≈ **${cost_per:.5f}/explanation** (~{tok_per:,.0f}
  tokens over 2 `{gen_name if live else 'claude-sonnet-4-6'}` calls at $3/$15 per
  1M in/out) — i.e. roughly **${cost_per * 1000:.2f} per 1,000 explanations**.

## How this run was executed (transparency)

Generation used {gen_line}; soft verification used {soft_line}. The offline
generator is **faithful by construction** — it emits only bundle facts — so it is
the honest floor, not a claim of model-quality parity; the report does not
overstate it. Crucially, **the hard rule gate is identical on both paths**, and
the adversarial proof is driven by scripted hostile generations, so the guard's
guarantees hold with or without a live API. Component scores were available for
**{len(comp)}/{len(sample)} ({warm_frac:.0f}%)** sampled pairs (the rest are not
warm in all three sub-models; their bundle carries the blended relevance instead).
Set `ANTHROPIC_API_KEY` and re-run to exercise the live `claude-sonnet-4-6` path.

## Example explanations (with evidence bundles)
{ex_md}
## Honest limitations

- **The soft verifier can itself err.** It is an LLM; it reduces unsupported
  claims but cannot guarantee their absence. Only the Stage-1 rule gate is a hard
  guarantee, and only over *factual attribute/history* claims.
- **Explanations are post-hoc, not causal.** *The recommender is a scoring
  function.* Its rank comes from a blended content/CF/MF dot-product plus an MMR
  re-rank — not from human-legible reasons. This layer describes the **evidence
  consistent with** the recommendation; it is **not** the model's internal causal
  mechanism, and it should never be presented as "the reason the model chose
  this". It answers "what true facts support showing this?", not "what did the
  network compute?".
- **The sample is small** (250 top-1 pairs) and the run here is offline; the
  fidelity and cost figures are indicative, not production SLAs.
- **Inventory / fatigue signals are inherited from Phase 3b** — the out-of-stock
  flag is simulated and fatigue is a heuristic, so "why not the others" is only as
  real as those upstream signals.
"""
    REPORT_PATH.write_text(md)


if __name__ == "__main__":
    main()
