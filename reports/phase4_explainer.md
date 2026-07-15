# Phase 4 — Agentic RAG Explanation Layer

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

**Leakage guard:** every tool filters to `t_dat < 2020-08-26`. The
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
| wrong_colour_attribute | ✅ caught | rule |
| wrong_material_descriptor | ✅ caught | rule |
| fabricated_purchase | ✅ caught | rule |
| speculation_occasion | ✅ caught | llm |
| invented_detail_sparse_desc | ✅ caught | rule |

**Caught 5/5** — 4 by the hard rule gate,
1 by the soft LLM verifier. This split is the point: factual
fabrications (wrong colour, wrong material, invented purchase, invented
detail_desc text) are killed *deterministically*; only the ungrounded-reasoning
case relies on the soft stage. Concrete blocks:

**`wrong_colour_attribute`** — caught by the **RULE gate**

> Hostile explanation: *"This is a purple item you'll like."*
>
> rule violation → field **colour_group_name**, claimed *"purple"*, true value *"black"* (`attribute_mismatch`)

**`wrong_material_descriptor`** — caught by the **RULE gate**

> Hostile explanation: *"This is a linen garment from the same style family."*
>
> rule violation → field **material/descriptor**, claimed *"linen"*, true value *"None"* (`descriptor_not_in_article`)

**`fabricated_purchase`** — caught by the **RULE gate**

> Hostile explanation: *"Recommended because you bought ski suits five times recently."*
>
> rule violation → field **customer_history**, claimed *"ski suit x5"*, true value *"never purchased"* (`history_purchase_not_found`)

## Metrics

- **Sample size: 250 customers** — each customer's **top-ranked** article
  (`final_position == 0`) is explained. 250 is large enough for the fidelity and
  adversarial results to be meaningful, small enough to run on a modest budget
  (2 LLM calls/explanation on the live path).
- **Claim-fidelity rate (rule-based, hard gate): 100.0%** — 250/250
  explanations passed; **0 blocked**, **0 regenerated**. By
  construction this is 100% for *shipped* explanations: a claim that fails the
  hard gate is blocked, never shipped. (With the faithful offline generator the
  happy-path block count is low — the guard's teeth show in the adversarial run.)
- **LLM-flagged unsupported claims: 0** across the sample.
- **Adversarial catch rate: 5/5.**
- **Mean latency: 6.40 ms/explanation** on the offline path.
  Estimated **real-model** cost ≈ **$0.01030/explanation** (~2,168
  tokens over 2 `claude-sonnet-4-6` calls at $3/$15 per
  1M in/out) — i.e. roughly **$10.30 per 1,000 explanations**.

## How this run was executed (transparency)

Generation used the deterministic **offline** generator (`template-offline`) — no `ANTHROPIC_API_KEY` was available in this environment; soft verification used a deterministic **heuristic** soft verifier standing in for the LLM. The offline
generator is **faithful by construction** — it emits only bundle facts — so it is
the honest floor, not a claim of model-quality parity; the report does not
overstate it. Crucially, **the hard rule gate is identical on both paths**, and
the adversarial proof is driven by scripted hostile generations, so the guard's
guarantees hold with or without a live API. Component scores were available for
**250/250 (100%)** sampled pairs (the rest are not
warm in all three sub-models; their bundle carries the blended relevance instead).
Set `ANTHROPIC_API_KEY` and re-run to exercise the live `claude-sonnet-4-6` path.

## Example explanations (with evidence bundles)

**Customer `fcab545e35c843a6…` → article `0536139006`**

- *Evidence bundle (excerpt):* article = **black pyjama bottom**, group *nightwear*, department *nightwear*, appearance *solid*. History = 127 pre-cutoff purchases, 20 distinct types, top ['trousers', 'dress', 'bag']. Rec context = final_position 0, blended relevance 1.0, components (content=0.768, cf=0.120, mf=1.000), 2 candidates blocked.
- *Generated WHY:* "This is a black pyjama bottom from the nightwear department. You have previously bought black items. 2 other candidate(s) were filtered out (2 out_of_stock)."
- *Rule gate:* PASS · *soft verifier:* no flags

**Customer `99f83f5d515829cf…` → article `0610776002`**

- *Evidence bundle (excerpt):* article = **black t-shirt**, group *garment upper body*, department *jersey basic*, appearance *solid*. History = 105 pre-cutoff purchases, 20 distinct types, top ['t-shirt', 'dress', 'costumes']. Rec context = final_position 0, blended relevance 1.0, components (content=0.899, cf=0.157, mf=1.000), 94 candidates blocked.
- *Generated WHY:* "This is a black t-shirt from the jersey basic department. You have purchased t-shirt 18 time(s) in your history before the cutoff. 94 other candidate(s) were filtered out (88 category_cap, 6 out_of_stock)."
- *Rule gate:* PASS · *soft verifier:* no flags

**Customer `6856985afaa13b0d…` → article `0490793027`**

- *Evidence bundle (excerpt):* article = **white t-shirt**, group *garment upper body*, department *tops fancy jersey*, appearance *placement print*. History = 4 pre-cutoff purchases, 3 distinct types, top ['blouse', 't-shirt', 'top']. Rec context = final_position 0, blended relevance 1.0, components (content=0.999, cf=0.577, mf=0.361), 61 candidates blocked.
- *Generated WHY:* "This is a white t-shirt from the tops fancy jersey department. You have purchased t-shirt 1 time(s) in your history before the cutoff. 61 other candidate(s) were filtered out (55 category_cap, 6 out_of_stock)."
- *Rule gate:* PASS · *soft verifier:* no flags

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
