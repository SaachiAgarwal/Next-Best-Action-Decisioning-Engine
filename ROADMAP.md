# ROADMAP — Next-Best-Action Decisioning Engine

Status-aware build plan. SEED = 42 everywhere stochastic. Every layer is a
standalone runnable script in src/. Leakage-safe evaluation is a hard gate.

Legend: [DONE] · [NEXT] · [TODO]

## Phase 0 — Product framing
- [DONE] AI PRD (9-section template)

## Phase 1 — Data & recommender
- [DONE] Sampling (100k, seed=42), action space (product-type, 128), event log,
  leakage-safe temporal split
- [DONE] Recommender study (4 experiments):
  - Exp A: product-type popularity + item-CF (popularity won)
  - Exp B: article-level CF (collapsed on sparsity)
  - Exp 3: product-type recency+frequency hybrid (beat popularity, +3.6% divergent)
  - Exp 4: article-level content + content/CF hybrid (first article model to beat
    popularity, +44% @12; content/CF equal weights = complementary signals)
- [DONE] Standalone feature/context layer (RFM + attributes + breadth,
  cold-start flagged, leakage-guarded, model-ready encoding)

## Phase 2 — Contextual bandit (NBA core)
- [NEXT] Context + reward definition; UCB contextual bandit over the 128
  product-type actions, conditioned on customer_context, with auditable
  decision logging (confidence term per decision); cold-start (is_cold_start)
  falls back to a non-contextual policy
- [TODO] Baselines (random, popularity) + reward / regret curves

## Phase 2b — Hierarchical drill-down (Option C)
- [TODO] Within the bandit's chosen product-type, select the specific article
  using the Exp 4 content+CF recommender -> SKU-level next best action
  (reuses Exp 4; turns the study into a working engine layer)

## Phase 3 — Constraints & arbitration
- [TODO] Eligibility + fatigue/cooldown (using ~12-day repurchase cadence) +
  budget caps; arbitration picks the final action under constraints, logging
  why each candidate was allowed/blocked (auditability)

## Phase 4 — RAG explainer + Lovable UI
- [TODO] Grounding corpus of real article facts + retrieval; explanation
  generation + claim-fidelity safeguard (block/log unsupported claims);
  Lovable app: customer -> NBA -> constraint-checked grounded "why"

## Phase 5 — Evaluation & write-up
- [TODO] Offline policy evaluation (IPS / replay) with variance; business-metric
  translation (conversion uplift, fatigue cost); Responsible-AI section
  (auditability, fairness across segments); Mitchell et al. 2019 model card;
  README + fresh-clone reproducibility pass

## Honest limitations (maintained throughout)
Observational data (no causal proof); off-policy evaluation bias; purchase is an
imperfect reward proxy; bounded cohort (2018-2020, sampled, mostly European);
model design informed by aggregate findings, hyperparameters tuned on a
feature-side validation slice with test labels touched once.
