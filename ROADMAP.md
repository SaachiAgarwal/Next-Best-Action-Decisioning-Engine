# ROADMAP — Next-Best-Action Decisioning Engine

Status-aware build plan. Reconciled to actual progress as of Week 2 complete.
Global rules: SEED = 42 everywhere stochastic. Every layer is a standalone
runnable script in src/. Named files, not throwaway notebooks. Specs for
upcoming phases are firmed up as we reach them, grounded in what the data
actually did the phase before.

Legend: [DONE] complete · [PARTIAL] started, gap remains · [TODO] not started

---

## WHERE WE ARE

Foundation and the candidate/scoring layer are built and rigorously tested.
The three things that make this a *decisioning engine* and a *PM* portfolio
piece — the bandit, the constraints/arbitration layer, the explanation agent,
the clickable UI, and the product+business framing — are the work ahead.

---

## PROJECT MAP

### Phase 0 — Product framing (NEW, front-loaded)
- [TODO] P0.1 — PRD / one-page product framing (problem, user, KPIs, the
  decision automated, scope, non-goals)

### Phase 1 — Data & candidate recommender
- [DONE] Cohort sampling (100k customers, seed=42, leakage-safe)
- [DONE] Action space (product-type, 128 actions) + event log + temporal split
- [PARTIAL] Feature/context layer — recency & frequency exist inside the
  hybrid, but no standalone reusable RFM + context vector yet (bandit needs it)
- [DONE] Candidate recommender: popularity, item-CF, granularity experiment
  (Exp A/B), recency+frequency hybrid (Exp 3) with divergent-slice eval

### Phase 2 — Contextual bandit (NBA core)
- [TODO] P2.1 — Context vector assembly + reward definition
- [TODO] P2.2 — UCB contextual bandit over candidates, with auditable
  decision logging (confidence terms recorded per decision)
- [TODO] P2.3 — Baselines (random, popularity) + reward/regret curves

### Phase 3 — Business constraints & arbitration (NEW — the "decisioning" layer)
- [TODO] P3.1 — Eligibility + fatigue/cooldown rules (using the ~12-day
  repurchase cadence), budget/frequency caps
- [TODO] P3.2 — Arbitration: pick the final action under constraints;
  log why each candidate was allowed/blocked (auditability)

### Phase 4 — RAG explanation agent + Lovable UI
- [TODO] P4.1 — Grounding corpus of real article facts + retrieval
- [TODO] P4.2 — Explanation generation + claim-fidelity safeguard (block any
  claim not supported by retrieved facts; log violations)
- [TODO] P4.3 — Lovable app: pick a customer -> see the NBA -> see the
  constraint-checked, grounded "why". Clickable, shareable demo.

### Phase 5 — Evaluation, business impact & write-up
- [TODO] P5.1 — Offline policy evaluation (IPS / replay) with variance
- [TODO] P5.2 — Business-metric translation (projected conversion uplift,
  revenue/customer, fatigue/over-targeting cost — the churn-project move)
- [TODO] P5.3 — Responsible-AI section: auditability, guardrails, fairness
  across segments, honest limitations
- [TODO] P5.4 — Model card (Mitchell et al. 2019) + README polish +
  fresh-clone reproducibility pass

---

## IMMEDIATE NEXT STEP
Phase 1 feature/context layer (the [PARTIAL] item) — a standalone RFM + context
vector. It's the prerequisite that unblocks the contextual bandit; the bandit
cannot be "contextual" without it.

---

## THE PORTFOLIO THESIS (what this project proves)
- Recommender: candidate generation + scoring (done, rigorously)
- Agent: contextual bandit making next-best-action decisions
- RAG: grounded, claim-checked explanations
- UI: a clickable Lovable demo (recruiters click, they don't read code)
- PM layer: PRD + business-metric impact + Responsible-AI/auditability —
  aligned to the "Responsible AI for high-stakes regulated decisioning" niche

---

## HONEST LIMITATIONS (maintained throughout, not bolted on at the end)
1. No real experiment — observational logs; simulated A/B and offline policy
   value are estimates, not causal proof.
2. Off-policy evaluation bias — a policy scored on logs it partly generated can
   look better than it is; report variance.
3. Reward proxy — purchase under-represents returns, satisfaction, and
   over-targeting fatigue.
4. Bounded cohort — 2018–2020, primarily European markets, sampled; limits
   generalization.
5. Test-set integrity — model *design* was informed by aggregate findings from
   earlier experiments; hyperparameters tuned on a feature-side validation
   slice, test labels touched only for final reporting.
