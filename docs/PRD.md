# AI PRD — Next-Best-Action Decisioning Engine

**Author:** Saachi Agarwal · **Status:** Living document · **v1.0**
*Portfolio project; framed as an internal product pitch at a fashion retailer. Structure follows the AI PRD template by Miqdad Jaffer (Product Lead, OpenAI).*

## 1. Executive Summary
This PRD describes a Next-Best-Action (NBA) Decisioning Engine that selects the single most valuable action to take with each customer at a given contact point, subject to business constraints, with a grounded, auditable explanation of why. It answers a clear need: personalization spend is rising while contact fatigue and explainability pressure grow, yet decisions are still made by broad segments. The engine pairs a contextual bandit with a business-rules layer and a RAG explanation agent. Target: a pilot on the CRM push/email channel.
**Success criteria:** (1) beat the popularity/segment baseline on conversion per contact; (2) reduce over-contact (fatigue cost) vs. current batch targeting; (3) 100% of explanations grounded in verified product facts (zero fabricated claims).

## 2. Market Opportunity
Retail personalization is a large, maturing market shifting from rule-based segmentation to adaptive, per-customer decisioning. The growth driver is not "AI hype" but a validated operational need: catalogs and contact channels have outgrown manual segmentation, and regulatory/trust pressure (explainability, fairness in automated decisions) is rising in parallel. The opportunity is to move from *recommendation* (a ranked list) to *decisioning* (one accountable, explainable action) — an area where most retailers have a capability gap. *(Portfolio note: quantified TAM/CAGR omitted as this is a demonstrative project; in a real PRD this section carries market-sizing data.)*

## 3. Strategic Alignment
For a fashion retailer whose strategy is customer lifetime value and efficient, trusted engagement, this engine aligns on three axes: it lifts relevance (conversion/LTV), it reduces fatigue and unsubscribes (retention and brand trust), and its auditability supports compliance with tightening automated-decision regulation. It plays to an existing strength — proprietary first-party purchase data — rather than a net-new competency. *(Personal alignment: this project sits in my niche — Responsible AI for high-stakes, regulated decisioning, carried over from fraud/risk product work.)*

## 4. Customer & User Needs
**Primary user:** the CRM / lifecycle marketing manager, accountable for both conversion and fatigue — competing goals today handled by blunt segment rules. **Job-to-be-done:** "For each customer I can reach, tell me the one action most likely to convert *without* over-contacting them, and let me trust and explain it." **Pain (high frequency, high severity, wide magnitude):** every contact cycle forces a relevance-vs-fatigue tradeoff across millions of customers with no per-customer basis. **End beneficiary:** the customer, who gets fewer, more relevant, explainable nudges. **Constraints:** first-party data only; explanations must be truthful; fairness across customer segments.

## 5. Value Proposition & Messaging
For CRM teams drowning in segment-level guesswork, the NBA engine delivers **one explainable next-best-action per customer** that lifts conversion while cutting fatigue — by combining recency/frequency/collaborative signal, a bandit that balances exploiting known preferences against exploring uncertain ones, and a constraints layer that enforces contact caps. Unlike a standard recommender, it produces an *auditable decision plus a grounded reason*, not just a ranked product list. Benefit, stated concretely: higher conversion per contact, lower unsubscribe risk, and a decision every stakeholder can inspect.

## 6. Competitive Advantage
Defensibility rests on three hard-to-copy assets, not UI: (1) proprietary first-party purchase history feeding the signal layer; (2) a decisioning architecture where business rules can *override and log* the model — the auditability that regulated decisioning demands and generic recommenders lack; (3) a claim-fidelity safeguard on explanations that makes the output trustworthy enough to ship in a compliance-sensitive setting. The moat is the *combination of adaptive personalization with enforceable, auditable governance*, which off-the-shelf recommendation APIs don't provide.

## 7. Product Scope & Use Cases
**In scope (v1):** per-customer NBA over product-type actions; recency+frequency+collaborative signal (built); contextual bandit for adaptive selection; business-constraints layer (eligibility, fatigue/cooldown using the observed ~12-day repurchase cadence, budget caps); RAG explanation with fidelity safeguard; a clickable demo (customer -> NBA -> grounded why).
**Desired outcomes:** beat popularity baseline (achieved on held-out data, concentrated on divergent customers); bandit regret below a random/popularity policy; 100% explanation fidelity.
**High-risk assumptions & tests:** *is finer granularity better?* — tested, no (SKU too sparse); *does personalization beat popularity?* — tested, yes, with richer signal at product-type level.
**Non-goals (v1):** real-time serving, live A/B (offline policy eval only), SKU-level recs, creative generation, multi-channel orchestration.

## 8. Non-Functional & AI-Specific Requirements
**General:** reproducible pipeline (seed-fixed); every layer a standalone runnable script; leakage-safe temporal evaluation as a hard gate.
**AI-specific:** action space = product-type (128 actions, justified by the granularity experiment); selection = UCB contextual bandit (chosen over a static ranker so decisions are adaptive *and* log a confidence term for audit); explanation = RAG grounded in real article facts with a **claim-fidelity safeguard** that blocks and logs any unsupported claim (target hallucination rate 0%); **fairness** measured as a guardrail across customer segments; honest limitations (off-policy bias, reward proxy, bounded cohort) tracked throughout and surfaced in a Mitchell-et-al. 2019 model card.

## 9. Go-to-Market Approach
Phased, evidence-first. **Phase 1 (build & prove) — the scope of this project:** offline end-to-end — establish the baseline, build the bandit + constraints + explanation layers, and evaluate with offline policy evaluation and business-metric translation (projected conversion uplift, fatigue reduction). This is where a portfolio project can produce real, defensible evidence.
**Phases 2-3 (pilot -> scale) — described for completeness, out of scope here:** *In a production setting*, Phase 2 would pilot a single channel (push or email) on a bounded early-adopter segment, measured against current batch targeting; Phase 3 would expand segments and channels on demonstrated lift, with continuous drift and fairness monitoring. Each phase advances only on measurable lift over baseline with no fatigue or fairness regression. *These phases require live traffic and A/B infrastructure and are therefore outside this project's scope — included to show the full path to production, not as work performed.*

## Honest Limitations
Observational data (no causal proof); offline policy evaluation carries off-policy bias; purchase is an imperfect reward proxy (ignores returns, satisfaction, fatigue); the cohort is bounded (2018-2020, sampled, primarily European markets). Model design was informed by aggregate findings from earlier experiments; hyperparameters were tuned on a feature-side validation slice, with test labels touched only for final reporting. These are stated up front and revisited in the model card.
