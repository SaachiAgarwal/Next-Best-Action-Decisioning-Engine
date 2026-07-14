# Phase 3b — Diversity + Constraint Re-Ranking

## The pathology this fixes (Phase 3a)

The production triple hybrid covers only **40.6%** of the catalog, draws **69%** of
recommendations from the top-10% head (Gini **0.886**), has intra-list diversity
**0.453**, and serves at mean popularity rank **~10,015** while customers actually
buy at rank **~35,279**. Accuracy metrics are blind to all of this. This layer
trades a controlled slice of accuracy to fix it.

## Two-stage architecture

**Stage 1 (retrieval):** the triple hybrid scores and returns the top-100
candidates per customer — "what is relevant". **Stage 2 (re-rank):** selects the
final 12 — "what should we actually show". Different questions: relevance is
necessary but not sufficient; the shown list must also be diverse, expose
inventory, and obey business rules. Separating them lets us tune the second
without touching the first.

## MMR, and why diversity ≠ coverage

MMR builds the list greedily, each pick maximizing
`λ·rel(i) − (1−λ)·max_{j∈selected} sim(i,j)` — trading relevance against
similarity to what's already chosen (cosine in the Exp 4 content space). λ=1 is
pure relevance (reproduces stage-1); λ→0 is pure diversity.

But **MMR alone cannot fix coverage**: it makes each *list* varied, yet everyone
could still get a varied list from the same popular head. So we add an explicit
**popularity penalty** — `adj_rel(i) = rel(i) − POP_PENALTY·pop_score(i)` — that
pushes the whole system into the long tail. λ controls *within-list* diversity;
POP_PENALTY controls *catalog* coverage. Both are needed.

## Business constraints (the decisioning layer)

Applied as hard filters at the operating point, each logged with a reason:
- **fatigue:** drop a product type the customer bought within 12
  days (the measured repurchase cadence) — a restock nudge days after purchase is
  wasted contact.
- **category cap:** at most 3 of one product type in the final
  12 — fixes the "12 near-identical items" failure (diagnostics found content
  lists averaged 1.4 distinct types).
- **eligibility:** drop out-of-stock articles. **⚠️ Inventory is SIMULATED** — a
  seeded random 5% marked out-of-stock; the dataset has no real stock data. This
  demonstrates the mechanism only.

## THE FRONTIER (the central deliverable)

Each row is one (λ, POP_PENALTY) setting; constraints off to isolate the
accuracy-coverage tradeoff. Baseline = **λ=1, pop=0** (the unmodified triple hybrid).

| λ | pop_pen | recall@12 | hit@12 | coverage@12 | mean pop rank | Gini | intra-list dissim |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.0 | 0.0214 | 0.0608 | 47.1% | 12,566 | 0.854 | 0.584 |
| 0.3 | 0.1 | 0.0215 | 0.0606 | 48.5% | 14,827 | 0.842 | 0.584 |
| 0.3 | 0.3 | 0.0204 | 0.0586 | 49.6% | 19,620 | 0.831 | 0.582 |
| 0.3 | 0.5 | 0.0189 | 0.0561 | 49.5% | 23,813 | 0.828 | 0.578 |
| 0.5 | 0.0 | 0.0230 | 0.0643 | 45.2% | 11,699 | 0.865 | 0.557 |
| 0.5 | 0.1 | 0.0226 | 0.0630 | 47.3% | 14,529 | 0.850 | 0.558 |
| 0.5 | 0.3 | 0.0213 | 0.0607 | 48.3% | 21,557 | 0.837 | 0.554 |
| 0.5 | 0.5 | 0.0200 | 0.0573 | 47.8% | 27,410 | 0.837 | 0.543 |
| 0.7 | 0.0 | 0.0232 | 0.0647 | 43.4% | 10,906 | 0.873 | 0.519 |
| 0.7 | 0.1 | 0.0229 | 0.0640 | 45.9% | 14,141 | 0.856 | 0.521 |
| 0.7 | 0.3 | 0.0212 | 0.0601 | 47.2% | 23,070 | 0.843 | 0.513 |
| 0.7 | 0.5 | 0.0197 | 0.0567 | 46.5% | 29,579 | 0.844 | 0.499 |
| 0.8 | 0.0 | 0.0232 | 0.0640 | 42.4% | 10,563 | 0.878 | 0.497 |
| 0.8 | 0.1 | 0.0229 | 0.0636 | 45.2% | 13,982 | 0.860 | 0.498 |
| 0.8 | 0.3 | 0.0214 | 0.0603 | 46.6% | 23,540 | 0.846 | 0.491 |
| 0.8 | 0.5 | 0.0194 | 0.0564 | 45.9% | 30,132 | 0.847 | 0.478 |
| 0.9 | 0.0 | 0.0232 | 0.0637 | 41.4% | 10,254 | 0.882 | 0.473 |
| 0.9 | 0.1 | 0.0230 | 0.0634 | 44.4% | 13,832 | 0.864 | 0.475 |
| 0.9 | 0.3 | 0.0215 | 0.0607 | 45.9% | 23,784 | 0.850 | 0.469 |
| 0.9 | 0.5 | 0.0194 | 0.0559 | 45.3% | 30,468 | 0.850 | 0.459 |
| 1.0 | 0.0 | 0.0229 | 0.0628 | 40.6% | 10,015 | 0.886 | 0.453 |
| 1.0 | 0.1 | 0.0227 | 0.0627 | 43.5% | 13,706 | 0.867 | 0.455 |
| 1.0 | 0.3 | 0.0212 | 0.0594 | 45.2% | 23,854 | 0.853 | 0.451 |
| 1.0 | 0.5 | 0.0195 | 0.0559 | 44.8% | 30,633 | 0.853 | 0.443 |

**Reading the frontier:** moving down/right (lower λ, higher POP_PENALTY) trades
recall for coverage, tail depth (higher mean pop rank), lower Gini, and higher
list diversity. Accuracy loss is **real, not free**.

## Recommended operating point — λ=0.3, POP_PENALTY=0.3

Chosen by a **product rule, not metric-maximization**: on the frontier (tuning
only), *maximize catalog coverage subject to ≤15% recall loss.* Two effects are
separated below: the **tuning** (MMR + pop-penalty, the accuracy-coverage trade)
and the **business constraints** applied on top (which enforce rules at a further
accuracy cost).

| metric | baseline (triple hybrid) | tuning only (λ=0.3, pop=0.3) | + business constraints |
|---|---|---|---|
| recall@12 | 0.0229 | 0.0204 (-11.1%) | 0.0116 (-49.4%) |
| hit@12 | 0.0628 | 0.0586 | 0.0357 |
| coverage@12 | 40.6% | 49.6% (1.22×) | 38.2% |
| mean pop rank | 10,015 | 19,620 | 15,511 |
| top-10% head share | 69.0% | 53.4% | 61.2% |
| Gini | 0.886 | 0.831 | 0.877 |
| intra-list dissimilarity | 0.453 | 0.582 | 0.622 |
| distinct types / list | 3.5 | 5.0 | 4.8 |
| segment fairness spread | 0.0688 | 0.0625 | 0.0400 |

**Plain language — the tuning:** at λ=0.3, POP_PENALTY=0.3 (no constraints),
recall goes 0.0229→0.0204 (-11.1%) while
coverage rises 40.6%→49.6%
(1.22×), Gini 0.886→0.831, and lists carry
5.0 distinct product types (up from 3.5).
This is a **real, deliberate trade** — recall for discovery/diversity — that
hit-rate alone would never surface.

**The honest catch — coverage is retrieval-bounded.** Re-ranking only reorders each
customer's fixed top-100 candidates, so it **cannot** reach
neighborhood CF's 77.8% coverage; the achievable gain here is modest
(40.6%→49.6%). Broadening coverage further
requires a **more diverse retriever**, not a smarter re-ranker — a limitation the
frontier makes explicit.

**Adding the business constraints** (fatigue + category cap + simulated OOS) costs
*more* accuracy (recall 0.0204→0.0116, now -49.4%
vs baseline) and does not raise article coverage (they filter, not broaden), but
they sharply improve **list quality and fairness**: distinct types per list rise to
4.8, intra-list dissimilarity to 0.622,
head share falls to 61.2%, and the **segment fairness spread**
goes 0.0688→0.0400 — re-ranking makes the model
**MORE equitable** across customer segments. Whether the large accuracy cost is worth
the rule-compliance and fairness gains is a genuine product call; this report gives
the numbers to make it, not a verdict.

## Constraint blocks (operating point)

Across all customers, the hard filters blocked: **fatigue 231,161**,
**category cap 744,048**, **out-of-stock (simulated)
73,207** candidates. Every block is logged with its
`rule_violated` in `rerank_block_log.parquet` — the Responsible-AI audit trail.

## Honest limitations

- **Re-ranking cannot exceed the retrieval ceiling** — it only reorders/filters the
  top-100; it cannot surface a relevant article stage-1 missed.
- **The accuracy loss is real**, not free (tuning -11.1% recall,
  -49.4% with constraints) — a deliberate product trade, not a Pareto
  improvement.
- **Inventory is simulated** (seeded 5% OOS) — the mechanism is real, the stock
  data is not.
- **Fatigue is a heuristic** (the measured 12-day cadence), not a learned policy.
