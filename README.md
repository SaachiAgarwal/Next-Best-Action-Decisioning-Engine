# Next-Best-Action Decisioning Engine

A recommender + re-ranking + agentic-explanation system on the H&M dataset, built to answer a question accuracy metrics hide: **not just what to recommend, but whether the catalogue is actually being used.**

---

## The finding

> **My best-accuracy model recommended 0.8% of the catalogue. Accuracy never told me — coverage did.**

| | best-accuracy model (MF) | lowest-accuracy model (neighbourhood CF) |
|---|---|---|
| hit@12 | **0.0517** | 0.0290 |
| catalogue coverage@12 | **0.82%** | **77.8%** |

- **Accuracy and coverage rank the five models in opposite orders.** The most accurate model is the least diverse; the most diverse is the least accurate.
- Customers actually buy deep in the tail — the mean popularity rank of real purchases is **~35,279** (of 79,269 articles). The best-accuracy model recommends at mean rank **~170**. It wins the metric by re-selling the head.
- A hit-rate leaderboard would have shipped the 0.82%-coverage model and called it the winner.

This repo is how I found that, and what I built to manage it.

---

## Live demo + data

- **Interactive dashboard:** _[Lovable URL — placeholder; not yet linked in-repo]_
- **Dataset:** [H&M Personalized Fashion Recommendations (Kaggle)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) — 100k customers sampled with their complete purchase histories, `seed=42`. Data is git-ignored by design (see below); code and reports are committed.

---

## Production model

The **Exp 5 triple hybrid** (content + neighbourhood CF + matrix factorisation), at article (SKU) level:

| hit@12 | recall@12 | coverage@12 |
|---|---|---|
| **0.0628** | **0.0229** | **40.6%** |

**2.0× the popularity baseline on hit rate, 2.5× on recall.** Absolute numbers are low because the task is predicting one specific SKU out of **79,269** from purchase history alone — see [Reading the numbers honestly](#reading-the-numbers-honestly).

---

## The seven experiments

Each experiment is a full write-up in [`reports/`](reports/). The negatives are kept as prominently as the wins — they are the point.

| # | Experiment | Granularity | Result |
|---|---|---|---|
| A | popularity + item-CF | product-type (128) | popularity won; personalisation had no headroom on the easy task |
| B | article-level CF | article (79k) | collapsed on sparsity — co-occurrence too thin at SKU level |
| 3 | recency + log-frequency hybrid | product-type | beat popularity; larger lift on the divergent (niche-taste) slice (**+3.6%** at k=6) |
| 4 | content + CF hybrid | article | first article-level model to beat popularity (hit@12 0.0454) |
| 5 | matrix factorisation + triple hybrid | article | MF the strongest single signal; **triple hybrid is the production model** |
| 6 | customer attributes | article | clean null — attribute weight tuned to **exactly 0** (behaviour subsumes demographics) |
| 7 | temporal signals | article | trend helps accuracy but costs **11pp coverage** (40.6%→29.2%); seasonality a horizon-limited null; contact-timing bands retained |

---

## Architecture

```
retrieval            re-ranking                     explanation                 audit
──────────           ──────────                     ───────────                 ─────
triple hybrid  ─►  MMR + popularity penalty  ─►  evidence tools ─► grounded  ─►  decision +
(content+CF+MF)    + fatigue/category/            generation ─► two-stage        block logs
                   inventory constraints          fidelity guard                 (parquet)
```

A **contextual bandit** (LinUCB over customer × action features) is also implemented and beat popularity on held-out data at product-type level. It is retained as a documented experiment with a stated offline-evaluation limitation (off-policy bias — see [Limitations](#honest-limitations)), not shipped as the default.

---

## What I built to manage the finding

**Diagnostics suite** — the metrics accuracy cannot see: coverage, popularity bias, Gini, intra-list diversity, cold-start asymmetry, segment fairness. This is what surfaced the coverage inversion above. → [`reports/phase3a_diagnostics.md`](reports/phase3a_diagnostics.md)

**Re-ranking layer** — MMR + popularity penalty + fatigue / category-cap / (simulated) inventory constraints, producing an explicit **accuracy-vs-coverage frontier**. Honest operating-point tradeoff: coverage **40.6% → 49.6%** (1.22×) for a **~11% recall cost**; adding the business constraints improves segment-fairness spread **0.0688 → 0.0400** at a larger accuracy cost. → [`reports/phase3b_reranker.md`](reports/phase3b_reranker.md)

**Agentic explanation layer** — tool-based evidence gathering → grounded generation → a **deterministic rule gate** (hard guarantee on factual attribute/history claims) + an **LLM soft verifier** (probabilistic, for unsupported reasoning). Adversarial test: **5/5 hallucination attempts blocked** (4 by the rule gate, 1 by the soft verifier). → [`reports/phase4_explainer.md`](reports/phase4_explainer.md)

---

## Reading the numbers honestly

- **Why absolute hit rates look low.** The task is 1-of-79,269 SKU prediction from purchase history alone, on a sampled dataset with no browsing, session, or image data. Relative to the popularity baseline the production model is **2.0× on hit rate and 2.5× on recall** — that ratio is the signal, not the absolute.
- **Product-type ≠ article numbers.** Product-type hit@12 (**0.8339**, vs popularity 0.8267 — a mere **+0.7%**) is *not* comparable to article-level hit@12 (**0.0628**): 128 options vs 79,269, two different tasks. The category task is easy and barely beats popularity; the SKU task is hard and doubles it.
- **Hit rate and recall disagree.** MF posts a higher hit rate than the content+CF hybrid (0.0517 vs 0.0454) but **lower recall** (0.0170 vs 0.0176) — it catches one popular item per basket and misses the rest. Reporting only one metric would mislead.

---

## Honest limitations

- **Observational data, no live experiment** — no causal claims are made anywhere.
- **Offline policy evaluation carries off-policy bias**; the bandit's exploration value in particular cannot be validated offline.
- **Purchase is an imperfect reward proxy** — no returns, satisfaction, or fatigue signal in the data.
- **Sampled cohort** — 100k customers, 2018–2020, primarily European markets.
- **Explanations are post-hoc** — the recommender is a scoring function, so the "why" describes correlated evidence, not the model's internal causal reasoning.
- **The explanation layer ran on a deterministic offline generator** (no API key available). The fidelity guard is model-agnostic and validates any generator's output identically — the guarantee is on the *checker*, not the writer.
- **Inventory constraints are simulated** (the dataset has no stock data); the mechanism is real, the stock is not.
- **Tuning discipline** — model design was informed by aggregate findings from earlier experiments; hyperparameters were tuned on a feature-side validation slice, with the held-out test labels touched once for the final numbers.

---

## Repo structure & reproduce

```
src/
  data/        load, clean, action space, event log, temporal splits
  models/      popularity, item-CF, content, MF, hybrids, LinUCB bandits, temporal
  eval/        metrics + beyond-accuracy diagnostics
  features/    RFM + context layer
  rerank/      MMR + popularity penalty + business constraints
  explain/     evidence tools + agent + two-stage fidelity verifier
  run_*.py     one runnable entrypoint per experiment/phase
reports/       one markdown write-up per experiment/phase
tests/         leakage guards, regression guards, artifact-integrity checks
data/          raw + processed (git-ignored); demo/ JSON (committed for the app)
docs/PRD.md · ROADMAP.md
```

**Setup**
```bash
pip install -r requirements.txt
# place the three H&M CSVs (articles, customers, transactions) in data/raw/
python -m src.data.load            # sample 100k customers, seed 42
python -m src.data.build_clean_actions
python -m src.data.build_events
python -m src.data.build_splits    # leakage-safe temporal split (cutoff 2020-08-26)
# then any experiment, e.g.:
python -m src.run_mf_exp5          # triple hybrid (production)
python -m src.run_rerank           # diversity/coverage frontier
python -m src.run_explain          # agentic explanation layer
```

**Data is git-ignored by design** — code and reports are committed, data is not (the committed `data/demo/*.json` is the one exception, for the web app).

**Tests**
```bash
python -m pytest -q
```
The suite covers leakage guards (pre-cutoff features / post-cutoff labels), regression guards (byte-identical demo exports, the w4=w5=0 temporal reproduction of the triple hybrid), and prior-artifact-integrity checks (128 actions, 15,246 evaluable customers).

---

## Full write-ups

| Report | What it covers |
|---|---|
| [week1_data_profile.md](reports/week1_data_profile.md) | Load, sample, clean, and the leakage-safe temporal split |
| [week2_baseline.md](reports/week2_baseline.md) | Popularity baseline + the evaluation harness |
| [week2_item_cf.md](reports/week2_item_cf.md) | Item-to-item collaborative filtering (product-type) |
| [week2_granularity_experiment.md](reports/week2_granularity_experiment.md) | Article-level CF and the sparsity collapse (Exp B) |
| [week2_exp3_hybrid.md](reports/week2_exp3_hybrid.md) | Recency + log-frequency hybrid; divergent-slice lift (Exp 3) |
| [exp4_content_hybrid.md](reports/exp4_content_hybrid.md) | Content + CF hybrid — first article-level model to beat popularity (Exp 4) |
| [exp5_mf.md](reports/exp5_mf.md) | Matrix factorisation + the triple hybrid production model (Exp 5) |
| [exp6_attributes.md](reports/exp6_attributes.md) | Customer attributes — the clean null (Exp 6) |
| [exp7_temporal.md](reports/exp7_temporal.md) | Trend, seasonality, and contact-timing bands (Exp 7) |
| [context_layer.md](reports/context_layer.md) | RFM + context feature layer |
| [phase2_bandit.md](reports/phase2_bandit.md) · [_v2](reports/phase2_bandit_v2.md) · [_v3](reports/phase2_bandit_v3.md) · [2c](reports/phase2c_bandit_shared.md) | LinUCB contextual bandit — fair evaluation, per-action features, shared model |
| [phase3a_diagnostics.md](reports/phase3a_diagnostics.md) | Beyond-accuracy diagnostics — where the coverage finding surfaced |
| [phase3b_reranker.md](reports/phase3b_reranker.md) | MMR + constraint re-ranking; the accuracy-coverage frontier |
| [metrics_summary.md](reports/metrics_summary.md) | Full cross-model metric table |
| [phase4_explainer.md](reports/phase4_explainer.md) | Agentic RAG explanation layer + two-stage fidelity guard |
| [phase5a_export.md](reports/phase5a_export.md) | Demo data exports (cohort, recommendations, explanations, frontier, diagnostics, product-type, contact-timing, segments) |

---

## Author

**Saachi Agarwal** — Senior PM · recommender systems and responsible AI for regulated decisioning.

[GitHub](https://github.com/SaachiAgarwal/Next-Best-Action-Decisioning-Engine) · _LinkedIn — placeholder_ · _Live demo — placeholder_
