# Phase 5a — Demo Data Export (for the Lovable web app)

A browser app can't run the pipeline or read parquet, so this phase pre-computes
everything the interactive demo shows and writes it to **`data/demo/`** as compact
JSON (committed — total **480 KB**). No new models were trained; this reads
existing artifacts (Exp 5 models, Phase 3b re-ranker + frontier, Phase 4 explainer,
`customer_context`, `event_log`, `articles`, `labels_article`).

## Cohort (38 customers, seed 42)

Selected from the **evaluable** set (customers with label-window ground truth) to
showcase variety: {'cold_start': 6, 'high_freq': 12, 'mid_history': 12, 'divergent': 8}. Buckets:
- **high_freq** — warm customers with rich pre-cutoff history (top-decile frequency).
- **mid_history** — warm, middle-of-the-distribution frequency.
- **divergent** — warm customers whose taste is far from popularity (lowest
  `cosine_to_popularity` from `divergent_customers_exp3`); these best show the
  model beating a popularity baseline.
- **cold_start** — zero pre-cutoff history; every model falls back to popularity
  (the demo shows this honestly). **Data caveat:** in this dataset no zero-history
  customer has any label-window purchase (`cold_start ∩ evaluable = ∅`), so these
  customers carry an **empty `ground_truth`**. They are included specifically to
  demonstrate the popularity fallback, not to score hits — stated here rather than
  hidden.

Customers are anonymized to handles `C01…C38`; the same handle keys
every file.

| handle | segment |
|---|---|
| C01 | high_freq |
| C02 | high_freq |
| C03 | high_freq |
| C04 | high_freq |
| C05 | high_freq |
| C06 | high_freq |
| C07 | high_freq |
| C08 | high_freq |
| C09 | high_freq |
| C10 | high_freq |
| C11 | high_freq |
| C12 | high_freq |
| C13 | mid_history |
| C14 | mid_history |
| C15 | mid_history |
| C16 | mid_history |
| C17 | mid_history |
| C18 | mid_history |
| C19 | mid_history |
| C20 | mid_history |
| C21 | mid_history |
| C22 | mid_history |
| C23 | mid_history |
| C24 | mid_history |
| C25 | divergent |
| C26 | divergent |
| C27 | divergent |
| C28 | divergent |
| C29 | divergent |
| C30 | divergent |
| C31 | divergent |
| C32 | divergent |
| C33 | cold_start |
| C34 | cold_start |
| C35 | cold_start |
| C36 | cold_start |
| C37 | cold_start |
| C38 | cold_start |

## File schemas (exact field names for the Lovable prompt)

All floats are rounded; keys are short. Handles (`C01`…) are the join key.

### `customers.json` — list of customer objects
```
{ id, cid, seg,          # id = anonymized handle (frontend key); cid = source hash (join/test only)
   profile: { age_band, club, cold(bool), freq(int), recency_days,
              distinct_types(int), dominant_types:[str] },
   history: [ { aid, name, type, colour } ]            # top ~15, pre-cutoff, recent-first
   ground_truth: { n(int), articles:[ { aid, name, type } ] }  # LABEL WINDOW
}
```
`history`/`profile` are **pre-cutoff** (`t_dat < 2020-08-26`);
`ground_truth` is the **label window**. They never mix (leakage guard).

### `recommendations.json` — `{ handle: { variant: [item x12] } }`
Five variants per customer: **`hybrid`** (triple hybrid, production), **`mf`**
(MF alone), **`content`** (content alone), **`rerank_div`** (re-ranked λ=0.7,
pop=0.0 — the diversity setting), **`rerank_cov`** (re-ranked λ=0.3,
pop=0.3 — max-coverage). Each item:
```
{ aid, name, type, colour, dept,
   sc: { c, cf, mf } | null,   # normalized triple-hybrid component scores
   rank(int 0-11), hit(bool) }   # hit = article is in this customer's ground truth
```
`sc` is `null` for cold-start customers (not warm in the sub-models); all five
variants then equal the popularity fallback. Powers the **model toggle** and
**diversity slider**.

### `explanations.json` — `{ handle: { ... } }`
```
{ top_article, why(str), fidelity("passed"|"blocked"), block_reason,
   bundle: { ...evidence bundle (article_facts, customer_history summary,
              recommendation_context w/ component scores, constraint_decisions) },
   adversarial?: { attempted_claim, blocked(bool), gate, violated_field, true_value } }
```
The Phase-4 grounded "why" + the evidence bundle it drew on + the hard-gate
fidelity result. A few customers carry a pre-computed **adversarial** block (a
false colour claim the rule gate rejects) for the "watch it refuse to lie" panel.

### `frontier.json` — list of `(lambda, pop)` operating points
```
{ lambda, pop, recall12, cov12, hit12, gini, dissim, mean_pop_rank, distinct_types }
```
Straight from `rerank_frontier.parquet` (24 points). Lets the slider map to real
recall-vs-coverage points and plot the tradeoff curve.

### `diagnostics.json` — headline per-model diagnostics
```
{ model, hit12, recall12, cov12, mean_pop_rank, gini, dissim, distinct_types }
```
From `diagnostics_results.parquet` (+ `recall@12` joined from `metrics_summary`).
The summary panel: **accuracy and coverage rank in opposite orders** — MF tops
pop-rank/accuracy tradeoffs but covers <1% of the catalog; neighborhood CF covers
78% but is least accurate.

## File sizes

| file | size |
|---|---|
| `customers.json` | 49.3 KB |
| `recommendations.json` | 363.7 KB |
| `explanations.json` | 63.2 KB |
| `frontier.json` | 3.5 KB |
| `diagnostics.json` | 0.7 KB |
| **total** | **480.4 KB** |

## Notes / honesty

- Recommendation lists mirror the production retrieval construction (blend of the
  min-max-normalized content/CF/MF scores; re-rank over the blend's top-100),
  so the demo's numbers are the real pipeline's, not a re-derivation.
- Cold-start customers legitimately collapse to the popularity fallback across all
  five variants — the demo does not hide this.
- Explanations are generated by the Phase-4 **offline** faithful generator and pass
  the deterministic rule gate; they describe the evidence, not the model's internal
  causal mechanism (see Phase 4 limitations).


---

# Phase 5b — Product-Type Layer (granularity comparison)

Adds `data/demo/producttype.json` (54.8 KB, committed) so the app can show
the **same 38 customers at both granularities**: which *category* to recommend
(128 product-type actions) and — from 5a — which *specific product* (~79,269
articles). This makes **task difficulty visible**: the low SKU hit-rate is not model
failure, it is what predicting 1-of-79,269 looks like.

## `producttype.json` schema
```
{
  "comparison": {
     "product_type": { n_actions, exp3_hit@12, exp3_recall@12,
                       popularity_hit@12, margin_over_popularity },
     "article":      { n_articles, triple_hybrid_hit@12, triple_hybrid_recall@12,
                       popularity_hit@12, lift_over_popularity },
     "note": "why the two numbers are not comparable"
  },
  "customers": {
     "C01": {
        "exp3_hybrid":  [ { action_id, type, rank, hit } x12 ],  # Exp 3 production model
        "popularity":   [ { action_id, type, rank, hit } x12 ],  # Exp A baseline
        "ground_truth": { n, types:[str] }                       # LABEL-WINDOW product types
     }, ...
  }
}
```
`hit` = the customer bought that product type in the label window (from
`labels.parquet`, the product-type labels). Cold-start customers have an empty
`ground_truth` here too (same caveat as 5a).

## The two layers, honestly

| layer | task | production model | hit@12 | recall@12 | vs popularity |
|---|---|---|---|---|---|
| **product-type** | 1-of-128 | Exp 3 recency+freq hybrid | **0.8339** | 0.6164 | pop 0.8267 → **+0.7%** |
| **article** | 1-of-79,269 | Exp 5 triple hybrid | **0.0628** | 0.0229 | pop 0.0314 → **2.0x** |

- Product-type hit@12 (**0.8339**) LOOKS far better than article
  (**0.0628**), but they are **different tasks and NOT
  comparable** — 1-of-128 vs 1-of-79,269.
- At **product-type**, personalization beats popularity by only **+0.7%**
  (0.8339 vs 0.8267): the category task is easy for
  everyone, so personalization adds little.
- At **article**, the model **2.0x** the popularity baseline
  (0.0628 vs 0.0314): personalization earns
  real value exactly where the task is hard.
- Together the layers describe a **hierarchical architecture**: choose the category
  (tractable, learnable, where a bandit can operate), then the specific product
  within it (where content and latent factors earn their keep). **Honest caveat:**
  the drill-down chain is **not implemented end-to-end** — these are two parallel
  views of the same customer, and the chain is the natural next extension.
