# Week 2 — Granularity Experiment: does recommendation granularity decide
whether personalization beats popularity?

## The question

Across Week 2 we test one hypothesis: **the granularity of the action space
determines whether a personalized model can beat a non-personalized popularity
baseline.** Two arms, same 15,246 core evaluable customers, same leakage-safe
split, same metrics harness.

## Experiment A — product-type (128 actions) [recap]

At product-type granularity the action space is tiny and concentrated. Popularity
was a very strong baseline (hit-rate@12 = 0.827); item-to-item CF learned
real structure (swimwear↔swimwear, bra↔underwear) but **did not beat popularity**
at any k. The repeat effect was large — allowing already-bought product types
lifted hit-rate@6 from 0.349 to 0.668 — because category
repurchase is common.

| k | model | repeats | hit_rate | recall | precision |
|---|---|---|---|---|---|
| 6 | item_cf | False | 0.3493 | 0.1717 | 0.0736 |
| 6 | item_cf | True | 0.6684 | 0.4086 | 0.1858 |
| 6 | popularity | n/a | 0.7371 | 0.4843 | 0.2124 |
| 12 | item_cf | False | 0.4649 | 0.2594 | 0.0548 |
| 12 | item_cf | True | 0.8095 | 0.5927 | 0.1350 |
| 12 | popularity | n/a | 0.8267 | 0.6020 | 0.1342 |
| 24 | item_cf | False | 0.5492 | 0.3380 | 0.0356 |
| 24 | item_cf | True | 0.9505 | 0.8693 | 0.0983 |
| 24 | popularity | n/a | 0.9559 | 0.8752 | 0.0985 |

## Experiment B — article-level (~79,269 articles, sparse)

The "action" is now the exact `article_id`. Predicting 1 of ~79k
articles is far harder than 1 of 128, so **absolute numbers are expected to be
much lower** — the point is the *relative* comparison.

**Sparse engineering.** A dense 79,269² similarity matrix would be
**25.1 GB** — infeasible. We stay sparse end to end
(scipy CSR): binary interaction `A` (1,908,334 nnz, 15.7 MB),
cosine similarity via `AᵀA` normalized (106,935,208 nnz, 855.8 MB),
and per-customer scoring as a sparse indicator × similarity product that touches
only co-occurring neighbors.

| k | model | repeats | hit_rate | recall | precision |
|---|---|---|---|---|---|
| 6 | article_item_cf | False | 0.0173 | 0.0044 | 0.0031 |
| 6 | article_item_cf | True | 0.0175 | 0.0067 | 0.0033 |
| 6 | article_popularity | n/a | 0.0219 | 0.0058 | 0.0038 |
| 12 | article_item_cf | False | 0.0294 | 0.0078 | 0.0026 |
| 12 | article_item_cf | True | 0.0290 | 0.0110 | 0.0028 |
| 12 | article_popularity | n/a | 0.0314 | 0.0093 | 0.0028 |
| 24 | article_item_cf | False | 0.0458 | 0.0130 | 0.0021 |
| 24 | article_item_cf | True | 0.0445 | 0.0174 | 0.0022 |
| 24 | article_popularity | n/a | 0.0559 | 0.0166 | 0.0026 |

## Cross-granularity comparison

- **Popularity hit-rate@12:** product-type **0.8267** vs article-level
  **0.0314** — popularity is dramatically weaker once the target is a
  specific SKU rather than a broad category.
- **Does personalization beat popularity?**
- **k=6:** article item-CF does **not** beat article-popularity (0.0175 vs 0.0219, Δ=-0.0044, best repeats=True). For contrast, product-type item-CF vs popularity was Δ=-0.0687.
- **k=12:** article item-CF does **not** beat article-popularity (0.0294 vs 0.0314, Δ=-0.0020, best repeats=False). For contrast, product-type item-CF vs popularity was Δ=-0.0172.
- **k=24:** article item-CF does **not** beat article-popularity (0.0458 vs 0.0559, Δ=-0.0101, best repeats=False). For contrast, product-type item-CF vs popularity was Δ=-0.0054.
- **Repeat effect @6:** product-type 0.3493→0.6684
  (+0.3192) vs article 0.0173→0.0175
  (+0.0002). Exact-SKU repurchase is rarer
  than category repurchase, so the repeats lift is
  weaker at article level.

## Honest conclusion

Granularity changes all three things we measured:

1. **Absolute hit-rate** collapses at article level (1-of-79k
   is intrinsically hard) — from ~0.83 down to ~0.031 for popularity@12.
2. **The popularity-vs-personalization gap** shifts: product-type popularity is
   too strong to beat; article-level popularity is weak, which is where
   personalization has the most room.
3. **The repeat effect** weakens as granularity fines (category repurchase is
   common; exact-SKU repurchase is rarer).

**Headline.** At article level, popularity is far weaker in absolute terms, and sparse item-CF **still does not decisively beat** article-popularity. Ultra-fine granularity trades popularity's strength for extreme sparsity, so neither the coarse nor the ultra-fine extreme is ideal.

**Motivation for Week 3.** Neither extreme is ideal: coarse (product-type) kills
personalization headroom because popularity is unbeatable; ultra-fine (article)
kills density so every model struggles on absolute hit-rate. This argues for
**richer signal** — content/article features, a mid-level grouping between
product-type and SKU, and recency/sequence features — rather than co-occurrence
counts alone. That is the Week 3 direction.
