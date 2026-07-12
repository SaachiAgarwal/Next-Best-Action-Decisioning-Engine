# Experiment 5 — Article-Level Matrix Factorization

## MF vs neighborhood CF

Both are collaborative filtering, but they escape sparsity differently.
**Neighborhood CF** (Exp B) relates two articles only through *explicit
co-occurrence*: it needs customers who bought **both**. With ~82k SKUs almost no
pair co-occurs, so the signal collapses. **Matrix factorization** learns a
low-dimensional latent vector for each customer and article so that
`dot(U, V)` reconstructs the interactions; articles used in similar contexts land
near each other in factor space, letting MF **generalize** to pairs that never
co-occurred. That is the theoretical reason MF *should* handle sparsity better.

We use **implicit-feedback ALS** (binary purchases, no ratings — explicit-rating
SVD would be inappropriate), aligned to the Exp B / Exp 4 article space.

## The sparsity problem (stated explicitly)

The customer×article interaction matrix is **97,696 × 79,269**
with only **0.0246% non-zero**. This extreme sparsity *is* the
problem MF is trying to solve — and, as the diagnostics show, MF is not immune to
it.

## Embedding-quality diagnostics (the honest check)

- **38% of articles have fewer than 5 interactions**, so their
  factors are recovered from almost no signal.
- Embedding norm grows with interaction count (thin articles barely move from the
  prior):

| interaction bucket | # articles | mean ‖V‖ |
|---|---|---|
| <5 | 30,018 | 0.007 |
| 5-49 | 38,918 | 0.039 |
| 50+ | 10,333 | 0.212 |

- **Nearest-neighbor sanity check.** For a representative popular article of each
  type, the top-5 MF neighbors — note they are **not** attribute-coherent, and
  the neighbors are themselves mostly long-tail (few interactions), i.e. noise.
  Overall NN type-coherence = **0.10** (Exp 4 content neighbors
  were near-perfectly type/colour coherent):

**0706016001** (trousers, black) — top-5 MF neighbors:

| article_id | product_type | colour | interactions | cos |
|---|---|---|---|---|
| 0579010037 | sweater | black | 1 | 0.849 |
| 0784476001 | jacket | dark beige | 1 | 0.849 |
| 0898410001 | hat/beanie | beige | 1 | 0.849 |
| 0695662010 | socks | dark grey | 1 | 0.849 |
| 0766595006 | trousers | black | 1 | 0.832 |

**0673677002** (sweater, black) — top-5 MF neighbors:

| article_id | product_type | colour | interactions | cos |
|---|---|---|---|---|
| 0567435001 | bag | black | 1 | 0.954 |
| 0593829011 | t-shirt | yellow | 1 | 0.950 |
| 0553403003 | robe | white | 1 | 0.950 |
| 0682088001 | sunglasses | black | 1 | 0.950 |
| 0500864001 | slippers | gold | 1 | 0.950 |

**0590928001** (bikini top, black) — top-5 MF neighbors:

| article_id | product_type | colour | interactions | cos |
|---|---|---|---|---|
| 0669642001 | sweater | black | 2 | 0.940 |
| 0712924012 | swimwear bottom | black | 204 | 0.929 |
| 0861995002 | t-shirt | light green | 3 | 0.926 |
| 0873829002 | skirt | black | 1 | 0.918 |
| 0388916001 | belt | yellowish brown | 2 | 0.914 |

**0723469001** (bra, black) — top-5 MF neighbors:

| article_id | product_type | colour | interactions | cos |
|---|---|---|---|---|
| 0649023001 | earring | gold | 1 | 0.896 |
| 0563276002 | t-shirt | black | 2 | 0.885 |
| 0568905014 | shorts | grey | 4 | 0.884 |
| 0723469002 | bra | light beige | 307 | 0.877 |
| 0911167005 | hoodie | light beige | 1 | 0.875 |

This is the **sparse-factor-recovery problem**, reported honestly: at this density
even well-observed articles get noisy neighbors, because their nearest points in
factor space are thin-data articles sitting near the prior.

## MF_FACTORS sweep (hit@12, repeats=True)

| factors | hit@12 |
|---|---|
| 32 | 0.05287 |
| 64 | 0.05169 |
| 128 | 0.05280 |

## Five-model comparison (all article level, 15,246 core customers)

| k | model | repeats | hit_rate | recall | precision |
|---|---|---|---|---|---|
| 6 | MF | False | 0.02079 | 0.00567 | 0.00363 |
| 6 | MF | True | 0.03922 | 0.01279 | 0.00731 |
| 6 | article popularity | n/a | 0.02191 | 0.00584 | 0.00378 |
| 6 | content | False | 0.00905 | 0.00286 | 0.00161 |
| 6 | content | True | 0.02138 | 0.00839 | 0.00386 |
| 6 | content+CF hybrid | True | 0.03161 | 0.01210 | 0.00597 |
| 6 | neighborhood CF | False | 0.01732 | 0.00444 | 0.00308 |
| 6 | neighborhood CF | True | 0.01751 | 0.00674 | 0.00333 |
| 12 | MF | False | 0.03017 | 0.00829 | 0.00268 |
| 12 | MF | True | 0.05169 | 0.01696 | 0.00496 |
| 12 | article popularity | n/a | 0.03142 | 0.00926 | 0.00283 |
| 12 | content | False | 0.01482 | 0.00460 | 0.00132 |
| 12 | content | True | 0.02807 | 0.01061 | 0.00260 |
| 12 | content+CF hybrid | True | 0.04539 | 0.01762 | 0.00451 |
| 12 | neighborhood CF | False | 0.02938 | 0.00784 | 0.00263 |
| 12 | neighborhood CF | True | 0.02899 | 0.01104 | 0.00277 |
| 24 | MF | False | 0.04716 | 0.01374 | 0.00215 |
| 24 | MF | True | 0.06926 | 0.02259 | 0.00337 |
| 24 | article popularity | n/a | 0.05588 | 0.01659 | 0.00260 |
| 24 | content | False | 0.02151 | 0.00697 | 0.00098 |
| 24 | content | True | 0.03640 | 0.01392 | 0.00173 |
| 24 | content+CF hybrid | True | 0.06336 | 0.02443 | 0.00324 |
| 24 | neighborhood CF | False | 0.04578 | 0.01300 | 0.00213 |
| 24 | neighborhood CF | True | 0.04447 | 0.01735 | 0.00223 |

**Headline questions:**
- **(a) Does MF beat neighborhood CF (Exp B)?** MF **beats** it at k=12
  (0.05169 vs 0.02938). Latent factors do help over raw co-occurrence.
- **(b) Does MF beat or match content-based (Exp 4)?** MF **beats** content
  (0.05169 vs 0.02807).
- **(c) Does MF beat article popularity?** MF **beats** popularity
  (0.05169 vs 0.03142).

## Triple hybrid: content + CF + MF

Tuned on the internal validation window (last 28 days of
pre-cutoff events, 14,842 customers; real test labels never touched),
grid [0.0, 0.5, 1.0] per weight.

**Tuned weights: content w1=1.0, CF w2=0.5, MF w3=1.0**
(internal hit@12 = 0.06616). Triple hit@12 on test = **0.06284**
vs Exp 4 two-signal content+CF = 0.04539 (Δ=+0.01745).

## Verdict — content features vs latent factors as sparsity escapes

**Latent factors (MF) are the strongest single article-level escape from sparsity here — not content.** The honest nuance: MF's *item–item* neighbors are near-random (NN type-coherence 0.10) because 38% of articles have <5 interactions, so thin-item factors sit near the prior — the sparse-factor-recovery problem is real and visible. But MF's *customer×item* **ranking** is strong anyway: the evaluable customers have dense purchase histories, so their customer factors are well estimated even when individual item factors are noisy. Sparse-factor recovery cripples item-similarity (neighbors), not recommendation for warm customers. This is where content keeps a real edge despite losing on aggregate hit-rate: content needs **zero** interaction density (an article's attributes exist the moment it does), so it is the more **robust** escape for brand-new / thin / cold-start items and gives interpretable structure for the explanation layer. MF wins aggregate recommendation on warm customers; content wins robustness and cold-start. Both beat the raw co-occurrence of Exp B — two different, valid escapes from the sparsity that sank neighborhood CF. The triple-hybrid tuning keeps a **large MF weight (w3=1.0)** (content w1=1.0, CF w2=0.5): MF largely **subsumes** the neighborhood-CF signal (same collaborative information, better generalized), while content stays as a distinct signal. Triple hit@12 0.06284 vs Exp 4 content+CF 0.04539 (Δ=+0.01745).
