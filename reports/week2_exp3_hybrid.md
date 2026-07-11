# Week 2 — Experiment 3: Recency + Frequency Weighted Hybrid

## The idea

Experiments A/B showed that *granularity* alone did not let personalization beat
popularity (product-type: popularity too strong; article: too sparse). Experiment 3
tests the other lever — **richer signal at the same 128-action product-type level**
— by blending four ingredients, all from feature-side events only:

1. **Log-damped frequency** `log(1 + count)` — habit matters, but a 10x buyer is
   not 10x more loyal; damping also stops volume outliers (the 1,237-purchase
   customer) from dominating.
2. **Recency** `w = 0.5^(Δ / 30d)` (true half-life) on the
   customer's most recent purchase of each action — a purchase one half-life
   (30 days) before the reference counts exactly half.
3. **Recency-weighted CF** — item-CF cosine similarity, with each of the
   customer's purchases weighted by its recency.
4. **Popularity prior** — the floor that guarantees the blend can match the
   baseline and handles cold-start (gamma dominates when history is empty).

`personal = log(1+count) * recency`; `final = α·personal + β·cf + γ·popularity`,
each component min-max normalized per customer before weighting. **Repeats are
included** (product-type repeat lift was +0.319 — fashion customers re-buy
categories, so excluding prior purchases would throw away the strongest signal).

## Tuning (validation, not test)

Weights were grid-searched to maximize **hit@12 on an internal validation window**
— the last 28 days of *pre-cutoff* events held out as
mini-labels, training on everything before. The real post-cutoff labels were never
used for tuning. Grid: [0.0, 0.25, 0.5, 1.0, 2.0] for each of α/β/γ
(124 combinations), 15,552
internal validation customers.

**Chosen weights: α=0.5, β=0.25, γ=2.0** (validation hit@12 = 0.8833).

## Aggregate results (real labels, 15,246 core customers)

| k | model | hit_rate | recall | precision |
|---|---|---|---|---|
| 6 | hybrid | 0.7480 | 0.4870 | 0.2114 |
| 6 | item_cf(rep) | 0.6684 | 0.4086 | 0.1858 |
| 6 | popularity | 0.7371 | 0.4843 | 0.2124 |
| 12 | hybrid | 0.8339 | 0.6164 | 0.1376 |
| 12 | item_cf(rep) | 0.8095 | 0.5927 | 0.1350 |
| 12 | popularity | 0.8267 | 0.6020 | 0.1342 |
| 24 | hybrid | 0.9608 | 0.8844 | 0.0995 |
| 24 | item_cf(rep) | 0.9505 | 0.8693 | 0.0983 |
| 24 | popularity | 0.9559 | 0.8752 | 0.0985 |

Hybrid vs popularity:
  - k=6: hybrid 0.7480 vs popularity 0.7371 (Δ=+0.0109, +1.5%)
  - k=12: hybrid 0.8339 vs popularity 0.8267 (Δ=+0.0071, +0.9%)
  - k=24: hybrid 0.9608 vs popularity 0.9559 (Δ=+0.0049, +0.5%)

## Divergent-customer slice (the honest test)

"Divergent" customers are the **bottom quartile by cosine similarity** between
their feature-side action mix and the global popularity distribution
(cosine ≤ 0.517) — the 3,812 evaluable customers whose taste is
least like the crowd. This is where personalization should help if it helps
anywhere. Slice saved to `divergent_customers_exp3.parquet` (deterministic).

| k | model | hit_rate | recall | precision |
|---|---|---|---|---|
| 6 | hybrid | 0.6556 | 0.4494 | 0.1610 |
| 6 | popularity | 0.6327 | 0.4355 | 0.1583 |
| 12 | hybrid | 0.7791 | 0.5945 | 0.1091 |
| 12 | popularity | 0.7678 | 0.5791 | 0.1065 |
| 24 | hybrid | 0.9504 | 0.8846 | 0.0804 |
| 24 | popularity | 0.9420 | 0.8715 | 0.0793 |

Hybrid vs popularity on the divergent slice:
  - k=6: hybrid 0.6556 vs popularity 0.6327 (Δ=+0.0228, +3.6%)
  - k=12: hybrid 0.7791 vs popularity 0.7678 (Δ=+0.0113, +1.5%)
  - k=24: hybrid 0.9504 vs popularity 0.9420 (Δ=+0.0084, +0.9%)

## Honest verdict

**The hybrid beats popularity on aggregate** at k=12 (Δ=+0.0071), and at every k tested — richer signal (recency + log-frequency + recency-weighted CF), not finer granularity, is what let personalization clear the bar. The lift is **larger on the divergent slice** (k=12 Δ=+0.0113), which is exactly where it should be: personalization helps most for customers whose taste diverges from the crowd, while staple-buyers — whom popularity already serves — dilute the aggregate gain.

## Tie-back to the granularity experiment

This arm isolates *signal* from *granularity*: same 128 actions as Experiment A,
but with recency + frequency + recency-weighted CF instead of raw co-occurrence.
Comparing the aggregate and divergent-slice results tells us whether the ceiling
product-type popularity imposed is about the coarse action space itself or about
the poverty of a pure co-occurrence signal — and, crucially, whether
personalization's value is **concentrated in the customers the crowd fails**,
which the aggregate number hides.
