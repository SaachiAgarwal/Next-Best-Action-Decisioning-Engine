# Experiment 6 — Customer Attributes in the Hybrid

## The design problem

Attributes describe the **customer**; the model must score **articles**. We bridge
with **segment lift**: for each demographic segment, which articles it buys
*disproportionately* vs global demand. Using **lift, not raw counts** is essential
— raw segment popularity just re-derives global popularity (every segment buys the
popular items most); lift isolates what is *distinctive*. Smoothing (K=20) shrinks
thin segments toward global so noise doesn't masquerade as signal.

## Do segments actually differ?

At **product-type** level (robust), segments show mild, sensible differences:

**age_band=<=25** — most distinctive product types (lift):

| product_type | lift |
|---|---|
| keychain | 2.34 |
| bucket hat | 2.21 |
| wireless earphone case | 2.17 |
| marker pen | 1.95 |
| tote bag | 1.82 |

**age_band=56+** — most distinctive product types (lift):

| product_type | lift |
|---|---|
| stain remover spray | 7.36 |
| towel | 2.76 |
| headband | 2.55 |
| flat shoes | 2.17 |
| straw hat | 2.01 |

**fashion_news_frequency=Regularly** — most distinctive product types (lift):

| product_type | lift |
|---|---|
| chem. cosmetics | 1.99 |
| wireless earphone case | 1.83 |
| bumbag | 1.57 |
| wood balls | 1.57 |
| eyeglasses | 1.57 |

But at **article** level — the granularity the models actually score — segment
lift is dominated by **single-purchase noise**: the highest-lift articles are
items bought exactly once by the segment (tied lift values), not a real
distinctive taste. Per-segment × per-article counts are ~0–1, so the article-level
attribute signal is weak and noisy. This already predicts a small w4.

## Arm 6a — four-signal hybrid (content + CF + MF + attributes)

**Tuned weights: content w1=1.0, CF w2=0.5, MF w3=1.0,
attributes w4=0.0** (internal validation hit@12 0.06616).

**The attribute weight tuned to w4=0** — attributes add *nothing* on top of the behavioral signals. This is a clean negative finding, consistent across the project: behavioral signal (what you bought) subsumes demographic signal (who you are). It echoes bandit v2, where a customer-only attribute context could not beat popularity — the model was starved of behavioral matching, and attributes alone did not fill the gap.

| model | hit@6 | hit@12 | hit@24 | recall@12 | prec@12 | cov@12 | mean pop rank | head% | Gini | diversity | fair spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| triple hybrid (Exp 5) | 0.0472 | 0.0628 | 0.0776 | 0.0229 | 0.0061 | 40.59% | 10,015 | 68.97% | 0.8857 | 0.4529 | 0.0688 |
| four-signal +attrs (Exp 6a) | 0.0472 | 0.0628 | 0.0776 | 0.0229 | 0.0061 | 40.59% | 10,015 | 68.97% | 0.8857 | 0.4529 | 0.0688 |

- **Accuracy:** unchanged (with w4=0.0, 6a is identical to the triple hybrid).
- **Coverage:** 40.6% vs 40.6%.
- **Fairness spread:** 0.0688 vs 0.0688 — attributes left the segment gap essentially unchanged.

## Arm 6b — attribute-based cold-start vs popularity

Cold-start population: **1,649** customers with no pre-cutoff history and a
label-window purchase. Current fallback = article popularity (the number to beat).
Weights tuned by **simulating cold-start on warm customers** (masking their
history, using attributes only, evaluating on their validation window) — tuned
**attr v1=0.0, popularity v2=0.5**. *Limitation: warm customers may
differ systematically from genuine cold-start customers, so the simulated tuning is
an approximation.*

| model | repeats | hit@6 | hit@12 | hit@24 | recall@12 | prec@12 |
|---|---|---|---|---|---|---|
| article popularity (fallback) | n/a | 0.0158 | 0.0273 | 0.0564 | 0.0106 | 0.0025 |
| attribute cold-start (Exp 6b) | n/a | 0.0158 | 0.0273 | 0.0564 | 0.0106 | 0.0025 |

- Cold-start coverage@12: attribute model **0.02%** vs blanket
  popularity **0.02%**.
- **Verdict:** attribute cold-start hit@12 0.0273 vs popularity 0.0273
  (+0.0000). Knowing "26-35, active club member" does **not** beat knowing nothing — demographic signal is too weak to personalize cold-start. The honest answer: cold-start needs a different solution (onboarding preferences, or content-based from a first click), not demographics.

## Honest synthesis

The project-wide pattern holds: **behavior dominates demographics.** For **warm**
customers, attributes add nothing (w4=0) on top of
content/CF/MF — what you *buy* is far more informative than *who you are*. For
**cold-start**, where behavior is absent and attributes are the only signal,
they still do not beat popularity
— demographic segments at this catalog's granularity are too coarse and too noisy
to personalize. This is consistent with bandit v2 (attributes couldn't beat
popularity) and quantifies *why*: article-level segment lift is single-purchase
noise, and product-type-level differences are mild. The actionable takeaway:
invest cold-start effort in **explicit preference capture** (onboarding, first-click
content signals), not demographic inference.
