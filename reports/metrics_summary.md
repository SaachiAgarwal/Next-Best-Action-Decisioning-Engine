# Consolidated Metrics Summary

> **⚠️ Article-level and product-type numbers are NOT comparable.** They are
> different prediction tasks — 1-of-79,269 articles vs. 1-of-128 product types —
> so raw metrics differ by an order of magnitude for structural reasons, not
> quality. **Compare only within a regime (within a table).**

> **†** The bandit rows (shared bandit, LinUCB v3) were evaluated on their
> **held-out set (4,574 customers)**, a *different* customer set from the 15,246
> core used for every other row. They are shown for reference and are **not
> strictly comparable** to the static models here.

## How to read this

- **hit-rate@k** — did we get *at least one* of the customer's purchases into the
  top-k? (breadth of "got something right")
- **recall@k** — what *fraction* of everything the customer bought did we cover?
- **precision@k** — what fraction of our k recommendations actually landed?
- **coverage / Gini / diversity / fairness / cold-article** — beyond-accuracy
  diagnostics (Phase 3a) that accuracy metrics are structurally blind to:
  how much catalog is reachable, how concentrated on the head, how varied each
  list is, how evenly customers are served, and whether cold items can be scored.

All accuracy metrics use the **best repeat setting per model** (repeats=True for
the CF/content/MF/hybrid models — fashion is repurchase-heavy; popularity has no
repeat notion). Evaluated on the same **15,246 core evaluable customers**
(article-level) via the Day 1 harness keyed on `article_id`.

## Article-level models (15,246 core customers)

| model | repeats | hit@6 | hit@12 | hit@24 | recall@6 | recall@12 | recall@24 | prec@6 | prec@12 | prec@24 | cov@12 | mean pop rank | head% | Gini | diversity | fair spread | cold-art% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| article popularity | n/a | 0.0219 | 0.0314 | 0.0559 | 0.0058 | 0.0093 | 0.0166 | 0.0038 | 0.0028 | 0.0026 | 0.0% | 6 | 100.0% | 0.9998 | 0.6869 | 0.0315 | 0.0% |
| neighborhood CF (Exp B) | True | 0.0175 | 0.0290 | 0.0445 | 0.0067 | 0.0110 | 0.0174 | 0.0033 | 0.0028 | 0.0022 | 77.8% | 33,228 | 19.8% | 0.5525 | 0.8057 | 0.0400 | 0.0% |
| content (Exp 4) | True | 0.0214 | 0.0281 | 0.0364 | 0.0084 | 0.0106 | 0.0139 | 0.0039 | 0.0026 | 0.0017 | 41.2% | 19,526 | 38.6% | 0.8487 | 0.1776 | 0.0563 | 100.0% |
| content+CF hybrid (Exp 4) | True | 0.0316 | 0.0454 | 0.0634 | 0.0121 | 0.0176 | 0.0244 | 0.0060 | 0.0045 | 0.0032 | — | — | — | — | — | — | — |
| MF (Exp 5) | True | 0.0392 | 0.0517 | 0.0693 | 0.0128 | 0.0170 | 0.0226 | 0.0073 | 0.0050 | 0.0034 | 0.8% | 170 | 100.0% | 0.9976 | 0.6275 | 0.0519 | 0.0% |
| triple hybrid (Exp 5, production) | True | 0.0472 | 0.0628 | 0.0776 | 0.0171 | 0.0229 | 0.0287 | 0.0088 | 0.0061 | 0.0039 | 40.6% | 10,015 | 69.0% | 0.8857 | 0.4529 | 0.0688 | 100.0% |
| shared bandit (Phase 2c, α=0.5) † | n/a | 0.0363 | 0.0509 | 0.0645 | — | — | — | — | — | — | — | — | — | — | — | — | — |

*Diagnostics columns are blank for the content+CF hybrid (it was not part of the
Phase 3a diagnostics run) and for the held-out bandit.*

## Product-type models (15,246 core customers)

Coverage is **not reported** for product-type models: with only 128 actions,
catalog coverage is near-total by construction and would be a vacuous number.

| model | repeats | hit@6 | hit@12 | hit@24 | recall@6 | recall@12 | recall@24 | prec@6 | prec@12 | prec@24 |
|---|---|---|---|---|---|---|---|---|---|---|
| popularity (Exp A) | n/a | 0.7371 | 0.8267 | 0.9559 | 0.4843 | 0.6020 | 0.8752 | 0.2124 | 0.1342 | 0.0985 |
| item-CF (Exp A) | True | 0.6684 | 0.8095 | 0.9505 | 0.4086 | 0.5927 | 0.8693 | 0.1858 | 0.1350 | 0.0983 |
| recency+freq hybrid (Exp 3, production) | incl. | 0.7480 | 0.8339 | 0.9608 | 0.4870 | 0.6164 | 0.8844 | 0.2114 | 0.1376 | 0.0995 |
| LinUCB v3 (best α=1.0) † | n/a | 0.6732 | 0.7339 | — | — | — | — | — | — | — |

## Synthesis — where recall tells a different story than hit-rate

The two accuracy views can rank models differently, and the beyond-accuracy
diagnostics rank them in the *opposite* order to accuracy. Most telling:
**MF** posts a strong hit@12 (0.0517) but its **recall@12 is only
0.0170** — it catches *a* purchase for many customers but a *thin slice* of
each customer's basket, because it concentrates on a tiny popular head (0.82%
catalog coverage, 100% head share). The triple hybrid's recall@12 (0.0229)
is higher for similar-order hit-rate — it spreads across more of the basket.
Neighborhood CF, weakest on hit-rate, has recall@12 0.0110 while covering
78% of the catalog — the healthiest breadth.

Restating the central finding from Phase 3a: **accuracy and coverage rank the
models in opposite orders.** The accuracy leaders (MF, triple hybrid) are the
coverage/diversity laggards; the coverage/diversity leader (neighborhood CF) is
the accuracy laggard. A single accuracy number — hit-rate especially — hides this
entirely, which is exactly why the full table (recall + precision + diagnostics)
exists.
