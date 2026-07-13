# Phase 3a — Recommender Diagnostics (beyond accuracy)

## Why hit-rate alone is insufficient

Hit-rate asks only "did we get one purchase right?" It is blind to four failure
modes that matter to the business and to customers:

1. **Popularity bias** — re-serving the head; are we more head-biased than demand?
2. **Catalog coverage** — how much inventory is ever recommended (dead stock, no discovery)?
3. **Intra-list diversity** — is each list varied, or 12 near-identical items?
4. **Cold-start & fairness** — who and what does the model fail on, hidden inside the average?

Production model under test: the **Exp 5 triple hybrid** (content + CF + MF,
weights 1.0/0.5/1.0), benchmarked against its components. All
recommendations use **repeats=True** (Exp 5's best setting). Evaluated on the
15,246 core customers.

## Summary table (model × diagnostic)

| model | hit@12 | coverage% | mean pop rank | top-10% share | Gini | intra-list dissim | distinct types | cold-cust hit@12 | seg spread | cold-article scorable |
|---|---|---|---|---|---|---|---|---|---|---|
| triple hybrid | 0.0628 | 40.59% | 10015 | 69.0% | 0.886 | 0.453 | 3.5 | 0.0273 | 0.0688 | 100% |
| MF | 0.0517 | 0.82% | 170 | 100.0% | 0.998 | 0.627 | 5.5 | 0.0273 | 0.0519 | 0% |
| content | 0.0281 | 41.16% | 19526 | 38.6% | 0.849 | 0.178 | 1.4 | 0.0273 | 0.0563 | 100% |
| neighborhood CF | 0.0290 | 77.80% | 33228 | 19.8% | 0.553 | 0.806 | 7.5 | 0.0273 | 0.0400 | 0% |
| popularity | 0.0314 | 0.02% | 6 | 100.0% | 1.000 | 0.687 | 7.0 | 0.0273 | 0.0315 | 0% |

*Definitions:* **coverage%** = share of the 79,269-article catalog appearing
in ≥1 top-12 list. **mean pop rank** = average popularity rank of recommended
articles (1 = most popular; higher = deeper into the tail). **top-10% share** = %
of recommendations from the 10% most popular articles. **Gini** = concentration of
recommendation frequency (0 even → 1 all-on-one). **intra-list dissim** = mean
pairwise content distance within a list (higher = more varied). **distinct types**
= avg distinct product types per 12-item list. **seg spread** = worst hit@12 gap
across customer segments. **cold-article scorable** = % of zero-interaction
articles the model can score at all.

## Catalog coverage by k

| model | coverage@6 | coverage@12 | coverage@24 |
|---|---|---|---|
| triple hybrid | 26.75% | 40.59% | 54.53% |
| MF | 0.49% | 0.82% | 1.34% |
| content | 29.58% | 41.16% | 53.40% |
| neighborhood CF | 56.29% | 77.80% | 90.15% |
| popularity | 0.01% | 0.02% | 0.03% |

## Popularity bias

Demand baseline: the mean popularity rank of what customers **actually bought** is
**35279**. A model is *over*-biased if its recommended mean rank is
**below** this (it serves the head harder than real demand does).

## Cold-start diagnostics

- **Cold-start customers** (1,649 with no pre-cutoff history): every
  model falls back to popularity, giving hit@12 = **0.0273**. No
  personalization is possible without history.
- **Cold-start articles** (2,671 with zero pre-cutoff
  interactions): the empirical case for content. **MF / CF / popularity can score
  0%** of them (they require interactions); **content can score 100%** (attributes
  exist the moment the article does). None are recommended by any model here (they
  are outside every model's fitted item space), but only content *could* surface
  them — the article-level cold-start answer.

## Fairness across customer segments (production model)

**age_band** (spread 0.0436):

| segment | n | hit@12 |
|---|---|---|
| 26-35 | 4,203 | 0.0707 |
| 36-45 | 1,642 | 0.0664 |
| 46-55 | 2,825 | 0.0648 |
| 56+ | 1,529 | 0.0595 |
| <=25 | 4,973 | 0.0555 |
| unknown | 74 | 0.0270 |

**club_member_status** (spread 0.0688):

| segment | n | hit@12 |
|---|---|---|
| PRE-CREATE | 160 | 0.0688 |
| ACTIVE | 15,060 | 0.0628 |
| unknown | 25 | 0.0400 |
| LEFT CLUB | 1 | 0.0000 |

**frequency_quartile** (spread 0.0344):

| segment | n | hit@12 |
|---|---|---|
| top | 3,811 | 0.0792 |
| high | 3,812 | 0.0666 |
| mid | 3,811 | 0.0606 |
| low | 3,812 | 0.0449 |

**recency_quartile** (spread 0.0564):

| segment | n | hit@12 |
|---|---|---|
| active | 3,811 | 0.0994 |
| warm | 3,812 | 0.0606 |
| cooling | 3,811 | 0.0483 |
| lapsed | 3,812 | 0.0430 |

## Honest synthesis

**Accuracy and coverage are in direct tension, and the leaderboard inverts when you stop looking only at hit-rate.** The starkest case is **MF**: it is the 2nd-most-accurate model (hit@12 0.0517) yet reaches a catastrophic **0.82% catalog coverage** — it recommends essentially only the head (100% of its picks are top-10% articles, Gini 0.998). MF beats neighborhood CF on accuracy by +0.0227 hit@12 but covers **0.82%** of the catalog vs CF's **77.8%** — a ~95× coverage gap. Hit-rate alone would have crowned MF and never revealed that it leaves ~99% of inventory dead.

The **production triple hybrid** (triple hybrid, hit@12 0.0628) is better — 41% coverage because its content component spreads it — but still gains only +0.0348 hit@12 over content for materially more popularity bias.

**Everything is more head-biased than real demand.** Customers actually buy deep into the tail — the mean popularity rank of true purchases is **35279** (of 79,269). Every model recommends far shallower: MF mean rank 170, triple hybrid 10015, content 19526, **neighborhood CF 33228** — CF is the *only* model whose recommendation depth is close to demand, and it is also the healthiest on coverage (78%) and diversity — but the weakest on accuracy. The accuracy winners are the demand-alignment losers.

**Diversity has its own trap:** **content** has broad coverage (41%) but the **least varied lists** (intra-list dissimilarity 0.178, only 1.4 distinct product types per 12 — it stacks near-identical same-type items). neighborhood CF is the most diverse.

**Fairness:** the least-uniform model across customer segments is **triple hybrid** (hit@12 spread 0.0688) — the more personalized the model, the wider the gap between well- and poorly-served segments.

**Cold-start:** every model collapses to popularity for history-less customers (hit@12 0.0273); and only **content** can even *score* the 2,671 zero-interaction articles (100% vs 0% for MF/CF/popularity) — the quantified case for content features.

**This motivates the next build: a diversity/coverage re-ranking layer** that trades a controlled slice of accuracy for catalog coverage, tail exposure, and intra-list diversity — tradeoffs hit-rate alone would never have surfaced.
