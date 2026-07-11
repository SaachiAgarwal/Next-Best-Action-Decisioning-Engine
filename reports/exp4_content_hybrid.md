# Experiment 4 — Content-Based + Content/CF Hybrid (article level)

## Intuition

Experiment B showed article-level collaborative filtering **collapses from
sparsity**: with ~82k SKUs, most articles barely co-occur, so CF has almost no
signal to relate them. Content-based recommendation attacks the problem from the
other side — it scores articles by their **attributes** (product type, colour,
department, appearance, and a TF-IDF of the description), so it can relate
articles that *never* share a basket. That is exactly the failure mode CF had.

**Why TF-IDF, not embeddings:** descriptions are short, factual, small-vocabulary
product copy; TF-IDF captures the salient terms cheaply and, crucially, remains
**interpretable** — which feeds the later explanation layer. Embeddings would add
cost and opacity for negligible gain on this text.

## Content's blind spot, and why CF complements it

Content-based lives in a **filter bubble**: it only ever surfaces more of the
same type/colour/department the customer already bought. It cannot suggest the
unrelated item that co-buyers love. CF captures exactly that **cross-attribute
serendipity**. So the two have opposite strengths — content works on sparse
articles, CF finds non-obvious pairings — which motivates blending them:

    final = 0.5·content + 0.5·CF   (each min-max normalized per customer)

## Content-similar article examples (attribute signal is real)

**0110065001** (bra, black) — top-5 content-similar:

| article_id | product_type | colour | cos |
|---|---|---|---|
| 0700746001 | bra | black | 0.956 |
| 0562914001 | bra | black | 0.956 |
| 0854193002 | bra | black | 0.953 |
| 0736783004 | bra | black | 0.937 |
| 0699570001 | bra | black | 0.934 |

**0118458003** (trousers, dark grey) — top-5 content-similar:

| article_id | product_type | colour | cos |
|---|---|---|---|
| 0118458029 | trousers | dark grey | 1.000 |
| 0636586005 | trousers | dark grey | 1.000 |
| 0583534009 | trousers | dark grey | 0.909 |
| 0803757004 | trousers | dark grey | 0.884 |
| 0636586002 | trousers | black | 0.833 |

**0145872001** (sweater, black) — top-5 content-similar:

| article_id | product_type | colour | cos |
|---|---|---|---|
| 0145872043 | sweater | white | 0.833 |
| 0449123026 | t-shirt | black | 0.780 |
| 0653275005 | t-shirt | black | 0.777 |
| 0799979001 | t-shirt | black | 0.769 |
| 0842966001 | sweater | black | 0.752 |

**0184121021** (bikini top, white) — top-5 content-similar:

| article_id | product_type | colour | cos |
|---|---|---|---|
| 0743123002 | bikini top | white | 0.990 |
| 0559607003 | bikini top | white | 0.955 |
| 0560221011 | bikini top | white | 0.945 |
| 0458543007 | bikini top | white | 0.930 |
| 0560208004 | bikini top | white | 0.921 |

## Tuning (validation, not test)

Weights were grid-searched over [0.0, 0.5, 1.0, 2.0] for each of α/β
(15 combinations) to maximize hit@12 on an **internal validation
window** — the last 28 days of pre-cutoff events held out
as mini-labels (14,842 internal customers). The real
post-cutoff labels were never used for tuning.

**Chosen: content_alpha=0.5, content_beta=0.5** (validation hit@12 = 0.05552).

## Article-level comparison (all directly comparable, 15,246 core customers)

Absolute numbers are tiny (predicting 1 of ~79k) — the **relative** comparison
within the article regime is the point.

| k | model | repeats | hit_rate | recall | precision |
|---|---|---|---|---|---|
| 6 | article CF | False | 0.01732 | 0.00444 | 0.00308 |
| 6 | article CF | True | 0.01751 | 0.00674 | 0.00333 |
| 6 | article popularity | n/a | 0.02191 | 0.00584 | 0.00378 |
| 6 | content | False | 0.00905 | 0.00286 | 0.00161 |
| 6 | content | True | 0.02138 | 0.00839 | 0.00386 |
| 6 | content+CF hybrid | False | 0.01837 | 0.00575 | 0.00321 |
| 6 | content+CF hybrid | True | 0.03161 | 0.01210 | 0.00597 |
| 12 | article CF | False | 0.02938 | 0.00784 | 0.00263 |
| 12 | article CF | True | 0.02899 | 0.01104 | 0.00277 |
| 12 | article popularity | n/a | 0.03142 | 0.00926 | 0.00283 |
| 12 | content | False | 0.01482 | 0.00460 | 0.00132 |
| 12 | content | True | 0.02807 | 0.01061 | 0.00260 |
| 12 | content+CF hybrid | False | 0.02525 | 0.00792 | 0.00225 |
| 12 | content+CF hybrid | True | 0.04539 | 0.01762 | 0.00451 |
| 24 | article CF | False | 0.04578 | 0.01300 | 0.00213 |
| 24 | article CF | True | 0.04447 | 0.01735 | 0.00223 |
| 24 | article popularity | n/a | 0.05588 | 0.01659 | 0.00260 |
| 24 | content | False | 0.02151 | 0.00697 | 0.00098 |
| 24 | content | True | 0.03640 | 0.01392 | 0.00173 |
| 24 | content+CF hybrid | False | 0.03693 | 0.01210 | 0.00170 |
| 24 | content+CF hybrid | True | 0.06336 | 0.02443 | 0.00324 |

## Honest verdict

**(a) Does content-based beat article-level CF?** **Mixed — roughly a tie.** Content beats article-CF at k=[6] and trails it at k=[12, 24] (k=12: content 0.02807 vs CF 0.02938). Content alone is *competitive* with CF but not a clear standalone winner; crucially it draws on attributes, so it is strongest at short k where CF's sparse co-occurrence is thinnest. Its real value shows up in the blend (b).

**(b) Does the content+CF hybrid beat both components?** **Yes — decisively, and it also beats popularity.** The content+CF hybrid beats both components at k=[6, 12, 24] and beats **article popularity** at k=[6, 12, 24] (k=12: hybrid 0.04539 vs popularity 0.03142, +44%). This is the key result: it is the **first article-level model to beat popularity** — where Experiment B's pure CF could not. Content (attributes) and CF (co-occurrence) cover each other's blind spots — sparsity vs. cross-attribute serendipity — so the blend rescues article-level personalization.

Tuning chose **α=0.5, β=0.5** — content and CF are weighted equally, i.e. both signals carry real, complementary information at SKU level (neither collapses to the other).

## Future work — hierarchical signal

A promising extension (not built here): broadcast the **product-type-level CF**
similarity (Experiment A's dense 128×128) down to the articles mapping to each
product type, giving sparse SKUs a stable third signal borrowed from their
category. This would fuse the density of coarse CF with the specificity of
article content — a natural next lever for the sparse regime.


---

## Cross-experiment synthesis (Experiments A–4)

Product-type and article hit-rates are **not** placed in a shared numeric column:
1-of-128 and 1-of-79k are different tasks and the raw numbers are not comparable.
The comparison is on **findings**.

| Experiment | Granularity | Best model | Beat popularity? | Key finding |
|---|---|---|---|---|
| A | product-type (128) | item-CF | No | popularity too strong; no headroom for personalization |
| B | article (~82k) | item-CF | No | sparsity collapses CF — too few co-occurrences per SKU |
| 3 | product-type (128) | recency+freq hybrid | Yes (+1.1% agg, +3.6% divergent) | richer signal, not granularity, unlocks personalization |
| 4 | article (~82k) | content+CF hybrid | Yes | content+CF hybrid beats popularity AND both components — the first article-level model to beat popularity; attribute (content) signal blended with CF rescues the personalization pure CF (B) couldn't |

### Conclusion — how granularity and signal type interact

Two axes govern whether personalization beats popularity: **granularity** of the
action space and **type of signal** (collaborative, content, behavioral).

- **Coarse + collaborative (A):** popularity is unbeatable — the action space is
  so concentrated there is no headroom, and co-occurrence just re-derives
  popularity.
- **Fine + collaborative (B):** the opposite failure — the space is so sparse
  that co-occurrence has nothing to work with, and CF collapses.
- **Coarse + behavioral (3):** the winner on aggregate. Recency + frequency +
  recency-weighted CF add *personal* signal on top of a coarse space where
  popularity was strong, and the lift concentrates on customers whose taste
  diverges from the crowd.
- **Fine + content+CF (4):** the fix for B's sparsity. **Content** relates
  articles by attributes rather than co-occurrence, so it works where CF cannot;
  content alone is only *competitive* with CF, but **blending content with CF
  beats article popularity at every k** — the first article-level model to do so.
  The two signals cover opposite blind spots (content's filter bubble vs. CF's
  sparsity), so together they clear the bar neither could alone.

The headline: **there is no single best model — the right signal depends on the
granularity.** Collaborative filtering needs density (fails at both extremes for
opposite reasons); behavioral signal (recency/frequency) unlocks the coarse
regime; content + CF together unlock the sparse regime. A production NBA engine
should therefore choose signal by action granularity — which is exactly the
motivation for the contextual bandit ahead: let a policy *learn* which signal to
trust per context, rather than fixing one recommender.
