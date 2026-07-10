# Week 2 — Item-to-Item Collaborative Filtering

## What this is

Item-to-item CF is the first **personalized** model. The intuition is
co-purchase: actions bought by the same customers are related, so we recommend
actions similar to what a customer has already bought. We build a 128×128
action-similarity matrix from feature-side co-purchases and, per customer, score
each candidate action by its similarity to the customer's own purchase history.

**Binary interactions.** For each customer we take the *set* of distinct actions
bought pre-cutoff (bought / didn't), not purchase counts. At product-type
granularity, buying "trousers" many times mostly reflects category volume rather
than proportionally stronger affinity, so binary co-buy is the cleaner signal.

## Normalization — why it prevents collapse into popularity

Raw co-occurrence C[i,j] (customers who bought both i and j) is dominated by
globally-popular actions: almost everyone buys trousers, so trousers co-occurs
heavily with *everything*. If we ranked neighbors by raw co-occurrence, every
action's top neighbor would just be the popular actions — item-CF would collapse
back into the popularity baseline and add nothing.

We fix this with **cosine similarity** on the binary customer-action vectors:

> sim[i,j] = C[i,j] / ( sqrt(C[i,i]) · sqrt(C[j,j]) )

Dividing by each action's own popularity (its diagonal) corrects for how common
each action is, so similarity measures *relative* co-purchase, not raw volume.
Self-similarity is zeroed. The result is real structure — see below.

## Learned structure (sanity check)

These neighbors are popularity-corrected; note they are topically coherent
(swimwear together, underwear together), **not** just "the popular ones":

**bikini top** — top-5 similar:

| action | cosine sim |
|---|---|
| swimwear bottom | 0.889 |
| bra | 0.484 |
| vest top | 0.477 |
| dress | 0.468 |
| shorts | 0.467 |

**swimwear bottom** — top-5 similar:

| action | cosine sim |
|---|---|
| bikini top | 0.889 |
| bra | 0.480 |
| shorts | 0.479 |
| vest top | 0.479 |
| dress | 0.473 |

**trousers** — top-5 similar:

| action | cosine sim |
|---|---|
| sweater | 0.675 |
| t-shirt | 0.644 |
| dress | 0.634 |
| top | 0.626 |
| blouse | 0.602 |

**bra** — top-5 similar:

| action | cosine sim |
|---|---|
| underwear bottom | 0.632 |
| vest top | 0.523 |
| sweater | 0.519 |
| t-shirt | 0.517 |
| top | 0.509 |

## Comparison vs popularity (same 15,246 core evaluable customers)

`item_cf (no repeats)` excludes actions the customer already bought pre-cutoff;
`item_cf (repeats)` allows them (fashion is repurchase-heavy, so this matters).

| k | model | hit_rate | recall | precision |
|---|---|---|---|---|
| 6 | item_cf (no repeats) | 0.3493 | 0.1717 | 0.0736 |
| 6 | item_cf (repeats) | 0.6684 | 0.4086 | 0.1858 |
| 6 | popularity | 0.7371 | 0.4843 | 0.2124 |
| 12 | item_cf (no repeats) | 0.4649 | 0.2594 | 0.0548 |
| 12 | item_cf (repeats) | 0.8095 | 0.5927 | 0.1350 |
| 12 | popularity | 0.8267 | 0.6020 | 0.1342 |
| 24 | item_cf (no repeats) | 0.5492 | 0.3380 | 0.0356 |
| 24 | item_cf (repeats) | 0.9505 | 0.8693 | 0.0983 |
| 24 | popularity | 0.9559 | 0.8752 | 0.0985 |

## Verdict (honest)

- **k=6:** item-CF does **not** beat popularity — best item-CF (`item_cf (repeats)`) 0.6684 vs popularity 0.7371 (Δ=-0.0687, -9.3%).
- **k=12:** item-CF does **not** beat popularity — best item-CF (`item_cf (repeats)`) 0.8095 vs popularity 0.8267 (Δ=-0.0172, -2.1%).
- **k=24:** item-CF does **not** beat popularity — best item-CF (`item_cf (repeats)`) 0.9505 vs popularity 0.9559 (Δ=-0.0054, -0.6%).

Item-CF does not beat popularity here; the repeats variant converges toward it as k grows (within ~0.6% at k=24) but never exceeds it. The likely cause is the **small, concentrated action space** (128 product types, heavily dominated by a few). When almost everyone's next purchase is one of a dozen popular types, there is little headroom for personalization to beat 'recommend the popular things' on coarse top-k hit-rate — the structure item-CF learns is real (see the neighbor examples) but doesn't translate into top-k gains at this granularity. This is an honest negative result: personalization likely needs a finer action space and/or richer signal (recency, sequence) to pay off, which is the motivation for later models.

## Repeats vs no-repeats

Allowing repeats (recommending actions the customer already bought pre-cutoff)
**materially changes** the result: because fashion customers re-buy the same
product types (someone who bought trousers buys trousers again), re-recommending
prior purchases is a strong signal. The `item_cf (repeats)` row is the stronger
configuration here; `no repeats` deliberately forces novelty and pays for it in
hit-rate/recall. The right default depends on the product goal — novelty/discovery
(no repeats) vs. next-purchase likelihood (repeats).

## Fitting & leakage

Item-CF is fit on `features_events.parquet` only (all `t_dat < 2020-08-26`);
`fit` asserts this. Cold-start customers with no pre-cutoff history fall back to
the popularity top-k.
