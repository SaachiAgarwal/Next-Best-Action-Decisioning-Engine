# Week 2 — Popularity Baseline & Evaluation Harness

## What this is and why it matters

The **popularity baseline** recommends the globally most-purchased actions to
*every* customer — no personalization. It serves two purposes:

1. **The bar.** Every personalized model must beat non-personalized popularity to
   justify its complexity. If a learned model can't clear this line, it isn't
   adding value.
2. **The cold-start fallback.** For customers with no history (no features to
   personalize on), the sensible default is exactly this global top-k.

## Leakage note

Popularity is computed from **feature-side events only**
(`features_events.parquet`, all `t_dat < 2020-08-26`). It never reads
the label window. `PopularityModel.fit` asserts the events it receives end before
the cutoff, so label-window data cannot leak into the ranking. Evaluation labels
come exclusively from the held-out label window.

## Metric definitions (plain language)

All metrics are computed per customer over the **15,246 core evaluable
customers** (customers with ≥1 label-window purchase *and* pre-cutoff history),
then averaged.

- **hit-rate@k** — did we get *at least one* action right? 1 if any of the top-k
  recommended actions was actually purchased in the label window, else 0.
- **recall@k** — of all the actions the customer actually bought, what fraction
  appear in our top-k?
- **precision@k** — of the k actions we recommended, what fraction were actually
  bought?

## Results

| k | hit_rate | recall | precision |
|---|---|---|---|
| 6 | 0.7371 | 0.4843 | 0.2124 |
| 12 | 0.8267 | 0.6020 | 0.1342 |
| 24 | 0.9559 | 0.8752 | 0.0985 |

## Interpretation (honest)

Popularity is a **strong** baseline in this setting, and it's important to say so
plainly. The action space is small (**128 actions**) and
highly concentrated — a handful of product types (trousers, dress, sweater,
t-shirt…) dominate purchases — so simply recommending the most common actions
catches a large share of real purchases.

**hit-rate@12 = 0.8267** (82.7%): recommending the same 12
popular actions to everyone lands at least one real purchase for roughly
83% of evaluable customers. **This is the bar personalized models
must clear.** Because popularity is this strong, a personalized model earns its
keep only if it meaningfully lifts hit-rate/recall above these numbers — beating
it by a rounding error would not be worth the added complexity.

The purchase-count and distinct-customer rankings differ in the top list; we use **purchase-count** as canonical (it reflects total demand volume, which is what an untargeted recommendation should surface).

## Canonical top-12 popular actions

| rank | action_id | product_type_name |
|---|---|---|
| 1 | 109 | trousers |
| 2 | 31 | dress |
| 3 | 98 | sweater |
| 4 | 103 | t-shirt |
| 5 | 106 | top |
| 6 | 11 | blouse |
| 7 | 118 | vest top |
| 8 | 15 | bra |
| 9 | 85 | shorts |
| 10 | 8 | bikini top |
| 11 | 100 | swimwear bottom |
| 12 | 113 | underwear bottom |
