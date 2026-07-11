# Phase 1 — Customer Context / Feature Layer

## What this is

The **customer context vector** is the per-customer state the contextual bandit
conditions on: one row per customer (`customer_context.parquet`), computed
strictly from pre-cutoff (feature-side) events. It answers **"who is this
customer"** — how recently/often/valuably they buy, how broad their taste, and
their attributes.

## Design note — context vs. per-action scores (deliberate)

The context is **aggregate customer-state**, intentionally distinct from the
recommender's **per-action affinity scores**. The scorer answers "which action is
good for this customer"; the context answers "who is this customer". Keeping them
separate is a deliberate architecture decision: it lets the bandit's context
**complement** the candidate scorer rather than duplicate its signal. The scorer
proposes and ranks actions; the context describes the person the policy is
deciding for — together they give the bandit both *what's available* and *who
it's for*, without redundancy.

## Feature list

**RFM (customer-level, from feature-side events):**
- `recency_days` — days from the customer's last pre-cutoff purchase to the cutoff.
- `frequency` — total pre-cutoff transaction count.
- `monetary_total` / `monetary_avg` — sum / mean of `price` (price is normalized
  to [0, 1], so this is a *relative engagement* signal, not currency).

**Attributes (from customers.parquet; nulls -> "unknown", never dropped):**
- `age` + `age_band` (`<=25 / 26-35 / 36-45 / 46-55 / 56+ / unknown`).
- `club_member_status`, `fashion_news_frequency`.

**Behavioral breadth:**
- `distinct_actions` — number of distinct product-type actions bought (variety).
- `tenure_days` — days from first pre-cutoff purchase to the cutoff.
- `avg_repurchase_gap_days` — mean gap between consecutive purchases
  (**0 by convention for single-purchase customers**; equals
  (last - first) / (frequency - 1)).
- `dominant_action_id` + `dominant_action_share` — the customer's most-purchased
  action and its share of their purchases (cold-start sentinel `-1`).

Plus `is_cold_start` — a flag distinguishing history-less customers.

## Distributions (customers with history, n=97,696)

| feature | mean | median | p90 | min | max |
|---|---|---|---|---|---|
| recency_days | 226.72 | 145.00 | 587.00 | 1.00 | 706.00 |
| frequency | 22.73 | 9.00 | 58.00 | 1.00 | 1199.00 |
| monetary_total | 0.63 | 0.24 | 1.58 | 0.00 | 40.17 |
| monetary_avg | 0.03 | 0.03 | 0.04 | 0.00 | 0.42 |
| distinct_actions | 7.54 | 5.00 | 18.00 | 1.00 | 54.00 |
| tenure_days | 489.83 | 563.00 | 696.00 | 1.00 | 706.00 |
| avg_repurchase_gap_days | 18.98 | 8.67 | 46.57 | 0.00 | 687.00 |

**age_band**

| band | customers | share |
|---|---|---|
| <=25 | 30,115 | 30.1% |
| 26-35 | 25,844 | 25.8% |
| 46-55 | 18,696 | 18.7% |
| 36-45 | 12,428 | 12.4% |
| 56+ | 11,743 | 11.7% |
| unknown | 1,174 | 1.2% |

**club_member_status**

| status | customers | share |
|---|---|---|
| ACTIVE | 92,733 | 92.7% |
| PRE-CREATE | 6,781 | 6.8% |
| unknown | 448 | 0.4% |
| LEFT CLUB | 38 | 0.0% |

**fashion_news_frequency**

| frequency | customers | share |
|---|---|---|
| NONE | 64,097 | 64.1% |
| Regularly | 34,663 | 34.7% |
| unknown | 1,191 | 1.2% |
| Monthly | 49 | 0.0% |

## Cold-start handling

Of **100,000** customers in the base, **2,304**
(2.3%) have no pre-cutoff history. They are **retained**,
not dropped, with explicit safe defaults: `frequency=0`, `monetary_*=0`,
`distinct_actions=0`, `dominant_action_id=-1`, `avg_repurchase_gap_days=0`,
`recency_days`/`tenure_days` left NaN (flagged), and `is_cold_start=True`. The
bandit must be able to decide for these customers (it falls back to popularity via
the recommender), so the context layer represents them explicitly rather than
omitting them.

## Model-ready encoding

`build_model_ready()` deterministically regenerates a numeric matrix from the raw
table: numeric features median-imputed (cold-start recency/tenure/age) then
standardized, categoricals one-hot encoded. Result: **100,000
rows x 24 feature columns**, **no NaNs**. The raw table is
saved as `customer_context.parquet`; the encoded matrix is regenerated on demand
(deterministic), so raw stays human-readable for the report and constraints layer.

## Leakage guarantee

Every feature derives only from events with `t_dat < 2020-08-26`;
`context.build` asserts the input's max date is strictly before the cutoff. No
post-cutoff (label-window) information enters the context vector.
