# Week 1 Data Profile — NBA Decisioning Engine

## Sampling note (read this first)

The working dataset for this project is **100,000 customers** (random sample,
**seed = 42**) together with their **complete purchase
history**. This sample was chosen so development runs quickly and reproducibly
on a local machine. The pipeline is **size-agnostic**: it runs identically on
the full H&M dataset (31,788,324 transactions) — only `SAMPLE_CUSTOMERS` in
`src/config.py` changes.

The sample covers **7.23%** of total transaction volume
(2,296,723 of 31,788,324 transactions). Because we sample whole customers and keep
their full histories, every sampled transaction, customer, and article is
internally consistent (see orphan counts below).

## Dataset shape (sampled)

| table | rows |
|---|---|
| customers | 100,000 |
| articles | 81,940 |
| transactions | 2,296,723 |

## Transactions: date & price range

- **Date range:** 2018-09-20 → 2020-09-22
- **Price range:** 0.000186 → 0.506780
  (price is **normalized to [0, 1]** in the source data — not a currency amount)

## Null summary

**transactions**

_No nulls._

**articles**

| column | null count |
|---|---|
| `detail_desc` | 316 |

**customers**

| column | null count |
|---|---|
| `FN` | 65,382 |
| `Active` | 66,292 |
| `club_member_status` | 448 |
| `fashion_news_frequency` | 1,191 |
| `age` | 1,174 |

## Referential integrity (orphans)

Counts of transaction references that point at a missing master row. Because we
sample whole customers and derive the article set from their transactions, both
should be ~0.

| orphan type | count |
|---|---|
| transaction `article_id` missing from articles | 0 |
| transaction `customer_id` missing from customers | 0 |

## Memory footprint

Transactions memory, before vs after optimization (price → float32,
sales_channel_id → category, ids kept as string):

| stage | memory (MB) |
|---|---|
| before optimization | 264.70 |
| after optimization | 237.14 |
| reduction | 27.56 (10.4%) |

Optimized footprint of each sampled table:

| table | memory (MB) |
|---|---|
| transactions | 237.14 |
| articles | 39.63 |
| customers | 19.64 |

## Column dtypes (transactions)

| column | dtype |
|---|---|
| `t_dat` | `datetime64[us]` |
| `customer_id` | `string` |
| `article_id` | `string` |
| `price` | `float32` |
| `sales_channel_id` | `category` |

## Notes

- `article_id` and `customer_id` are stored as **strings** to preserve leading
  zeros; loading them as integers would silently break every join.
- The sampled parquet files live in `data/processed/` and are **git-ignored** —
  this report is the only record of the data that survives in the repo.

<!-- day3: cleaning + action space (regenerated) -->

## Cleaning

Cleaning is explicit and logged — no rows are dropped silently. Categorical
metadata nulls are kept as an explicit `unknown` category; only genuinely
impossible transactions (non-positive price, or a date outside the known
dataset window 2018-09-20 → 2020-09-22)
are removed. Every transformation and the number of rows/values it affected:

| step | table | target | strategy | affected | note |
|---|---|---|---|---|---|
| nulls | customers | `FN` | impute | 65,382 | binary flag; missing means not flagged -> 0 (fill='0.0') |
| nulls | customers | `Active` | impute | 66,292 | binary flag; missing means not active -> 0 (fill='0.0') |
| nulls | customers | `club_member_status` | keep-as-category | 448 | categorical metadata (fill='unknown') |
| nulls | customers | `fashion_news_frequency` | keep-as-category | 1,191 | categorical metadata (fill='unknown') |
| nulls | customers | `age` | keep | 1,174 | numeric; imputation deferred to feature engineering (left as-is) |
| nulls | articles | `detail_desc` | keep-as-category | 316 | free-text description; fill placeholder (fill='unknown') |
| text | articles | `prod_name` | strip+lowercase | 81,318 | standardized categorical name text |
| text | articles | `product_type_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `product_group_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `graphical_appearance_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `colour_group_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `perceived_colour_value_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `perceived_colour_master_name` | strip+lowercase | 81,864 | standardized categorical name text |
| text | articles | `department_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `index_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `index_group_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `section_name` | strip+lowercase | 81,940 | standardized categorical name text |
| text | articles | `garment_group_name` | strip+lowercase | 81,940 | standardized categorical name text |
| filter | transactions | `price <= 0` | drop (impossible) | 0 | non-positive price |
| filter | transactions | `t_dat outside [2018-09-20, 2020-09-22]` | drop (impossible) | 0 | date outside known dataset window |
| filter | transactions | `TOTAL removed` | drop (impossible) | 0 | 2,296,723 -> 2,296,723 rows |

## Action Space

**Granularity: `product_type_name`.** Product-type granularity balances
recommendation precision against statistical learnability — it avoids
article-level sparsity (too many actions, too little signal each) while staying
more actionable than broad product groups.

- **Total actions (action-space size): 128**
- An `article_id → action` mapping (`article_action_map.parquet`) is retained so
  a recommended action can be drilled down to the specific articles behind it.
- **Long tail:** 30 of 128 actions
  have fewer than 100 purchases in the sampled data;
  these thin actions will be the hardest to learn reliably downstream.

Top 15 actions by purchase volume:

| action_id | product_type_name | article_count | total_purchases | distinct_customers |
|---|---|---|---|---|
| 109 | trousers | 8,716 | 303,986 | 53,771 |
| 31 | dress | 8,499 | 235,692 | 44,238 |
| 98 | sweater | 7,401 | 200,055 | 46,869 |
| 103 | t-shirt | 5,942 | 159,422 | 40,628 |
| 106 | top | 3,380 | 114,102 | 36,931 |
| 11 | blouse | 3,361 | 108,604 | 32,234 |
| 118 | vest top | 2,430 | 101,758 | 30,791 |
| 15 | bra | 1,975 | 97,416 | 28,435 |
| 85 | shorts | 3,000 | 82,168 | 25,937 |
| 8 | bikini top | 805 | 81,479 | 24,623 |
| 100 | swimwear bottom | 1,116 | 79,685 | 24,607 |
| 113 | underwear bottom | 2,179 | 78,064 | 24,616 |
| 88 | skirt | 2,274 | 67,381 | 23,743 |
| 84 | shirt | 2,700 | 55,916 | 23,803 |
| 60 | leggings/tights | 1,343 | 53,047 | 21,904 |
