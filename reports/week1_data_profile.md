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
