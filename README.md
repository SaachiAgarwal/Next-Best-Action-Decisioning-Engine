# Next-Best-Action Decisioning Engine

A Next-Best-Action (NBA) recommendation engine built on the H&M dataset that decides the single best action to present to each customer.

## Problem

H&M's product catalog is large and constantly changing, and customers struggle to surface the items that are actually relevant to them. Browsing thousands of articles is a poor experience and leaves most of the catalog undiscovered. The goal of this project is to recommend the *next best action* for each customer — the most relevant product (or product group) to put in front of them next. Rather than returning a long ranked list, the engine arbitrates candidate actions against scoring and business rules to commit to one recommendation per customer.

## Architecture

The engine is organized as four layers:

1. **Candidate Generation** — narrow the full catalog down to a manageable set of plausible actions per customer.
2. **Scoring / Ranking** — score each candidate action for the customer using a predictive model.
3. **Business Constraints** — apply eligibility, inventory, and policy rules that filter or adjust scored candidates.
4. **Decisioning / Arbitration** — select the single next best action from the constrained, scored candidates.

## Project Structure

```
Next-Best-Action-Decisioning-Engine/
├── data/
│   ├── raw/                  # source H&M data (not committed)
│   └── processed/            # derived data (not committed)
├── src/
│   ├── __init__.py
│   ├── config.py             # paths + config constants
│   └── data/
│       ├── __init__.py
│       ├── load.py           # read raw source files
│       ├── clean.py          # cleaning / type coercion
│       ├── action_space.py   # define the action space
│       ├── event_log.py      # per-customer event log
│       └── splits.py         # time-based train/val/test splits
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
├── tests/
│   ├── __init__.py
│   └── test_data_layer.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Status

- [x] Week 1 Day 1: repo setup
- [ ] Data layer (load, clean, action space, event log, splits)
- [ ] Candidate generation
- [ ] Scoring / ranking model
- [ ] Business constraints
- [ ] Decisioning / arbitration
- [ ] Evaluation

## Data

The H&M dataset is **not** committed to this repository. Download the H&M
CSV files and place them under `data/raw/`. Both `data/raw/` and
`data/processed/` are git-ignored.
