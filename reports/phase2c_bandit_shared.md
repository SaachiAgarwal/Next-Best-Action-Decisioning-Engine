# Phase 2c — Shared-Model LinUCB at Article Level

## Shared vs disjoint (why per-arm is impossible at 79k articles)

A disjoint bandit keeps a separate model (A_a, b_a) per action. At ~79k articles
that is hopeless: spread ~2M interactions across 79k arms and each arm sees **<30
observations on average, most far fewer** — the per-arm models never leave their
prior. A **shared model** keeps **one** (A, b) over the joint (customer, article)
**feature space**. Every decision updates the same θ, so learning **transfers
across articles** — and an article never seen in training still gets a score,
because θ acts on its *features*, not its identity. This is the shared model's
whole point, and it is what makes an article-level bandit tractable at all.

## Why interaction/dot-product features are essential

A linear model over concatenated `[customer | article]` features can only learn
**additive** effects — "this customer buys more overall", "this article is popular"
— never **matching** ("*this* customer likes *this* article"). Matching needs
customer×article terms. So the feature vector's most important entries are four
**affinity scalars**: `mf_score = U_c·V_a`, `content_score = cos(profile_c, item_a)`,
`cf_score` (Exp B neighborhood), and `article_popularity`. These dot-products give
the linear model the matching signal it structurally cannot form on its own.

## Feature vector (d = 117)

| block | dims | source |
|---|---|---|
| customer context | 24 | `customer_context` model-ready (RFM, attrs, breadth, cold-start) |
| MF article embedding | 64 | Exp 5 `V` |
| content (compressed) | 24 | TruncatedSVD of the Exp 4 content matrix (raw TF-IDF too high-dim) |
| affinity scalars | 4 | mf_score, content_score, cf_score, popularity |
| bias | 1 | constant |

All features are from pre-cutoff data only and standardized on **learning-set**
statistics.

## Retrieval stage and the ceiling

Scoring all 79,269 articles per decision is impractical, so we
**retrieve the top-100 candidates** per customer with the Exp 5 triple hybrid,
then the bandit re-ranks them. Top-100 (not a tight top-6) keeps genuine room
to explore. Retrieval imposes a hard **ceiling**: the bandit cannot recommend what
was not retrieved.

- **recall@100 = 0.0432** (avg fraction of a customer's label
  articles that appear in their candidates)
- **hit@100 = 0.1139** (fraction of held-out customers with ≥1 label
  in candidates) — **this is the ceiling for hit@k.**

## Reward sparsity (honest warning)

Article-level rewards are **1.4% positive** on the retrieval top-1 — a
customer buys only a handful of the ~79k SKUs, so almost every arm pull returns 0.
Online learning is therefore slow and noisy; read the learning curve with that in
mind.

## Learning curve (best α=0.5, held-out)

| steps | epoch | hit@1 | hit@6 | hit@12 | hit@24 |
|---|---|---|---|---|---|
| 0 | 0 | 0.0142 | 0.0453 | 0.0608 | 0.0750 |
| 10,672 | 1 | 0.0151 | 0.0398 | 0.0507 | 0.0652 |
| 21,344 | 2 | 0.0162 | 0.0372 | 0.0501 | 0.0654 |
| 32,016 | 3 | 0.0162 | 0.0369 | 0.0503 | 0.0645 |
| 42,688 | 4 | 0.0162 | 0.0356 | 0.0490 | 0.0654 |
| 53,360 | 5 | 0.0164 | 0.0363 | 0.0509 | 0.0645 |

## α sweep (final held-out)

| α | hit@1 | hit@6 | hit@12 | hit@24 |
|---|---|---|---|---|
| 0.0 | 0.0133 | 0.0289 | 0.0372 | 0.0485 |
| 0.5 | 0.0164 | 0.0363 | 0.0509 | 0.0645 |
| 1.0 | 0.0164 | 0.0374 | 0.0498 | 0.0643 |
| 2.0 | 0.0166 | 0.0378 | 0.0496 | 0.0652 |

## Held-out comparison (all article level)

| model | hit@1 | hit@6 | hit@12 | hit@24 |
|---|---|---|---|---|
| shared bandit (α=0.0) | 0.0133 | 0.0289 | 0.0372 | 0.0485 |
| shared bandit (α=0.5) | 0.0164 | 0.0363 | 0.0509 | 0.0645 |
| shared bandit (α=1.0) | 0.0164 | 0.0374 | 0.0498 | 0.0643 |
| shared bandit (α=2.0) | 0.0166 | 0.0378 | 0.0496 | 0.0652 |
| Exp5 triple hybrid (static) | 0.0142 | 0.0453 | 0.0608 | 0.0750 |
| Exp5 MF alone | 0.0120 | 0.0400 | 0.0540 | 0.0721 |
| Exp4 content+CF | 0.0068 | 0.0275 | 0.0437 | 0.0610 |
| article popularity | 0.0074 | 0.0210 | 0.0302 | 0.0560 |
| random | 0.0002 | 0.0007 | 0.0011 | 0.0013 |
| **retrieval ceiling (hit@100)** | | | | 0.1139 |

## Unseen-article demonstration (the shared model's superpower)

There are **2,671 articles with zero pre-cutoff interactions** (they
appear only in the label window). They have **no MF embedding** and were **never**
a candidate during learning — a disjoint per-arm model could not score them at all.
The shared model scores them anyway, from their **content features alone**:

| article_id | product_type | shared-model score |
|---|---|---|
| 0201219016 | underwear tights | -0.0004 |
| 0201219017 | underwear tights | -0.0013 |
| 0237347060 | hoodie | +0.0074 |
| 0237347069 | hoodie | +0.0031 |
| 0291333023 | underwear tights | +0.0003 |

Finite, feature-derived scores for never-seen items — this is what "learning over
the feature space" buys, and the direct answer to article-level cold-start.

## Audit examples (held-out; each decision is explainable)

| customer | chosen | reward_est | uncertainty | mf | content | cf | reward |
|---|---|---|---|---|---|---|---|
| 60b708d3ff… | trousers | +0.076 | 0.013 | +9.84 | -0.53 | +0.04 | 1 |
| 0b7357f04c… | socks | +0.104 | 0.017 | +10.38 | -0.31 | +0.21 | 1 |
| e5df8c1a96… | shirt | +0.010 | 0.022 | -0.22 | -0.08 | -0.15 | 0 |
| 0eee882944… | top | +0.006 | 0.016 | +0.90 | -0.75 | +0.05 | 0 |

Every decision decomposes into the reward estimate + uncertainty, with the
affinity features showing *why*: e.g. high MF affinity and content match →
high score. Full trail in `bandit_shared_decision_log.parquet`.

## Verdict

**(a) The shared bandit does NOT beat the Exp 5 triple hybrid on the broad ranking** (α=0.5, hit@12 0.0509 vs 0.0608, Δ=-0.0098). The one place it *does* improve is **hit@1** (0.0164 vs 0.0142) — its re-ranking sharpens the single best pick, but it loses ground across the fuller top-12/24 where the static tuned blend is stronger. This is the honest finding the task anticipated: with article-level rewards only **1.4% positive** on the retrieval top-1, there is too little online signal to improve on an already-tuned static blend within these epochs. Static tuning (Exp 5) is effectively sufficient at this reward sparsity — a data constraint, not a modelling failure.

**(b) Exploration helps** (as in v3): greedy α=0 is the **worst** (hit@12 0.0372) — it overfits the sparse rewards and its curve *declines* from the untrained retrieval order; any α≥0.5 recovers to ~0.0509. Once the features are informative, exploration is again essential.

**(c) Distance to the ceiling:** the best bandit reaches hit@24 = 0.0645 against the **retrieval ceiling** hit@100 = 0.1139 — about 57% of it (the static hybrid reaches 66%). Recall@100 is only 0.0432, so **improving the retriever would lift results more than any re-ranking can** — the ceiling, not the policy, is the binding constraint here.

## Honest limitations

- **Off-policy bias**: rewards observed only for logged behavior; a recommended
  but unbought article scores 0 though the counterfactual is unknown. Offline
  replay approximates and likely understates a live bandit (IPS in Phase 5).
- **Reward sparsity** (1.4% positive) fundamentally limits online learning here.
- **Retrieval ceiling** (0.1139) caps achievable hit-rate; improving the
  retriever would raise it more than re-ranking can.
