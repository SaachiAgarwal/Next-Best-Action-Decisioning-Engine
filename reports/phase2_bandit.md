# Phase 2 — LinUCB Contextual Bandit (NBA core)

## What a contextual bandit is

A contextual bandit repeatedly (1) sees a **context** (here, the customer's
context vector), (2) **chooses an action** (one of 128 product-types), and (3)
observes a **reward** (did they buy that product-type in the label window). It
must balance **exploiting** actions it believes are good against **exploring**
uncertain ones to learn. Unlike a static recommender, it *learns from the
outcomes of its own decisions*.

## Why LinUCB (not Thompson Sampling)

LinUCB keeps one linear reward model per action and scores each action as

    p_a = θ_a · x  +  α · sqrt(xᵀ A_a⁻¹ x)

a **reward estimate** plus a **deterministic uncertainty bonus**. It picks the
highest. Because the exploration term is deterministic (not a random posterior
draw as in Thompson Sampling), **every decision is reproducible and inspectable** —
you can always say exactly why an action was chosen (estimate vs. uncertainty).
That auditable decision trail is the Responsible-AI requirement, so we choose
LinUCB over Thompson Sampling here.

## How context enters

Each action's linear model conditions on the customer context vector `x`
(the model-ready encoding of `customer_context.parquet` — RFM, attributes,
breadth — plus a bias term, d=25). `θ_a` are the learned reward
weights for action `a`; the confidence term uses `A_a⁻¹`, which shrinks (less
exploration) as action `a` accrues observations in the direction of `x`.

## Replay methodology & honest limitation

The bandit learns online, but we only have logged data, so we **replay** over the
15,246 core evaluable customers in a seeded order: build context → select action →
look up reward from the label window → update. **Off-policy caveat (stated
plainly):** we only see a reward for the action the bandit picks; when that action
is not in the customer's label set, reward = 0 (standard replay assumption). This
can bias the estimate (the bandit is scored partly on data its own policy shaped)
and is revisited with proper offline policy evaluation (IPS/replay with variance)
in Phase 5.

## Results — bandit vs baselines

**Average single-action reward** (fraction of customers for whom the *one* chosen
action was actually purchased):

| policy | avg reward |
|---|---|
| LinUCB (contextual) | 0.2047 |
| popularity | 0.3471 |
| Exp 3 hybrid (static personalized) | 0.3491 |
| random | 0.0216 |
| oracle (upper bound) | 1.0000 |

**Cumulative reward** at replay checkpoints (of 15,246 decisions):

| policy | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| bandit | 258 | 1,043 | 2,092 | 3,121 |
| popularity | 1,341 | 2,685 | 4,008 | 5,292 |
| exp3 | 1,343 | 2,684 | 4,023 | 5,322 |
| random | 78 | 164 | 247 | 329 |

Final cumulative **regret vs oracle** = 12,125
(oracle = always pick an action the customer actually bought).

**hit_rate@k** (bandit returns its top-k actions by p_a — comparable to the
recommender numbers):

| k | bandit | popularity | random |
|---|---|---|---|
| 1 | 0.3333 | 0.3471 | 0.0205 |
| 6 | 0.6668 | 0.7371 | 0.1242 |
| 12 | 0.7733 | 0.8267 | 0.2290 |

## Exploration sensitivity (α sweep)

| α | avg reward |
|---|---|
| 0.0 | 0.3439 |
| 0.5 | 0.2047 |
| 1.0 | 0.1387 |
| 2.0 | 0.0530 |

α = 0 is pure exploitation; higher α explores more. Offline single-pass reward is
**maximized at α=0.0** (greedy) because exploration cost isn't recovered
in one pass. We run the headline + audit log at **α = 0.5** — a
genuine bandit that still explores (so the uncertainty term is meaningful in the
audit trail) — while reporting the greedy number transparently. In a live online
deployment a small positive α is preferred to keep learning; config default
`BANDIT_ALPHA = 1.0` is the exploratory textbook setting.

## Cold-start handling

Customers with `is_cold_start = True` (no pre-cutoff history, context is all
defaults) cannot be conditioned on meaningfully, so the bandit **falls back to a
non-contextual policy** (global most-popular action) and logs the decision with
`is_cold_start_fallback = True`. Default-zero context is never fed as if it were
real signal. (The core evaluable set is all warm by construction, so the replay
here contains no cold-start rows; the fallback path is covered by tests.)

## Example decisions from the audit log (the "why" trail)

Each row reconstructs exactly why the bandit chose what it did — reward estimate +
uncertainty bonus = UCB score — plus the top-3 contenders:

| customer | chosen | reward_est | uncertainty | ucb | reward | top-3 (action:ucb) |
|---|---|---|---|---|---|---|
| 9cf9ade87f11… | trousers | +0.000 | 1.449 | +1.449 | 1 | trousers:1.449 | soft toys:1.449 | bra extender:1.449 |
| ad0f9c554126… | trousers | +0.433 | 1.788 | +2.221 | 1 | trousers:2.221 | soft toys:2.070 | bra extender:2.070 |
| e743fd2c77b9… | dress | +0.000 | 1.664 | +1.664 | 0 | pyjama jumpsuit/playsuit:1.664 | soft toys:1.664 | bra extender:1.664 |
| 5e83bc94c7dc… | trousers | +0.484 | 1.092 | +1.576 | 0 | trousers:1.576 | soft toys:1.320 | bra extender:1.320 |

This is the Responsible-AI showcase: `bandit_decision_log.parquet` has **one row
per decision** with the full reasoning, so any recommendation is auditable after
the fact.

## Honest verdict

**No — the contextual bandit does not beat popularity at product-type**, and that is the expected, honest result. At the chosen operating point (α=0.5) the bandit's single-pick reward is **0.2047** vs popularity **0.3471**; exploration costs reward in a single offline replay pass. Its **greedy** (α=0) reward — the learned reward model with no exploration cost — is **0.3439**, essentially matching popularity (0.3471, Δ=-0.0032). This is consistent with every prior experiment: at the concentrated 128-action product-type level, popularity is a very high bar with little headroom.

The bandit's contribution is therefore **not** raw single-pick lift but the decision *process*: it **adapts** (learns per-context reward weights online) and every choice is **auditable and deterministic** — a reward estimate plus an explicit uncertainty bonus (see the audit log). This is the substrate for the divergent-customer story from Exp 3 (personalization pays off where the crowd is wrong) and for the constraints/arbitration layer next: a logged, inspectable decision is what a regulated NBA engine actually needs, over a marginal hit-rate gain.
