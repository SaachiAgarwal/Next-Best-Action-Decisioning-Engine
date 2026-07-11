"""Phase 2 runner: LinUCB contextual bandit — replay, baselines, audit log, report.

Run with:  python -m src.run_bandit

Offline REPLAY: the bandit learns online, but we only have logged data, so we
replay over the evaluable customers in a seeded order — select an action, look up
the reward from the label window, and update. Honest off-policy caveat: we only
observe a reward for the action the bandit picks; when the picked action is not in
the customer's label set the reward is 0 (the standard replay assumption). This
biases estimates and is revisited in Phase 5 offline policy evaluation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.eval import evaluable, metrics
from src.features import context as ctx_mod
from src.models.linucb import LinUCB
from src.models.popularity import PopularityModel, load_feature_events

ALPHA_SWEEP = [0.0, 0.5, 1.0, 2.0]
HITK = [1, 6, 12]
# Operating point for the headline run + audit log: keeps genuine exploration
# (so the uncertainty term is meaningful in the audit trail) while not over-
# exploring. The sweep shows greedy (alpha=0) maximizes single-pass offline
# reward; we report that tradeoff honestly.
CHOSEN_ALPHA = 0.5
LOG_PATH = config.PROCESSED_DIR / "bandit_decision_log.parquet"


def build_context_matrix():
    """Model-ready context with a bias term. Returns (cid->row, X, is_cold)."""
    ctx_raw = pd.read_parquet(config.PROCESSED_DIR / "customer_context.parquet", engine="pyarrow")
    mr, _scaler, _names = ctx_mod.build_model_ready(ctx_raw)
    feat_cols = [c for c in mr.columns if c != "customer_id"]
    X = mr[feat_cols].to_numpy(dtype=np.float64)
    X = np.hstack([np.ones((len(mr), 1)), X])  # bias term
    cid_to_row = {c: i for i, c in enumerate(mr["customer_id"].tolist())}
    is_cold = dict(zip(ctx_raw["customer_id"].tolist(), ctx_raw["is_cold_start"].tolist()))
    return cid_to_row, X, is_cold, X.shape[1]


def _replay(order, X, cid_to_row, label_sets, action_ids, name_by_id, top_action,
            is_cold, alpha, log=False):
    """One replay pass. Returns dict with cumulative reward + (optional) log rows."""
    bandit = LinUCB(len(action_ids), X.shape[1], alpha=alpha, action_ids=action_ids)
    cum = 0
    curve = np.empty(len(order), dtype=np.float64)
    rows = []
    for t, cid in enumerate(order):
        labels = label_sets[cid]
        x = X[cid_to_row[cid]]
        cold = bool(is_cold.get(cid, False))
        if cold:
            chosen, r_est, bonus = top_action, 0.0, 0.0   # non-contextual fallback
            ucb_scores = None
        else:
            chosen, r_est, bonus, ucb_scores = bandit.select_action(x)
        reward = 1 if chosen in labels else 0
        if not cold:
            bandit.update(chosen, x, reward)
        cum += reward
        curve[t] = cum
        if log:
            if ucb_scores is not None:
                top3_idx = np.argsort(-ucb_scores)[:3]
                top3 = " | ".join(f"{name_by_id[action_ids[i]]}:{ucb_scores[i]:.3f}"
                                  for i in top3_idx)
                ucb_val = float(ucb_scores[bandit._index[chosen]])
            else:
                top3 = f"{name_by_id[top_action]} (popularity fallback)"
                ucb_val = 0.0
            rows.append({
                "customer_id": cid, "chosen_action_id": int(chosen),
                "chosen_action_name": name_by_id[chosen],
                "reward_estimate": round(r_est, 5), "uncertainty_bonus": round(bonus, 5),
                "ucb_score": round(ucb_val, 5), "reward_observed": reward,
                "is_cold_start_fallback": cold, "top3_actions_with_scores": top3,
            })
    return {"bandit": bandit, "cum": cum, "curve": curve, "avg": cum / len(order),
            "log": rows}


def main():
    feature_events = load_feature_events()
    cid_to_row, X, is_cold, d = build_context_matrix()
    evaluable_ids, label_sets = evaluable.get_evaluable()
    n = len(evaluable_ids)
    print(f"Contextual bandit replay: {n:,} evaluable customers, context d={d} (incl. bias)")

    actions = pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
    action_ids = actions["action_id"].tolist()
    name_by_id = dict(zip(actions["action_id"], actions["product_type_name"]))

    pop_model = PopularityModel().fit(feature_events)     # leakage-safe (feature side)
    top_action = pop_model.ranked_actions[0]

    # Static personalized anchor: Exp 3 hybrid top-1 action per customer.
    exp3_top = _exp3_top_actions(feature_events, evaluable_ids)

    # Seeded replay order.
    rng = np.random.default_rng(config.SEED)
    order = list(evaluable_ids)
    order.sort()                        # deterministic base order
    rng.shuffle(order)

    # --- Main replay (chosen alpha) with logging + baselines -----------------
    main_run = _replay(order, X, cid_to_row, label_sets, action_ids, name_by_id,
                       top_action, is_cold, CHOSEN_ALPHA, log=True)
    bandit = main_run["bandit"]

    # Baselines over the same order.
    rng_b = np.random.default_rng(config.SEED)
    rnd_cum = pop_cum = ex3_cum = 0
    rnd_curve = np.empty(n); pop_curve = np.empty(n); ex3_curve = np.empty(n)
    for t, cid in enumerate(order):
        labels = label_sets[cid]
        a_rand = action_ids[int(rng_b.integers(len(action_ids)))]
        rnd_cum += 1 if a_rand in labels else 0
        pop_cum += 1 if top_action in labels else 0
        ex3_cum += 1 if exp3_top.get(cid, -1) in labels else 0
        rnd_curve[t], pop_curve[t], ex3_curve[t] = rnd_cum, pop_cum, ex3_cum

    oracle_cum = n  # every evaluable customer bought >=1 action, so oracle = 1 each step
    results = {
        "bandit_avg": main_run["avg"], "bandit_cum": main_run["cum"],
        "random_avg": rnd_cum / n, "popularity_avg": pop_cum / n, "exp3_avg": ex3_cum / n,
        "oracle_avg": 1.0, "regret": oracle_cum - main_run["cum"],
        "curves": {"bandit": main_run["curve"], "random": rnd_curve,
                   "popularity": pop_curve, "exp3": ex3_curve},
        "n": n, "d": d,
    }

    # --- hit_rate@k with the final learned bandit ----------------------------
    hitk = _hit_at_k(bandit, X, cid_to_row, label_sets, evaluable_ids, action_ids,
                     pop_model, is_cold, top_action)
    results["hitk"] = hitk

    # --- Alpha sweep ---------------------------------------------------------
    sweep = {}
    for a in ALPHA_SWEEP:
        run = _replay(order, X, cid_to_row, label_sets, action_ids, name_by_id,
                      top_action, is_cold, a, log=False)
        sweep[a] = run["avg"]
    results["sweep"] = sweep
    results["chosen_alpha"] = CHOSEN_ALPHA
    results["greedy_avg"] = sweep.get(0.0, main_run["avg"])

    # --- Save audit log ------------------------------------------------------
    log_df = pd.DataFrame(main_run["log"])
    log_df["customer_id"] = log_df["customer_id"].astype("string")
    log_df.to_parquet(LOG_PATH, engine="pyarrow")

    _print_and_report(results, log_df, name_by_id, top_action)
    print("\nDONE.")


def _exp3_top_actions(feature_events, evaluable_ids):
    """Each evaluable customer's top-1 action from the Exp 3 hybrid (static anchor)."""
    from src.models.hybrid import HybridModel
    wpath = config.PROCESSED_DIR / "hybrid_weights_exp3.json"
    if wpath.exists():
        w = json.loads(wpath.read_text())
        a, b, g = w["alpha"], w["beta"], w["gamma"]
    else:
        a, b, g = config.ALPHA_EXP3, config.BETA_EXP3, config.GAMMA_EXP3
    hybrid = HybridModel().fit(feature_events, reference_date=config.CUTOFF_DATE)
    recs = hybrid.recommend_all(evaluable_ids, k=1, alpha=a, beta=b, gamma=g)
    return {c: (v[0] if v else -1) for c, v in recs.items()}


def _hit_at_k(bandit, X, cid_to_row, label_sets, evaluable_ids, action_ids,
              pop_model, is_cold, top_action):
    """hit_rate@k for the final bandit vs popularity vs random, k in HITK."""
    band_recs, pop_recs, rand_recs = {}, {}, {}
    pop_topk = pop_model.ranked_actions
    rng = np.random.default_rng(config.SEED)
    for cid in evaluable_ids:
        if is_cold.get(cid, False):
            band_recs[cid] = pop_topk[:max(HITK)]
        else:
            # Rank by the learned reward estimate (exploitation), not UCB — this
            # is the model's recommendation, judged on what it learned.
            band_recs[cid] = bandit.top_k_exploit(X[cid_to_row[cid]], max(HITK))
        pop_recs[cid] = pop_topk[:max(HITK)]
        rand_recs[cid] = [action_ids[i] for i in rng.choice(len(action_ids), max(HITK), replace=False)]
    out = {}
    for k in HITK:
        out[k] = {
            "bandit": metrics.hit_rate_at_k(band_recs, label_sets, k),
            "popularity": metrics.hit_rate_at_k(pop_recs, label_sets, k),
            "random": metrics.hit_rate_at_k(rand_recs, label_sets, k),
        }
    return out


def _checkpoints(curve, n):
    return {f"{int(100*f)}%": int(curve[min(int(f * n) - 1, n - 1)])
            for f in (0.25, 0.5, 0.75, 1.0)}


def _print_and_report(results, log_df, name_by_id, top_action):
    n = results["n"]
    print("\n" + "=" * 60)
    print("BANDIT vs BASELINES (single-action reward = hit on chosen action)")
    print("=" * 60)
    print(f"  {'policy':<22} {'avg reward':>10} {'cum reward':>11}")
    print(f"  {'LinUCB (contextual)':<22} {results['bandit_avg']:>10.4f} {results['bandit_cum']:>11,}")
    print(f"  {'popularity':<22} {results['popularity_avg']:>10.4f} {int(results['popularity_avg']*n):>11,}")
    print(f"  {'Exp3 hybrid (static)':<22} {results['exp3_avg']:>10.4f} {int(results['exp3_avg']*n):>11,}")
    print(f"  {'random':<22} {results['random_avg']:>10.4f} {int(results['random_avg']*n):>11,}")
    print(f"  {'oracle (upper bound)':<22} {results['oracle_avg']:>10.4f} {n:>11,}")
    print(f"  final cumulative regret vs oracle: {results['regret']:,}")

    print("\n  hit_rate@k (final bandit vs baselines):")
    print(f"  {'k':>3} | {'bandit':>7} | {'popularity':>10} | {'random':>7}")
    for k in HITK:
        h = results["hitk"][k]
        print(f"  {k:>3} | {h['bandit']:>7.4f} | {h['popularity']:>10.4f} | {h['random']:>7.4f}")

    print("\n  alpha sweep (avg reward):")
    for a, v in results["sweep"].items():
        print(f"    alpha={a:<4} -> {v:.4f}")

    _write_report(results, log_df, name_by_id, top_action)


def _write_report(results, log_df, name_by_id, top_action):
    path = config.REPORTS_DIR / "phase2_bandit.md"
    n = results["n"]
    cps = {name: _checkpoints(curve, n) for name, curve in results["curves"].items()}

    def curve_row(name):
        c = cps[name]
        return f"| {name} | {c['25%']:,} | {c['50%']:,} | {c['75%']:,} | {c['100%']:,} |"

    sweep_rows = "\n".join(f"| {a} | {v:.4f} |" for a, v in results["sweep"].items())
    best_alpha = max(results["sweep"], key=results["sweep"].get)

    hk = results["hitk"]
    hitk_rows = "\n".join(
        f"| {k} | {hk[k]['bandit']:.4f} | {hk[k]['popularity']:.4f} | {hk[k]['random']:.4f} |"
        for k in HITK)

    # Example audit decisions: a couple of hits and a couple of misses.
    ex = pd.concat([log_df[log_df["reward_observed"] == 1].head(2),
                    log_df[log_df["reward_observed"] == 0].head(2)])
    ex_rows = "\n".join(
        f"| {r.customer_id[:12]}… | {r.chosen_action_name} | {r.reward_estimate:+.3f} "
        f"| {r.uncertainty_bonus:.3f} | {r.ucb_score:+.3f} | {r.reward_observed} | {r.top3_actions_with_scores} |"
        for r in ex.itertuples())

    band = results["bandit_avg"]; pop = results["popularity_avg"]
    greedy = results["greedy_avg"]; delta = greedy - pop
    verdict = (
        f"**No — the contextual bandit does not beat popularity at product-type**, and "
        f"that is the expected, honest result. At the chosen operating point "
        f"(α={results['chosen_alpha']}) the bandit's single-pick reward is "
        f"**{band:.4f}** vs popularity **{pop:.4f}**; exploration costs reward in a "
        f"single offline replay pass. Its **greedy** (α=0) reward — the learned reward "
        f"model with no exploration cost — is **{greedy:.4f}**, essentially matching "
        f"popularity ({pop:.4f}, Δ={delta:+.4f}). This is consistent with every prior "
        f"experiment: at the concentrated 128-action product-type level, popularity is a "
        f"very high bar with little headroom.\n\n"
        f"The bandit's contribution is therefore **not** raw single-pick lift but the "
        f"decision *process*: it **adapts** (learns per-context reward weights online) "
        f"and every choice is **auditable and deterministic** — a reward estimate plus an "
        f"explicit uncertainty bonus (see the audit log). This is the substrate for the "
        f"divergent-customer story from Exp 3 (personalization pays off where the crowd "
        f"is wrong) and for the constraints/arbitration layer next: a logged, inspectable "
        f"decision is what a regulated NBA engine actually needs, over a marginal "
        f"hit-rate gain.")

    content = f"""# Phase 2 — LinUCB Contextual Bandit (NBA core)

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
breadth — plus a bias term, d={results['d']}). `θ_a` are the learned reward
weights for action `a`; the confidence term uses `A_a⁻¹`, which shrinks (less
exploration) as action `a` accrues observations in the direction of `x`.

## Replay methodology & honest limitation

The bandit learns online, but we only have logged data, so we **replay** over the
{n:,} core evaluable customers in a seeded order: build context → select action →
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
| LinUCB (contextual) | {results['bandit_avg']:.4f} |
| popularity | {results['popularity_avg']:.4f} |
| Exp 3 hybrid (static personalized) | {results['exp3_avg']:.4f} |
| random | {results['random_avg']:.4f} |
| oracle (upper bound) | {results['oracle_avg']:.4f} |

**Cumulative reward** at replay checkpoints (of {n:,} decisions):

| policy | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
{curve_row('bandit')}
{curve_row('popularity')}
{curve_row('exp3')}
{curve_row('random')}

Final cumulative **regret vs oracle** = {results['regret']:,}
(oracle = always pick an action the customer actually bought).

**hit_rate@k** (bandit returns its top-k actions by p_a — comparable to the
recommender numbers):

| k | bandit | popularity | random |
|---|---|---|---|
{hitk_rows}

## Exploration sensitivity (α sweep)

| α | avg reward |
|---|---|
{sweep_rows}

α = 0 is pure exploitation; higher α explores more. Offline single-pass reward is
**maximized at α={best_alpha}** (greedy) because exploration cost isn't recovered
in one pass. We run the headline + audit log at **α = {results['chosen_alpha']}** — a
genuine bandit that still explores (so the uncertainty term is meaningful in the
audit trail) — while reporting the greedy number transparently. In a live online
deployment a small positive α is preferred to keep learning; config default
`BANDIT_ALPHA = {config.BANDIT_ALPHA}` is the exploratory textbook setting.

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
{ex_rows}

This is the Responsible-AI showcase: `bandit_decision_log.parquet` has **one row
per decision** with the full reasoning, so any recommendation is auditable after
the fact.

## Honest verdict

{verdict}
"""
    path.write_text(content)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
