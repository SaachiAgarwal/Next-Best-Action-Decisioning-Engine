"""Run the popularity baseline and write reports/week2_baseline.md.

Run with:  python -m src.models.run_popularity_baseline

Steps:
  1. Fit popularity on feature-side events only (no label leakage).
  2. Build the evaluable core set + label sets.
  3. Recommend the global top-24 to every evaluable customer.
  4. Evaluate hit_rate/recall/precision @ [6, 12, 24].
  5. Compare the two popularity variants; pick canonical.
  6. Print the results table + top-12 popular actions by name; write the report.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.eval import evaluable, metrics
from src.models.popularity import PopularityModel, load_feature_events

KS = [6, 12, 24]


def _action_names():
    actions = pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
    return dict(zip(actions["action_id"], actions["product_type_name"]))


def _named(action_ids, names):
    return [f"{names.get(a, '?')} (id={a})" for a in action_ids]


def _write_report(results, model, names, n_eval, variants_agree):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "week2_baseline.md"

    hr12 = float(results.loc[results["k"] == 12, "hit_rate"].iloc[0])

    results_rows = "\n".join(
        f"| {int(r.k)} | {r.hit_rate:.4f} | {r.recall:.4f} | {r.precision:.4f} |"
        for r in results.itertuples()
    )
    top12 = model.ranked_actions[:12]
    top12_rows = "\n".join(
        f"| {rank} | {a} | {names.get(a, '?')} |"
        for rank, a in enumerate(top12, start=1)
    )

    variant_note = (
        "The purchase-count and distinct-customer rankings produce the **same** "
        "top-12 order, so the choice is immaterial here."
        if variants_agree else
        "The purchase-count and distinct-customer rankings differ in the top list; "
        "we use **purchase-count** as canonical (it reflects total demand volume, "
        "which is what an untargeted recommendation should surface)."
    )

    content = f"""# Week 2 — Popularity Baseline & Evaluation Harness

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
(`features_events.parquet`, all `t_dat < {config.CUTOFF_DATE}`). It never reads
the label window. `PopularityModel.fit` asserts the events it receives end before
the cutoff, so label-window data cannot leak into the ranking. Evaluation labels
come exclusively from the held-out label window.

## Metric definitions (plain language)

All metrics are computed per customer over the **{n_eval:,} core evaluable
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
{results_rows}

## Interpretation (honest)

Popularity is a **strong** baseline in this setting, and it's important to say so
plainly. The action space is small (**{len(model.ranked_actions)} actions**) and
highly concentrated — a handful of product types (trousers, dress, sweater,
t-shirt…) dominate purchases — so simply recommending the most common actions
catches a large share of real purchases.

**hit-rate@12 = {hr12:.4f}** ({hr12 * 100:.1f}%): recommending the same 12
popular actions to everyone lands at least one real purchase for roughly
{hr12 * 100:.0f}% of evaluable customers. **This is the bar personalized models
must clear.** Because popularity is this strong, a personalized model earns its
keep only if it meaningfully lifts hit-rate/recall above these numbers — beating
it by a rounding error would not be worth the added complexity.

{variant_note}

## Canonical top-12 popular actions

| rank | action_id | product_type_name |
|---|---|---|
{top12_rows}
"""
    path.write_text(content)
    print(f"\nWrote {path}")


def main():
    # 1. Fit popularity on feature-side events only.
    feature_events = load_feature_events()
    model = PopularityModel().fit(feature_events)

    # 2. Evaluable core set + labels.
    evaluable_ids, label_sets = evaluable.get_evaluable()
    n_eval = len(evaluable_ids)
    print(f"Evaluable core customers: {n_eval:,}")

    # 3. Recommend global top-24 to every evaluable customer.
    recommendations = model.recommend_all(evaluable_ids, k=24)

    # 4. Evaluate.
    results = metrics.evaluate(recommendations, label_sets, ks=KS)
    print("\n" + "=" * 52)
    print("POPULARITY BASELINE RESULTS (core evaluable set)")
    print("=" * 52)
    print(f"{'k':>4} | {'hit_rate':>9} | {'recall':>8} | {'precision':>9}")
    for r in results.itertuples():
        print(f"{int(r.k):>4} | {r.hit_rate:>9.4f} | {r.recall:>8.4f} | {r.precision:>9.4f}")

    # 5. Compare popularity variants.
    variants_agree = model.ranked_actions[:12] == model.ranked_by_customers[:12]
    print("\nPopularity variants (top-12):")
    print(f"  purchase-count vs distinct-customer top-12 identical: {variants_agree}")
    if not variants_agree:
        print("  -> using purchase-count as canonical (total demand volume).")

    # 6. Top-12 by name.
    names = _action_names()
    print("\nCanonical top-12 popular actions:")
    for rank, a in enumerate(model.ranked_actions[:12], start=1):
        print(f"  {rank:>2}. {names.get(a, '?'):<20} (id={a})")

    _write_report(results, model, names, n_eval, variants_agree)
    print("\nDONE.")


if __name__ == "__main__":
    main()
