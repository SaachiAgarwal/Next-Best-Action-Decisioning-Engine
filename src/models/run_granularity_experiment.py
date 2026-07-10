"""Week 2 Day 3 — article-level granularity experiment (Experiment B) + cross
comparison against product-type (Experiment A).

Run with:  python -m src.models.run_granularity_experiment

Experiment A (product-type, 128 actions) is recomputed read-only here purely so
the cross-granularity table is self-consistent; it does NOT modify any
product-type artifact. Experiment B outputs use *_article names only.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.eval import evaluable, metrics
# Experiment A (product-type) — read-only reuse.
from src.models.popularity import PopularityModel, load_feature_events
from src.models.item_cf import ItemCF
# Experiment B (article-level).
from src.models.popularity_article import ArticlePopularityModel
from src.models.item_cf_article import ArticleItemCF

KS = [6, 12, 24]


# --------------------------------------------------------------------------
# Article-level labels
# --------------------------------------------------------------------------
def build_article_labels(evaluable_ids):
    """Distinct label-window article_ids per evaluable customer. Saves parquet.

    Uses the frozen split (event_log + config.CUTOFF_DATE); does not recompute it.
    """
    el = pd.read_parquet(
        config.PROCESSED_DIR / "event_log.parquet",
        columns=["customer_id", "t_dat", "article_id"], engine="pyarrow",
    )
    cutoff = pd.Timestamp(config.CUTOFF_DATE)
    label = el[(el["t_dat"] >= cutoff) & (el["customer_id"].isin(evaluable_ids))]
    labels_article = (
        label[["customer_id", "article_id"]].drop_duplicates()
        .sort_values(["customer_id", "article_id"]).reset_index(drop=True)
    )
    labels_article["customer_id"] = labels_article["customer_id"].astype("string")
    labels_article["article_id"] = labels_article["article_id"].astype("string")
    labels_article.to_parquet(config.PROCESSED_DIR / "labels_article.parquet", engine="pyarrow")

    label_sets = {
        cid: set(grp)
        for cid, grp in labels_article.groupby("customer_id", sort=False)["article_id"]
    }
    return labels_article, label_sets


def _rows(recs, label_sets, model, repeats):
    df = metrics.evaluate(recs, label_sets, ks=KS)
    df.insert(1, "model", model)
    df.insert(2, "repeats", repeats)
    return df


def _print_table(df, title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    print(f"{'k':>3} | {'model':<22} | {'repeats':<7} | {'hit_rate':>8} | {'recall':>7} | {'precision':>9}")
    print("-" * 74)
    for r in df.itertuples():
        print(f"{int(r.k):>3} | {r.model:<22} | {str(r.repeats):<7} | "
              f"{r.hit_rate:>8.4f} | {r.recall:>7.4f} | {r.precision:>9.4f}")


def main():
    feature_events = load_feature_events()
    evaluable_ids, label_sets_A = evaluable.get_evaluable()  # product-type label sets
    n_eval = len(evaluable_ids)
    print(f"Evaluable core customers: {n_eval:,}")

    # ---------------- Experiment A (product-type), recomputed read-only -------
    pop_A = PopularityModel().fit(feature_events)
    cf_A = ItemCF(popularity_model=pop_A).fit(feature_events)
    A_pop = _rows(pop_A.recommend_all(evaluable_ids, k=24), label_sets_A, "popularity", "n/a")
    A_cf_nore = _rows(cf_A.recommend_all(evaluable_ids, k=24, include_repeats=False),
                      label_sets_A, "item_cf", "False")
    A_cf_rep = _rows(cf_A.recommend_all(evaluable_ids, k=24, include_repeats=True),
                     label_sets_A, "item_cf", "True")
    exp_A = pd.concat([A_pop, A_cf_nore, A_cf_rep], ignore_index=True)

    # ---------------- Experiment B (article-level) ----------------------------
    labels_article, label_sets_B = build_article_labels(evaluable_ids)
    # Assert identical evaluable customer set as Experiment A.
    assert set(label_sets_B.keys()) == set(evaluable_ids), \
        "article-level evaluable customer set differs from Experiment A"
    print(f"labels_article.parquet: {len(labels_article):,} (customer, article) pairs, "
          f"{labels_article['customer_id'].nunique():,} customers "
          f"(matches Experiment A: {set(label_sets_B) == set(evaluable_ids)})")

    pop_B = ArticlePopularityModel().fit(feature_events)
    cf_B = ArticleItemCF(popularity_model=pop_B).fit(feature_events)  # prints memory

    B_pop = _rows(pop_B.recommend_all(evaluable_ids, k=24), label_sets_B, "article_popularity", "n/a")
    B_cf_nore = _rows(cf_B.recommend_all(evaluable_ids, k=24, include_repeats=False),
                      label_sets_B, "article_item_cf", "False")
    B_cf_rep = _rows(cf_B.recommend_all(evaluable_ids, k=24, include_repeats=True),
                     label_sets_B, "article_item_cf", "True")
    exp_B = pd.concat([B_pop, B_cf_nore, B_cf_rep], ignore_index=True)

    _print_table(exp_B.sort_values(["k", "model", "repeats"]), "EXPERIMENT B — ARTICLE-LEVEL RESULTS")

    # ---------------- Cross-granularity comparison ----------------------------
    comparison = _cross_comparison(exp_A, exp_B)
    print("\n" + "=" * 74)
    print("CROSS-GRANULARITY COMPARISON (product-type vs article)")
    print("=" * 74)
    for line in comparison["console"]:
        print(line)

    _write_report(exp_A, exp_B, cf_B, comparison, n_eval)
    print("\nDONE.")


def _hit(df, model, repeats, k):
    row = df[(df["model"] == model) & (df["repeats"] == repeats) & (df["k"] == k)]
    return float(row["hit_rate"].iloc[0])


def _cross_comparison(exp_A, exp_B):
    """Derive the headline cross-granularity findings."""
    # Popularity hit@12.
    popA12 = _hit(exp_A, "popularity", "n/a", 12)
    popB12 = _hit(exp_B, "article_popularity", "n/a", 12)

    # Does item-CF beat popularity? (best repeat setting per level)
    def best_cf_vs_pop(df, cf_model, pop_model, k):
        pop = _hit(df, pop_model, "n/a", k)
        cf_best = max(_hit(df, cf_model, "False", k), _hit(df, cf_model, "True", k))
        which = "True" if _hit(df, cf_model, "True", k) >= _hit(df, cf_model, "False", k) else "False"
        return pop, cf_best, which, cf_best - pop

    # Repeats lift at k=6.
    A_norep6 = _hit(exp_A, "item_cf", "False", 6)
    A_rep6 = _hit(exp_A, "item_cf", "True", 6)
    B_norep6 = _hit(exp_B, "article_item_cf", "False", 6)
    B_rep6 = _hit(exp_B, "article_item_cf", "True", 6)

    console = []
    console.append(f"  popularity hit_rate@12 : product-type {popA12:.4f}  vs  article {popB12:.4f}  "
                   f"(article is {'weaker' if popB12 < popA12 else 'stronger'})")
    for k in KS:
        popA, cfA, whichA, dA = best_cf_vs_pop(exp_A, "item_cf", "popularity", k)
        popB, cfB, whichB, dB = best_cf_vs_pop(exp_B, "article_item_cf", "article_popularity", k)
        console.append(
            f"  k={k:>2}: product-type item-CF {cfA:.4f} vs pop {popA:.4f} (Δ={dA:+.4f}) | "
            f"article item-CF {cfB:.4f} vs pop {popB:.4f} (Δ={dB:+.4f}, best repeats={whichB})")
    console.append(f"  repeats lift @6: product-type {A_norep6:.4f}->{A_rep6:.4f} "
                   f"(+{A_rep6 - A_norep6:.4f}) | article {B_norep6:.4f}->{B_rep6:.4f} "
                   f"(+{B_rep6 - B_norep6:.4f})")

    return {
        "console": console,
        "popA12": popA12, "popB12": popB12,
        "A_repeats6": (A_norep6, A_rep6), "B_repeats6": (B_norep6, B_rep6),
        "beats": {
            k: {
                "A": best_cf_vs_pop(exp_A, "item_cf", "popularity", k),
                "B": best_cf_vs_pop(exp_B, "article_item_cf", "article_popularity", k),
            } for k in KS
        },
    }


def _write_report(exp_A, exp_B, cf_B, comparison, n_eval):
    path = config.REPORTS_DIR / "week2_granularity_experiment.md"
    m = cf_B.memory

    def table(df, models):
        rows = []
        for r in df.sort_values(["k", "model", "repeats"]).itertuples():
            if r.model in models:
                rows.append(f"| {int(r.k)} | {r.model} | {r.repeats} | "
                            f"{r.hit_rate:.4f} | {r.recall:.4f} | {r.precision:.4f} |")
        return "\n".join(rows)

    beats = comparison["beats"]
    beat_lines = []
    for k in KS:
        popB, cfB, whichB, dB = beats[k]["B"]
        popA, cfA, whichA, dA = beats[k]["A"]
        verdict_B = "**beats**" if dB > 0.002 else ("ties" if abs(dB) <= 0.002 else "does **not** beat")
        beat_lines.append(
            f"- **k={k}:** article item-CF {verdict_B} article-popularity "
            f"({cfB:.4f} vs {popB:.4f}, Δ={dB:+.4f}, best repeats={whichB}). "
            f"For contrast, product-type item-CF vs popularity was Δ={dA:+.4f}.")

    A_norep6, A_rep6 = comparison["A_repeats6"]
    B_norep6, B_rep6 = comparison["B_repeats6"]
    popA12, popB12 = comparison["popA12"], comparison["popB12"]

    # Headline logic.
    any_beat = any(beats[k]["B"][3] > 0.002 for k in KS)
    headline = (
        "At article level, popularity is far weaker in absolute terms, and "
        + ("personalization (sparse item-CF) **does** open a gap over popularity — "
           "the headroom that product-type granularity lacked."
           if any_beat else
           "sparse item-CF **still does not decisively beat** article-popularity. "
           "Ultra-fine granularity trades popularity's strength for extreme "
           "sparsity, so neither the coarse nor the ultra-fine extreme is ideal.")
    )

    content = f"""# Week 2 — Granularity Experiment: does recommendation granularity decide
whether personalization beats popularity?

## The question

Across Week 2 we test one hypothesis: **the granularity of the action space
determines whether a personalized model can beat a non-personalized popularity
baseline.** Two arms, same {n_eval:,} core evaluable customers, same leakage-safe
split, same metrics harness.

## Experiment A — product-type (128 actions) [recap]

At product-type granularity the action space is tiny and concentrated. Popularity
was a very strong baseline (hit-rate@12 = {popA12:.3f}); item-to-item CF learned
real structure (swimwear↔swimwear, bra↔underwear) but **did not beat popularity**
at any k. The repeat effect was large — allowing already-bought product types
lifted hit-rate@6 from {A_norep6:.3f} to {A_rep6:.3f} — because category
repurchase is common.

| k | model | repeats | hit_rate | recall | precision |
|---|---|---|---|---|---|
{table(exp_A, {"popularity", "item_cf"})}

## Experiment B — article-level (~{m['n_articles']:,} articles, sparse)

The "action" is now the exact `article_id`. Predicting 1 of ~{m['n_articles'] // 1000}k
articles is far harder than 1 of 128, so **absolute numbers are expected to be
much lower** — the point is the *relative* comparison.

**Sparse engineering.** A dense {m['n_articles']:,}² similarity matrix would be
**{m['dense_would_be_gb']:.1f} GB** — infeasible. We stay sparse end to end
(scipy CSR): binary interaction `A` ({m['A_nnz']:,} nnz, {m['A_mb']:.1f} MB),
cosine similarity via `AᵀA` normalized ({m['C_nnz']:,} nnz, {m['sim_mb']:.1f} MB),
and per-customer scoring as a sparse indicator × similarity product that touches
only co-occurring neighbors.

| k | model | repeats | hit_rate | recall | precision |
|---|---|---|---|---|---|
{table(exp_B, {"article_popularity", "article_item_cf"})}

## Cross-granularity comparison

- **Popularity hit-rate@12:** product-type **{popA12:.4f}** vs article-level
  **{popB12:.4f}** — popularity is dramatically weaker once the target is a
  specific SKU rather than a broad category.
- **Does personalization beat popularity?**
{chr(10).join(beat_lines)}
- **Repeat effect @6:** product-type {A_norep6:.4f}→{A_rep6:.4f}
  (+{A_rep6 - A_norep6:.4f}) vs article {B_norep6:.4f}→{B_rep6:.4f}
  (+{B_rep6 - B_norep6:.4f}). Exact-SKU repurchase is {'rarer' if (B_rep6 - B_norep6) < (A_rep6 - A_norep6) else 'comparable'}
  than category repurchase, so the repeats lift is
  {'weaker' if (B_rep6 - B_norep6) < (A_rep6 - A_norep6) else 'similar'} at article level.

## Honest conclusion

Granularity changes all three things we measured:

1. **Absolute hit-rate** collapses at article level (1-of-{m['n_articles'] // 1000}k
   is intrinsically hard) — from ~{popA12:.2f} down to ~{popB12:.3f} for popularity@12.
2. **The popularity-vs-personalization gap** shifts: product-type popularity is
   too strong to beat; article-level popularity is weak, which is where
   personalization has the most room.
3. **The repeat effect** weakens as granularity fines (category repurchase is
   common; exact-SKU repurchase is rarer).

**Headline.** {headline}

**Motivation for Week 3.** Neither extreme is ideal: coarse (product-type) kills
personalization headroom because popularity is unbeatable; ultra-fine (article)
kills density so every model struggles on absolute hit-rate. This argues for
**richer signal** — content/article features, a mid-level grouping between
product-type and SKU, and recency/sequence features — rather than co-occurrence
counts alone. That is the Week 3 direction.
"""
    path.write_text(content)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
