"""Consolidated metrics table: complete hit/recall/precision + diagnostics.

Run with:  python -m src.run_metrics_table

Fills the recall/precision gaps across all article-level models on the same
15,246 core evaluable set, joins the already-computed beyond-accuracy diagnostics,
and separately tabulates the product-type experiments. No new models.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.eval import evaluable, metrics
# Article-level models.
from src.models.popularity_article import ArticlePopularityModel
from src.models.item_cf_article import ArticleItemCF
from src.models.content_based_exp4 import ContentModel
from src.models.hybrid_content_cf_exp4 import ContentCFHybrid
from src.models.mf_exp5 import MFModel, TripleHybrid
# Product-type models.
from src.models.popularity import PopularityModel, load_feature_events
from src.models.item_cf import ItemCF
from src.models.hybrid import HybridModel

KS = [6, 12, 24]
OUT = config.PROCESSED_DIR / "metrics_summary.parquet"


def _acc_row(recs, label_sets, model, extra=None):
    df = metrics.evaluate(recs, label_sets, ks=KS)
    row = {"model": model}
    for r in df.itertuples():
        row[f"hit@{int(r.k)}"] = r.hit_rate
        row[f"recall@{int(r.k)}"] = r.recall
        row[f"precision@{int(r.k)}"] = r.precision
    if extra:
        row.update(extra)
    return row


def _curve_final(path, alpha):
    """Final held-out hit@k for a bandit's best alpha, from its learning curve."""
    g = pd.read_parquet(path, engine="pyarrow")
    g = g[g["alpha"] == alpha].sort_values("learning_steps_seen")
    last = g.iloc[-1]
    return {c: float(last[c]) for c in g.columns if c.startswith("heldout_hit@")}


# --------------------------------------------------------------------------
# Article-level table
# --------------------------------------------------------------------------
def article_table():
    fe = pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    articles = pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    la = pd.read_parquet(config.PROCESSED_DIR / "labels_article.parquet", engine="pyarrow")
    label_sets = {c: set(g) for c, g in la.groupby("customer_id", sort=False)["article_id"]}
    ev = list(label_sets)
    assert len(ev) == 15246
    n_eval = len(ev)

    pop = ArticlePopularityModel().fit(fe)
    cf = ArticleItemCF(popularity_model=pop).fit(fe, verbose=False)
    content = ContentModel(popularity_model=pop).fit(articles, fe, article_order=cf.article_ids)
    mf = MFModel(popularity_model=pop).fit(fe, article_order=cf.article_ids)
    cch = ContentCFHybrid(content, cf)
    triple = TripleHybrid(content, cf, mf)
    w4 = json.loads((config.PROCESSED_DIR / "hybrid_weights_exp4.json").read_text())
    w5 = json.loads((config.PROCESSED_DIR / "hybrid_weights_exp5.json").read_text())

    rows = [
        _acc_row(pop.recommend_all(ev, k=24), label_sets, "article popularity", {"repeats": "n/a", "n": n_eval}),
        _acc_row(cf.recommend_all(ev, k=24, include_repeats=True), label_sets, "neighborhood CF (Exp B)", {"repeats": "True", "n": n_eval}),
        _acc_row(content.recommend_all(ev, k=24, include_repeats=True), label_sets, "content (Exp 4)", {"repeats": "True", "n": n_eval}),
        _acc_row(cch.recommend_all(ev, k=24, alpha=w4["content_alpha"], beta=w4["content_beta"], include_repeats=True), label_sets, "content+CF hybrid (Exp 4)", {"repeats": "True", "n": n_eval}),
        _acc_row(mf.recommend_all(ev, k=24, include_repeats=True), label_sets, "MF (Exp 5)", {"repeats": "True", "n": n_eval}),
        _acc_row(triple.recommend_all(ev, k=24, w1=w5["w1"], w2=w5["w2"], w3=w5["w3"], include_repeats=True), label_sets, "triple hybrid (Exp 5, production)", {"repeats": "True", "n": n_eval}),
    ]

    # Shared bandit (Phase 2c) — held-out subset, flagged.
    sb = _curve_final(config.PROCESSED_DIR / "bandit_shared_learning_curve.parquet", 0.5)
    rows.append({"model": "shared bandit (Phase 2c, α=0.5) †", "repeats": "n/a", "n": 4574,
                 "hit@6": sb.get("heldout_hit@6"), "hit@12": sb.get("heldout_hit@12"),
                 "hit@24": sb.get("heldout_hit@24")})

    acc = pd.DataFrame(rows)

    # Join diagnostics (from the already-computed run).
    diag = pd.read_parquet(config.PROCESSED_DIR / "diagnostics_results.parquet", engine="pyarrow")
    diag_map = {"popularity": "article popularity", "neighborhood CF": "neighborhood CF (Exp B)",
                "content": "content (Exp 4)", "MF": "MF (Exp 5)", "triple hybrid": "triple hybrid (Exp 5, production)"}
    diag = diag.assign(model=diag["model"].map(diag_map))
    diag_cols = ["model", "coverage_pct@12", "mean_pop_rank", "pct_top10", "gini",
                 "intra_list_dissim", "avg_distinct_types", "seg_spread@12", "cold_article_scorable_pct"]
    return acc.merge(diag[diag_cols], on="model", how="left"), n_eval


# --------------------------------------------------------------------------
# Product-type table
# --------------------------------------------------------------------------
def producttype_table():
    fe = load_feature_events()
    evaluable_ids, label_sets = evaluable.get_evaluable()   # product-type action label sets
    ev = list(evaluable_ids)
    n_eval = len(ev)

    pop = PopularityModel().fit(fe)
    cf = ItemCF(popularity_model=pop).fit(fe)
    hybrid = HybridModel().fit(fe, reference_date=config.CUTOFF_DATE)
    w3 = json.loads((config.PROCESSED_DIR / "hybrid_weights_exp3.json").read_text())

    rows = [
        _acc_row(pop.recommend_all(ev, k=24), label_sets, "popularity (Exp A)", {"repeats": "n/a", "n": n_eval}),
        _acc_row(cf.recommend_all(ev, k=24, include_repeats=True), label_sets, "item-CF (Exp A)", {"repeats": "True", "n": n_eval}),
        _acc_row(hybrid.recommend_all(ev, k=24, alpha=w3["alpha"], beta=w3["beta"], gamma=w3["gamma"]), label_sets, "recency+freq hybrid (Exp 3, production)", {"repeats": "incl.", "n": n_eval}),
    ]
    # LinUCB v3 (held-out, product-type) — flagged; curve HITK = [1,6,12] (no @24).
    v3 = _curve_final(config.PROCESSED_DIR / "bandit_learning_curve_v3.parquet", 1.0)
    rows.append({"model": "LinUCB v3 (best α=1.0) †", "repeats": "n/a", "n": 4574,
                 "hit@6": v3.get("heldout_hit@6"), "hit@12": v3.get("heldout_hit@12")})
    return pd.DataFrame(rows), n_eval


def _fmt(v):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"


def _get(row, col):
    return row[col] if col in row else None


def _article_md(df):
    hdr = ("| model | repeats | hit@6 | hit@12 | hit@24 | recall@6 | recall@12 | recall@24 "
           "| prec@6 | prec@12 | prec@24 | cov@12 | mean pop rank | head% | Gini | diversity | fair spread | cold-art% |")
    sep = "|" + "---|" * 18
    lines = [hdr, sep]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {_get(r, 'repeats')} | {_fmt(_get(r, 'hit@6'))} | {_fmt(_get(r, 'hit@12'))} | {_fmt(_get(r, 'hit@24'))} "
            f"| {_fmt(_get(r, 'recall@6'))} | {_fmt(_get(r, 'recall@12'))} | {_fmt(_get(r, 'recall@24'))} "
            f"| {_fmt(_get(r, 'precision@6'))} | {_fmt(_get(r, 'precision@12'))} | {_fmt(_get(r, 'precision@24'))} "
            f"| {_pct(_get(r, 'coverage_pct@12'))} | {_num(_get(r, 'mean_pop_rank'))} | {_pct(_get(r, 'pct_top10'))} "
            f"| {_fmt(_get(r, 'gini'))} | {_fmt(_get(r, 'intra_list_dissim'))} | {_fmt(_get(r, 'seg_spread@12'))} | {_pct(_get(r, 'cold_article_scorable_pct'))} |")
    return "\n".join(lines)


def _pt_md(df):
    hdr = ("| model | repeats | hit@6 | hit@12 | hit@24 | recall@6 | recall@12 | recall@24 "
           "| prec@6 | prec@12 | prec@24 |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {_get(r, 'repeats')} | {_fmt(_get(r, 'hit@6'))} | {_fmt(_get(r, 'hit@12'))} | {_fmt(_get(r, 'hit@24'))} "
            f"| {_fmt(_get(r, 'recall@6'))} | {_fmt(_get(r, 'recall@12'))} | {_fmt(_get(r, 'recall@24'))} "
            f"| {_fmt(_get(r, 'precision@6'))} | {_fmt(_get(r, 'precision@12'))} | {_fmt(_get(r, 'precision@24'))} |")
    return "\n".join(lines)


def _pct(v):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.1f}%"


def _num(v):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:,.0f}"


def main():
    art, n_art = article_table()
    pt, n_pt = producttype_table()

    # Persist combined (long form with a regime column).
    art2 = art.assign(regime="article"); pt2 = pt.assign(regime="product-type")
    pd.concat([art2, pt2], ignore_index=True).to_parquet(OUT, engine="pyarrow")

    print("\n=== ARTICLE-LEVEL (15,246 core; bandit on 4,574 held-out †) ===")
    print(art[["model", "hit@12", "recall@12", "precision@12", "coverage_pct@12"]].to_string(index=False))
    print("\n=== PRODUCT-TYPE (15,246 core; bandit on held-out †) ===")
    print(pt[["model", "hit@12", "recall@12", "precision@12"]].to_string(index=False))

    _write_report(art, pt, n_art)
    print(f"\nWrote metrics_summary.parquet ({len(art) + len(pt)} rows)")
    print("DONE.")


def _write_report(art, pt, n_eval):
    path = config.REPORTS_DIR / "metrics_summary.md"

    # MF recall vs hit for the synthesis.
    def g(df, m, col):
        r = df[df["model"] == m]
        return float(r[col].iloc[0]) if len(r) and not pd.isna(r[col].iloc[0]) else float("nan")
    mf_hit = g(art, "MF (Exp 5)", "hit@12"); mf_rec = g(art, "MF (Exp 5)", "recall@12")
    tri_rec = g(art, "triple hybrid (Exp 5, production)", "recall@12")
    cf_rec = g(art, "neighborhood CF (Exp B)", "recall@12")

    content = f"""# Consolidated Metrics Summary

> **⚠️ Article-level and product-type numbers are NOT comparable.** They are
> different prediction tasks — 1-of-79,269 articles vs. 1-of-128 product types —
> so raw metrics differ by an order of magnitude for structural reasons, not
> quality. **Compare only within a regime (within a table).**

> **†** The bandit rows (shared bandit, LinUCB v3) were evaluated on their
> **held-out set (4,574 customers)**, a *different* customer set from the 15,246
> core used for every other row. They are shown for reference and are **not
> strictly comparable** to the static models here.

## How to read this

- **hit-rate@k** — did we get *at least one* of the customer's purchases into the
  top-k? (breadth of "got something right")
- **recall@k** — what *fraction* of everything the customer bought did we cover?
- **precision@k** — what fraction of our k recommendations actually landed?
- **coverage / Gini / diversity / fairness / cold-article** — beyond-accuracy
  diagnostics (Phase 3a) that accuracy metrics are structurally blind to:
  how much catalog is reachable, how concentrated on the head, how varied each
  list is, how evenly customers are served, and whether cold items can be scored.

All accuracy metrics use the **best repeat setting per model** (repeats=True for
the CF/content/MF/hybrid models — fashion is repurchase-heavy; popularity has no
repeat notion). Evaluated on the same **{n_eval:,} core evaluable customers**
(article-level) via the Day 1 harness keyed on `article_id`.

## Article-level models (15,246 core customers)

{_article_md(art)}

*Diagnostics columns are blank for the content+CF hybrid (it was not part of the
Phase 3a diagnostics run) and for the held-out bandit.*

## Product-type models (15,246 core customers)

Coverage is **not reported** for product-type models: with only 128 actions,
catalog coverage is near-total by construction and would be a vacuous number.

{_pt_md(pt)}

## Synthesis — where recall tells a different story than hit-rate

The two accuracy views can rank models differently, and the beyond-accuracy
diagnostics rank them in the *opposite* order to accuracy. Most telling:
**MF** posts a strong hit@12 ({mf_hit:.4f}) but its **recall@12 is only
{mf_rec:.4f}** — it catches *a* purchase for many customers but a *thin slice* of
each customer's basket, because it concentrates on a tiny popular head (0.82%
catalog coverage, 100% head share). The triple hybrid's recall@12 ({tri_rec:.4f})
is higher for similar-order hit-rate — it spreads across more of the basket.
Neighborhood CF, weakest on hit-rate, has recall@12 {cf_rec:.4f} while covering
78% of the catalog — the healthiest breadth.

Restating the central finding from Phase 3a: **accuracy and coverage rank the
models in opposite orders.** The accuracy leaders (MF, triple hybrid) are the
coverage/diversity laggards; the coverage/diversity leader (neighborhood CF) is
the accuracy laggard. A single accuracy number — hit-rate especially — hides this
entirely, which is exactly why the full table (recall + precision + diagnostics)
exists.
"""
    path.write_text(content)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
