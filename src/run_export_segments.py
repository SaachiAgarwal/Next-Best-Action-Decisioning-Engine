"""Phase 5d — portfolio-level segment statistics export.

The demo's segment table was computed on the 38-customer demo cohort, so 3 of 4
segments showed 0% hit — a tiny-sample artifact that reads as model failure. This
replaces it with REAL statistics across all 15,246 evaluable customers, using the
production Exp 5 triple hybrid. Aggregate numbers only (~20-30), not customer rows.

STANDALONE: writes only ``data/demo/segments.json``; never touches the other demo
exports. Run with:  python -m src.run_export_segments
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.eval import metrics, diagnostics as dg
from src.models.popularity_article import ArticlePopularityModel
from src.models.item_cf_article import ArticleItemCF
from src.models.content_based_exp4 import ContentModel
from src.models.mf_exp5 import MFModel, TripleHybrid

DEMO_DIR = config.PROCESSED_DIR.parent / "demo"
OUT = DEMO_DIR / "segments.json"
REPORT_PATH = config.REPORTS_DIR / "phase5a_export.md"
W123 = (1.0, 0.5, 1.0)          # Exp 5 tuned triple-hybrid weights
K = 12

# Status thresholds (Task 3), relative to the overall evaluable hit@12.
ABOVE, BELOW = 1.1, 0.9


def _status(seg_hit, overall_hit):
    if seg_hit is None:
        return "No ground truth"
    if seg_hit >= ABOVE * overall_hit:
        return "Above average"
    if seg_hit <= BELOW * overall_hit:
        return "Below average"
    return "At average"


def _round(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    fe = pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    articles = pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    articles["article_id"] = articles["article_id"].astype("string")
    ctx = pd.read_parquet(config.PROCESSED_DIR / "customer_context.parquet",
                          columns=["customer_id", "frequency", "is_cold_start"], engine="pyarrow")
    ctx["customer_id"] = ctx["customer_id"].astype("string")
    la = pd.read_parquet(config.PROCESSED_DIR / "labels_article.parquet", engine="pyarrow")
    la["customer_id"] = la["customer_id"].astype("string")
    la["article_id"] = la["article_id"].astype("string")
    label_sets = {c: set(g) for c, g in la.groupby("customer_id", sort=False)["article_id"]}
    ev_ids = list(label_sets)
    ev_set = set(ev_ids)
    dv = pd.read_parquet(config.PROCESSED_DIR / "divergent_customers_exp3.parquet", engine="pyarrow")
    dv["customer_id"] = dv["customer_id"].astype("string")
    print(f"Phase 5d — segment stats over {len(ev_ids):,} evaluable customers.")

    # --- Build the production triple hybrid + score all evaluable customers ---
    pop = ArticlePopularityModel().fit(fe)
    cf = ArticleItemCF(popularity_model=pop).fit(fe, verbose=False)
    content = ContentModel(popularity_model=pop).fit(
        articles, fe, article_order=cf.article_ids, reference_date=config.CUTOFF_DATE)
    mf = MFModel(popularity_model=pop).fit(fe, article_order=cf.article_ids)
    triple = TripleHybrid(content, cf, mf)
    n_cat = len(cf.article_ids)
    ptmap = {a: n for a, n in zip(articles["article_id"], articles["product_type_name"])}
    pop_rank = dg.popularity_ranks(pop.ranked_articles)
    top10_cut, top1_cut = max(1, n_cat // 10), max(1, n_cat // 100)
    top10_set = set(pop.ranked_articles[:top10_cut])

    print("Scoring triple hybrid for all evaluable customers…")
    recs = triple.recommend_all(ev_ids, k=24, w1=W123[0], w2=W123[1], w3=W123[2],
                                include_repeats=True)

    # --- Segment membership (OVERLAPPING views) ---
    cev = ctx[ctx["customer_id"].isin(ev_set)]
    freq = dict(zip(cev["customer_id"], cev["frequency"]))
    q33, q67 = np.quantile(cev["frequency"].to_numpy(), [1 / 3, 2 / 3])
    def _freq_seg(c):
        f = freq.get(c, 0)
        return "low frequency" if f <= q33 else ("high frequency" if f > q67 else "mid frequency")
    freq_members = {"low frequency": [], "mid frequency": [], "high frequency": []}
    for c in ev_ids:
        freq_members[_freq_seg(c)].append(c)
    div_thresh = float(dv["cosine_to_popularity"].max())
    div_members = [c for c in ev_ids if c in set(dv["customer_id"])]

    cold_total = int(ctx["is_cold_start"].sum())

    # --- Overall baseline ---
    overall = _seg_stats(ev_ids, recs, label_sets, content, ptmap, n_cat, top10_set,
                         pop_rank, top1_cut, top10_cut)
    oh = overall["hit12"]
    print(f"Overall evaluable hit@12={oh:.4f} recall@12={overall['recall12']:.4f}")

    # --- Per-segment ---
    seg_defs = [
        ("low frequency", f"pre-cutoff purchase count ≤ {q33:.0f} (bottom tercile)", freq_members["low frequency"]),
        ("mid frequency", f"pre-cutoff purchase count {q33:.0f}–{q67:.0f} (middle tercile)", freq_members["mid frequency"]),
        ("high frequency", f"pre-cutoff purchase count > {q67:.0f} (top tercile)", freq_members["high frequency"]),
        ("divergent taste", f"bottom-quartile cosine-to-popularity ≤ {div_thresh:.4f} (Phase 3b/Exp 3 slice)", div_members),
    ]
    segments = []
    for name, definition, ids in seg_defs:
        s = _seg_stats(ids, recs, label_sets, content, ptmap, n_cat, top10_set,
                       pop_rank, top1_cut, top10_cut)
        s.update({"segment": name, "definition": definition,
                  "share": _round(len(ids) / len(ev_ids)),
                  "status": _status(s["hit12"], oh)})
        segments.append(s)

    # --- Biggest addressable gap (count × deficit-vs-average), exclusive freq segs only
    # to avoid double-counting overlaps; divergent considered too but flagged as a view.
    gap_cands = segments  # all four; divergent may win on volume×deficit
    def _gap_score(s):
        return s["customers"] * max(0.0, oh - s["hit12"])
    biggest = max(gap_cands, key=_gap_score)
    stmt = (f"{biggest['segment'].capitalize()} customers ({biggest['customers']:,}, "
            f"{biggest['share']:.0%} of the evaluable base) convert at "
            f"{biggest['hit12']:.1%} vs the {oh:.1%} average — the largest addressable "
            f"gap by volume.")
    print("Biggest gap:", stmt)

    obj = {
        "overall": {"customers": len(ev_ids), "hit12": _round(oh),
                    "recall12": _round(overall["recall12"]),
                    "avg_distinct_types": _round(overall["avg_distinct_types"], 2),
                    "coverage12": _round(overall["coverage12"], 2)},
        "segments": [_slim(s) for s in segments],
        "cold_start": {"customers": cold_total,
                       "note": "outside evaluable set — no ground truth"},
        "biggest_gap": {"segment": biggest["segment"], "statement": stmt},
        "caveat": ("Computed across all 15,246 evaluable customers, not the "
                   "38-customer demo cohort."),
    }
    OUT.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    json.loads(OUT.read_text())
    size = OUT.stat().st_size

    _print_table(obj)
    print(f"\nWrote {OUT}  ({size/1024:.1f} KB)")
    _append_report(obj, size, q33, q67, div_thresh, cold_total)
    print("DONE.")


def _seg_stats(ids, recs, label_sets, content, ptmap, n_cat, top10_set,
               pop_rank, top1_cut, top10_cut):
    ids = [c for c in ids if c in recs]
    sub = {c: recs[c] for c in ids}
    ls = {c: label_sets[c] for c in ids if c in label_sets}
    acc = metrics.evaluate(sub, ls, ks=[K])
    r12 = {c: v[:K] for c, v in sub.items()}
    cov = dg.coverage(sub, n_cat, K, top10_set)
    bias = dg.popularity_bias(r12, pop_rank, n_cat, top1_cut, top10_cut)
    div = dg.intra_list_diversity(sub, content.item_matrix, content.article_index,
                                  ptmap, K, sample=min(3000, len(sub)))
    true_rank = dg.label_mean_pop_rank(ls, pop_rank, n_cat)
    return {"customers": len(ids),
            "hit12": float(acc.iloc[0]["hit_rate"]), "recall12": float(acc.iloc[0]["recall"]),
            "avg_distinct_types": div["avg_distinct_types"], "coverage12": cov["coverage_pct"],
            "mean_rec_pop_rank": bias["mean_pop_rank"], "mean_true_pop_rank": true_rank}


def _slim(s):
    return {"segment": s["segment"], "definition": s["definition"], "customers": s["customers"],
            "share": s["share"], "hit12": _round(s["hit12"]), "recall12": _round(s["recall12"]),
            "avg_distinct_types": _round(s["avg_distinct_types"], 2),
            "coverage12": _round(s["coverage12"], 2),
            "mean_rec_pop_rank": _round(s["mean_rec_pop_rank"], 0),
            "mean_true_pop_rank": _round(s["mean_true_pop_rank"], 0), "status": s["status"]}


def _print_table(obj):
    print("\n" + "=" * 92)
    print(f"{'segment':18s} {'n':>7s} {'share':>6s} {'hit@12':>7s} {'recall@12':>9s} "
          f"{'types':>5s} {'cov%':>6s} {'rec_rank':>9s} {'true_rank':>9s}  status")
    print("=" * 92)
    o = obj["overall"]
    print(f"{'OVERALL':18s} {o['customers']:>7,} {'100%':>6s} {o['hit12']:>7.4f} "
          f"{o['recall12']:>9.4f} {o['avg_distinct_types']:>5.1f} {o['coverage12']:>6.1f}")
    for s in obj["segments"]:
        print(f"{s['segment']:18s} {s['customers']:>7,} {s['share']:>6.0%} {s['hit12']:>7.4f} "
              f"{s['recall12']:>9.4f} {s['avg_distinct_types']:>5.1f} {s['coverage12']:>6.1f} "
              f"{s['mean_rec_pop_rank']:>9,.0f} {s['mean_true_pop_rank']:>9,.0f}  {s['status']}")
    c = obj["cold_start"]
    print(f"{'cold start':18s} {c['customers']:>7,} {'—':>6s} {'null':>7s}   ({c['note']})")


def _append_report(obj, size, q33, q67, div_thresh, cold_total):
    marker = "## Phase 5d — Portfolio Segment Statistics"
    existing = REPORT_PATH.read_text()
    if marker in existing:
        existing = existing[:existing.index(marker)].rstrip() + "\n\n"
    o = obj["overall"]
    rows = "\n".join(
        f"| {s['segment']} | {s['customers']:,} | {s['share']:.0%} | {s['hit12']:.4f} | "
        f"{s['recall12']:.4f} | {s['avg_distinct_types']:.1f} | {s['coverage12']:.1f}% | "
        f"{s['mean_rec_pop_rank']:,.0f} | {s['mean_true_pop_rank']:,.0f} | {s['status']} |"
        for s in obj["segments"])
    section = f"""{marker} (`segments.json`)

Replaces the demo's segment table — which was computed on the 38-customer cohort
and showed 0% hit for 3 of 4 segments (a tiny-sample artifact, not model failure) —
with **real statistics across all 15,246 evaluable customers**, scored by the
production Exp 5 triple hybrid. Aggregate numbers only. Size **{size/1024:.1f} KB**.
Standalone script `src/run_export_segments.py`; the other demo files are untouched.

**Segment definitions (OVERLAPPING views — a divergent customer can also be
high-frequency, so segment counts do NOT sum to the evaluable total):**
- **Frequency terciles** partition the evaluable set (sum = 15,246): low ≤ {q33:.0f}
  pre-cutoff purchases, mid {q33:.0f}–{q67:.0f}, high > {q67:.0f}.
- **Divergent taste** is an overlapping slice: the Phase 3b/Exp 3 bottom-quartile by
  cosine-to-popularity (cosine ≤ {div_thresh:.4f}), 25% of the base.
- **Cold start** ({cold_total:,}) is disjoint — outside the evaluable set (no
  label-window purchase), so it has **no ground truth**: hit rate is `null`, not 0%.

**Status thresholds (Task 3), vs the overall hit@12 of {o['hit12']:.4f}:** Above
average ≥ {ABOVE}×, Below average ≤ {BELOW}×, else At average; cold-start = "No
ground truth".

| segment | customers | share | hit@12 | recall@12 | types/list | coverage@12 | mean rec pop-rank | mean true pop-rank | status |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | {o['customers']:,} | 100% | {o['hit12']:.4f} | {o['recall12']:.4f} | {o['avg_distinct_types']:.1f} | {o['coverage12']:.1f}% | — | — | — |
{rows}
| cold start | {cold_total:,} | — | null | — | — | — | — | — | No ground truth |

`mean rec pop-rank` vs `mean true pop-rank` is the **demand-alignment gap** per
segment (higher rank = deeper in the tail): where the model recommends vs where the
segment actually buys.

**Biggest addressable gap** (max customers × deficit-vs-average):
> {obj['biggest_gap']['statement']}

### Schema — `segments.json`
```
{{
  "overall": {{ customers, hit12, recall12, avg_distinct_types, coverage12 }},
  "segments": [ {{ segment, definition, customers, share, hit12, recall12,
                  avg_distinct_types, coverage12, mean_rec_pop_rank,
                  mean_true_pop_rank, status }} ],
  "cold_start": {{ customers, note }},          # hit rate is null — no ground truth
  "biggest_gap": {{ segment, statement }},
  "caveat": "…all 15,246 evaluable customers, not the 38-customer demo cohort."
}}
```
"""
    REPORT_PATH.write_text(existing.rstrip() + "\n\n" + section)


if __name__ == "__main__":
    main()
