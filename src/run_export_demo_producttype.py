"""Phase 5b — product-type layer export for the demo.

Adds a *category-granularity* view to data/demo/ so the app can show the SAME 38
customers at both granularities: which CATEGORY to recommend (128 product-type
actions, hit@12 = 0.834) and — from Phase 5a — which SPECIFIC PRODUCT (~79k
articles, hit@12 = 0.063). The point is to make task difficulty visible: the low
SKU hit-rate is not model failure, it is what predicting 1-of-79,269 looks like.

Reuses the Exp 3 recency+frequency hybrid (product-type production model) and the
Exp A popularity baseline, exactly as Exp 3 built them. Reads the Phase 5a cohort
from data/demo/customers.json so the two layers stay in lockstep. No prior
artifacts are modified.

Run with:  python -m src.run_export_demo_producttype
"""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.models.popularity import PopularityModel
from src.models.hybrid import HybridModel

DEMO_DIR = config.PROCESSED_DIR.parent / "demo"
OUT = DEMO_DIR / "producttype.json"
REPORT_PATH = config.REPORTS_DIR / "phase5a_export.md"
K = 12


def _pack(action_ids, aname, label_set):
    out = []
    for rank, a in enumerate(action_ids):
        a = int(a)
        out.append({"action_id": a, "type": aname.get(a), "rank": rank,
                    "hit": a in label_set})
    return out


def _comparison(diag):
    """Authoritative granularity-comparison numbers (from metrics_summary/diagnostics)."""
    tr = diag[diag["model"] == "triple hybrid"].iloc[0]
    n_articles = round(tr["coverage_count@12"] / (tr["coverage_pct@12"] / 100.0))
    note = (
        "Product-type hit@12 (0.834) LOOKS far better than article-level (0.063), but "
        "they are DIFFERENT TASKS and are NOT comparable: predicting 1-of-128 categories "
        "vs 1-of-79,269 specific products. At the category level the task is easy for "
        "everyone, so the personalized model beats popularity by only +0.7% (0.834 vs "
        "0.827). At the article level the task is hard, and personalization DOUBLES the "
        "popularity baseline (0.063 vs 0.031). Together they describe a hierarchical "
        "architecture — pick the category (tractable, learnable, where a bandit can "
        "operate), then the specific product within it (where content + latent factors "
        "earn their keep). NOTE: the drill-down chain is NOT implemented end-to-end; "
        "these are two parallel views of the same customer, and the chain is the natural "
        "next extension."
    )
    return {
        "product_type": {"n_actions": 128, "exp3_hit@12": 0.8339, "exp3_recall@12": 0.6164,
                         "popularity_hit@12": 0.8267, "margin_over_popularity": "+0.7%"},
        "article": {"n_articles": int(n_articles), "triple_hybrid_hit@12": 0.0628,
                    "triple_hybrid_recall@12": 0.0229, "popularity_hit@12": 0.0314,
                    "lift_over_popularity": "2.0x"},
        "note": note,
    }


def main():
    if not (DEMO_DIR / "customers.json").exists():
        raise SystemExit("Run Phase 5a first (python -m src.run_export_demo).")
    cohort = json.loads((DEMO_DIR / "customers.json").read_text())
    handle_cid = [(c["id"], c["cid"]) for c in cohort]
    cids = [cid for _, cid in handle_cid]
    print(f"Phase 5b — product-type layer for {len(cids)} demo customers. Building models …")

    fe = pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    actions = pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
    aname = dict(zip(actions["action_id"].astype(int), actions["product_type_name"]))
    lab = pd.read_parquet(config.PROCESSED_DIR / "labels.parquet", engine="pyarrow")
    lab["customer_id"] = lab["customer_id"].astype("string")
    label_sets = {c: set(int(x) for x in g) for c, g in
                  lab.groupby("customer_id", sort=False)["action_id"]}
    diag = pd.read_parquet(config.PROCESSED_DIR / "diagnostics_results.parquet", engine="pyarrow")
    w3 = json.loads((config.PROCESSED_DIR / "hybrid_weights_exp3.json").read_text())
    a, b, g = w3["alpha"], w3["beta"], w3["gamma"]

    pop = PopularityModel().fit(fe)
    hybrid = HybridModel().fit(fe, reference_date=config.CUTOFF_DATE)

    customers = {}
    for handle, cid in handle_cid:
        ls = label_sets.get(cid, set())
        hyb = hybrid.recommend(cid, k=K, alpha=a, beta=b, gamma=g)
        popr = pop.recommend(cid, k=K)
        gt_types = sorted({aname.get(int(x)) for x in ls} - {None})
        customers[handle] = {
            "exp3_hybrid": _pack(hyb, aname, ls),
            "popularity": _pack(popr, aname, ls),
            "ground_truth": {"n": len(ls), "types": gt_types},
        }

    obj = {"comparison": _comparison(diag), "customers": customers}
    OUT.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    json.loads(OUT.read_text())     # validate

    size_kb = OUT.stat().st_size / 1024
    _print_summary(handle_cid, customers, obj["comparison"], size_kb)
    _append_report(obj["comparison"], size_kb, len(cids))
    print("\nDONE.")


def _print_summary(handle_cid, customers, comp, size_kb):
    print("\n" + "=" * 70)
    print("SAMPLE — product-type recommendations (exp3 hybrid)")
    print("=" * 70)
    for handle, _ in handle_cid[:6]:
        c = customers[handle]
        top = c["exp3_hybrid"][0]
        hits = sum(1 for it in c["exp3_hybrid"] if it["hit"])
        print(f"  {handle}: top='{top['type']}' hit={top['hit']} | "
              f"gt={c['ground_truth']['n']} types, {hits}/12 hits | "
              f"gt_types={c['ground_truth']['types'][:4]}")
    print(f"\ncomparison: product_type hit@12={comp['product_type']['exp3_hit@12']} "
          f"(pop {comp['product_type']['popularity_hit@12']}, "
          f"{comp['product_type']['margin_over_popularity']}) vs "
          f"article hit@12={comp['article']['triple_hybrid_hit@12']} "
          f"(pop {comp['article']['popularity_hit@12']}, {comp['article']['lift_over_popularity']})")
    print(f"producttype.json: {size_kb:.1f} KB")


def _append_report(comp, size_kb, n):
    pt, ar = comp["product_type"], comp["article"]
    section = f"""

---

# Phase 5b — Product-Type Layer (granularity comparison)

Adds `data/demo/producttype.json` ({size_kb:.1f} KB, committed) so the app can show
the **same {n} customers at both granularities**: which *category* to recommend
(128 product-type actions) and — from 5a — which *specific product* (~{ar['n_articles']:,}
articles). This makes **task difficulty visible**: the low SKU hit-rate is not model
failure, it is what predicting 1-of-{ar['n_articles']:,} looks like.

## `producttype.json` schema
```
{{
  "comparison": {{
     "product_type": {{ n_actions, exp3_hit@12, exp3_recall@12,
                       popularity_hit@12, margin_over_popularity }},
     "article":      {{ n_articles, triple_hybrid_hit@12, triple_hybrid_recall@12,
                       popularity_hit@12, lift_over_popularity }},
     "note": "why the two numbers are not comparable"
  }},
  "customers": {{
     "C01": {{
        "exp3_hybrid":  [ {{ action_id, type, rank, hit }} x12 ],  # Exp 3 production model
        "popularity":   [ {{ action_id, type, rank, hit }} x12 ],  # Exp A baseline
        "ground_truth": {{ n, types:[str] }}                       # LABEL-WINDOW product types
     }}, ...
  }}
}}
```
`hit` = the customer bought that product type in the label window (from
`labels.parquet`, the product-type labels). Cold-start customers have an empty
`ground_truth` here too (same caveat as 5a).

## The two layers, honestly

| layer | task | production model | hit@12 | recall@12 | vs popularity |
|---|---|---|---|---|---|
| **product-type** | 1-of-{pt['n_actions']} | Exp 3 recency+freq hybrid | **{pt['exp3_hit@12']}** | {pt['exp3_recall@12']} | pop {pt['popularity_hit@12']} → **{pt['margin_over_popularity']}** |
| **article** | 1-of-{ar['n_articles']:,} | Exp 5 triple hybrid | **{ar['triple_hybrid_hit@12']}** | {ar['triple_hybrid_recall@12']} | pop {ar['popularity_hit@12']} → **{ar['lift_over_popularity']}** |

- Product-type hit@12 (**{pt['exp3_hit@12']}**) LOOKS far better than article
  (**{ar['triple_hybrid_hit@12']}**), but they are **different tasks and NOT
  comparable** — 1-of-{pt['n_actions']} vs 1-of-{ar['n_articles']:,}.
- At **product-type**, personalization beats popularity by only **{pt['margin_over_popularity']}**
  ({pt['exp3_hit@12']} vs {pt['popularity_hit@12']}): the category task is easy for
  everyone, so personalization adds little.
- At **article**, the model **{ar['lift_over_popularity']}** the popularity baseline
  ({ar['triple_hybrid_hit@12']} vs {ar['popularity_hit@12']}): personalization earns
  real value exactly where the task is hard.
- Together the layers describe a **hierarchical architecture**: choose the category
  (tractable, learnable, where a bandit can operate), then the specific product
  within it (where content and latent factors earn their keep). **Honest caveat:**
  the drill-down chain is **not implemented end-to-end** — these are two parallel
  views of the same customer, and the chain is the natural next extension.
"""
    with open(REPORT_PATH, "a") as f:
        f.write(section)


if __name__ == "__main__":
    main()
