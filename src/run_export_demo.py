"""Phase 5a — export a small, self-contained JSON dataset for a Lovable web demo.

The browser app cannot run the pipeline or read parquet, so everything it shows is
pre-computed here and written as compact JSON to ``data/demo/`` (committed). No new
models are trained — this reads existing artifacts and reuses the Exp 5 models +
Phase 3b re-ranker + Phase 4 explainer, exactly as prior phases built them.

Run with:  python -m src.run_export_demo

Files written (schemas documented in reports/phase5a_export.md):
  customers.json       — per-customer profile, pre-cutoff history, label-window truth
  recommendations.json — top-12 from 5 model variants (toggle + diversity slider)
  explanations.json    — grounded "why" + evidence bundle + fidelity + adversarial
  frontier.json        — recall-vs-coverage points across (lambda, pop_penalty)
  diagnostics.json     — headline per-model diagnostics (accuracy vs coverage panel)

Leakage guard: history/profile are pre-cutoff (t_dat < CUTOFF_DATE); ground_truth
is the label window — the two never mix.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.models.popularity_article import ArticlePopularityModel
from src.models.item_cf_article import ArticleItemCF
from src.models.content_based_exp4 import ContentModel, _row_minmax
from src.models.hybrid_content_cf_exp4 import cf_score_chunk
from src.models.mf_exp5 import MFModel
from src.rerank.reranker import ReRanker
from src.explain import agent as A
from src.explain.verifier import RuleGate, SoftVerifier
from src.run_explain import explain_and_verify, _ScriptedGenerator

DEMO_DIR = config.PROCESSED_DIR.parent / "demo"     # data/demo/
REPORT_PATH = config.REPORTS_DIR / "phase5a_export.md"
K = 12
N_RETRIEVE = config.N_RETRIEVE
# Frontier settings surfaced by the diversity slider.
DIV = (0.7, 0.0)     # diversity setting
COV = (0.3, 0.3)     # max-coverage setting
# diagnostics model -> metrics_summary model (to attach recall@12).
_RECALL_MAP = {
    "triple hybrid": "triple hybrid (Exp 5, production)",
    "MF": "MF (Exp 5)", "content": "content (Exp 4)",
    "neighborhood CF": "neighborhood CF (Exp B)", "popularity": "article popularity",
}


def _r(x, n=4):
    """Round a float for compact JSON; pass through None/str; coerce numpy scalars."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return round(float(x), n)


# ---------------------------------------------------------------------------
# Cohort selection (Task 1)
# ---------------------------------------------------------------------------
def select_cohort(ctx, evaluable, divergent, warm_set, rng):
    """~40 customers showcasing variety.

    Warm buckets are drawn from the **evaluable** (labelled) set so the demo can
    show honest hits/misses. Cold-start is a data-driven exception: in this dataset
    *no* zero-history customer has any label-window purchase (cold_start ∩
    evaluable = ∅), so a few true cold-start customers are included with an empty
    ground_truth — their purpose is to demonstrate the popularity fallback, and the
    report says so plainly.
    """
    c = ctx[ctx["customer_id"].isin(evaluable)].copy()
    warm = c[(~c["is_cold_start"]) & (c["customer_id"].isin(warm_set))]
    cold = ctx[ctx["is_cold_start"]]           # NOT evaluable-filtered (see docstring)

    def _sample(df, n):
        if len(df) == 0:
            return []
        idx = rng.choice(len(df), min(n, len(df)), replace=False)
        return df.iloc[idx]["customer_id"].tolist()

    fq = warm["frequency"]
    hi_pool = warm[fq >= fq.quantile(0.90)]
    mid_pool = warm[(fq >= fq.quantile(0.40)) & (fq <= fq.quantile(0.60))]
    div_pool = warm[warm["customer_id"].isin(set(divergent.head(400)["customer_id"]))]

    picks, seg = [], {}
    for cid in _sample(hi_pool, 12):
        picks.append(cid); seg[cid] = "high_freq"
    for cid in _sample(mid_pool, 12):
        if cid not in seg:
            picks.append(cid); seg[cid] = "mid_history"
    # Divergent: most-divergent first (deterministic), warm, not already chosen.
    dord = divergent[divergent["customer_id"].isin(set(div_pool["customer_id"]))]
    for cid in dord["customer_id"].tolist():
        if cid not in seg and len([s for s in seg.values() if s == "divergent"]) < 8:
            picks.append(cid); seg[cid] = "divergent"
    for cid in _sample(cold, 6):
        if cid not in seg:
            picks.append(cid); seg[cid] = "cold_start"
    return picks, seg


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------
def _norm_rows(content, cf, mf, warm_ids):
    """Per-catalog min-max normalized component score rows for warm customers."""
    Cn = _row_minmax(content.score_chunk(warm_ids))
    Fn = _row_minmax(cf_score_chunk(cf, warm_ids))
    Mn = _row_minmax(mf.score_chunk(warm_ids))
    return Cn, Fn, Mn


def _topk(score_row, k):
    top = np.argpartition(-score_row, k - 1)[:k]
    return top[np.argsort(-score_row[top])]


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    print("Phase 5a — demo export. Loading artifacts + building models …")

    ctx = pd.read_parquet(config.PROCESSED_DIR / "customer_context.parquet", engine="pyarrow")
    ctx["customer_id"] = ctx["customer_id"].astype("string")
    articles = pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    articles["article_id"] = articles["article_id"].astype("string")
    el = pd.read_parquet(config.PROCESSED_DIR / "event_log.parquet",
                         columns=["customer_id", "t_dat", "article_id"], engine="pyarrow")
    el["customer_id"] = el["customer_id"].astype("string")
    el["article_id"] = el["article_id"].astype("string")
    el["t_dat"] = pd.to_datetime(el["t_dat"])
    la = pd.read_parquet(config.PROCESSED_DIR / "labels_article.parquet", engine="pyarrow")
    la["customer_id"] = la["customer_id"].astype("string")
    la["article_id"] = la["article_id"].astype("string")
    label_sets = {c: list(g) for c, g in la.groupby("customer_id", sort=False)["article_id"]}
    evaluable = set(label_sets)
    divergent = pd.read_parquet(config.PROCESSED_DIR / "divergent_customers_exp3.parquet",
                                engine="pyarrow")
    divergent["customer_id"] = divergent["customer_id"].astype("string")
    frontier = pd.read_parquet(config.PROCESSED_DIR / "rerank_frontier.parquet", engine="pyarrow")
    diag = pd.read_parquet(config.PROCESSED_DIR / "diagnostics_results.parquet", engine="pyarrow")
    msum = pd.read_parquet(config.PROCESSED_DIR / "metrics_summary.parquet", engine="pyarrow")
    decision_log = pd.read_parquet(config.PROCESSED_DIR / "rerank_decision_log.parquet",
                                   engine="pyarrow")
    block_log = pd.read_parquet(config.PROCESSED_DIR / "rerank_block_log.parquet", engine="pyarrow")

    fe = pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")
    pop = ArticlePopularityModel().fit(fe)
    cf = ArticleItemCF(popularity_model=pop).fit(fe, verbose=False)
    content = ContentModel(popularity_model=pop).fit(articles, fe, article_order=cf.article_ids)
    mf = MFModel(popularity_model=pop).fit(fe, article_order=cf.article_ids)
    w5 = json.loads((config.PROCESSED_DIR / "hybrid_weights_exp5.json").read_text())
    article_ids = cf.article_ids
    aidx = content.article_index
    warm_set = (set(content.customer_articles) & set(cf.customer_articles)
                & set(mf.customer_index))

    # Article-fact lookup (human readable).
    af = articles.set_index("article_id")
    facts = {a: {"name": str(af.at[a, "prod_name"]), "type": str(af.at[a, "product_type_name"]),
                 "colour": str(af.at[a, "colour_group_name"]),
                 "dept": str(af.at[a, "department_name"]),
                 "group": str(af.at[a, "product_group_name"])}
             for a in article_ids if a in af.index}

    # Re-ranker (product type + headness, same construction as Phase 3b).
    product_type = np.array([facts.get(a, {}).get("type") for a in article_ids], dtype=object)
    n_cat = len(article_ids)
    pop_score = np.zeros(n_cat)
    prank = {a: i for i, a in enumerate(pop.ranked_articles)}
    for i, a in enumerate(article_ids):
        pop_score[i] = 1.0 - prank.get(a, n_cat - 1) / (n_cat - 1)
    rr = ReRanker(article_ids, product_type, pop_score)

    # -- Cohort --
    rng = np.random.default_rng(config.SEED)
    picks, seg = select_cohort(ctx, evaluable, divergent, warm_set, rng)
    handles = {cid: f"C{i + 1:02d}" for i, cid in enumerate(picks)}
    print(f"Selected {len(picks)} demo customers: "
          f"{ {s: sum(1 for v in seg.values() if v == s) for s in set(seg.values())} }")

    # -- Score all warm demo customers in one batch --
    warm_demo = [c for c in picks if c in warm_set]
    rows_norm = {}
    if warm_demo:
        Cn, Fn, Mn = _norm_rows(content, cf, mf, warm_demo)
        for j, c in enumerate(warm_demo):
            rows_norm[c] = (Cn[j], Fn[j], Mn[j])

    pop_top = list(pop.ranked_articles[:K])
    ctx_by = {r.customer_id: r for r in ctx[ctx["customer_id"].isin(picks)].itertuples(index=False)}
    el_by = {c: g for c, g in el[el["customer_id"].isin(picks)].groupby("customer_id", sort=False)}
    cutoff = pd.Timestamp(config.CUTOFF_DATE)

    customers_json, recs_json, comp_for_expl = [], {}, {}
    for cid in picks:
        h = handles[cid]
        lab = set(label_sets.get(cid, []))
        # --- profile + history (pre-cutoff only) ---
        r = ctx_by[cid]
        hist_ev = el_by.get(cid)
        hist_items, dom = [], []
        if hist_ev is not None:
            hist_ev = hist_ev[hist_ev["t_dat"] < cutoff].sort_values("t_dat", ascending=False)
            tcounts = hist_ev["article_id"].map(lambda a: facts.get(a, {}).get("type"))
            dom = [t for t in tcounts.value_counts().index.tolist()[:3] if t]
            for row in hist_ev.head(15).itertuples(index=False):
                f = facts.get(row.article_id, {})
                hist_items.append({"aid": row.article_id, "name": f.get("name"),
                                   "type": f.get("type"), "colour": f.get("colour")})
        gt = [{"aid": a, "name": facts.get(a, {}).get("name"),
               "type": facts.get(a, {}).get("type")} for a in sorted(lab)][:20]
        customers_json.append({
            "id": h, "cid": cid, "seg": seg[cid],
            "profile": {"age_band": str(r.age_band), "club": str(r.club_member_status),
                        "cold": bool(r.is_cold_start), "freq": int(r.frequency),
                        "recency_days": _r(r.recency_days, 0),
                        "distinct_types": int(r.distinct_actions), "dominant_types": dom},
            "history": hist_items,
            "ground_truth": {"n": len(lab), "articles": gt},
        })

        # --- recommendations: 5 variants ---
        def _pack(art_list, comp_row):
            out = []
            for rank, a in enumerate(art_list):
                f = facts.get(a, {})
                sc = None
                if comp_row is not None and a in aidx:
                    k = aidx[a]
                    sc = {"c": _r(comp_row[0][k]), "cf": _r(comp_row[1][k]), "mf": _r(comp_row[2][k])}
                out.append({"aid": a, "name": f.get("name"), "type": f.get("type"),
                            "colour": f.get("colour"), "dept": f.get("dept"),
                            "sc": sc, "rank": rank, "hit": a in lab})
            return out

        if cid in rows_norm:
            cr, fr, mr = rows_norm[cid]
            blend = w5["w1"] * cr + w5["w2"] * fr + w5["w3"] * mr
            hyb = [article_ids[i] for i in _topk(blend, K)]
            mfr = [article_ids[i] for i in _topk(mr, K)]
            cont = [article_ids[i] for i in _topk(cr, K)]
            # re-rank over blend top-N candidates (frontier settings, constraints off)
            cand = _topk(blend, N_RETRIEVE)
            M = content.item_matrix[cand]
            sim = (M @ M.T).toarray()
            sel_div, _ = rr.rerank(cand, blend[cand], sim, K, DIV[0], DIV[1])
            sel_cov, _ = rr.rerank(cand, blend[cand], sim, K, COV[0], COV[1])
            rdiv = [article_ids[i] for i in sel_div]
            rcov = [article_ids[i] for i in sel_cov]
            comp_row = (cr, fr, mr)
            recs_json[h] = {"hybrid": _pack(hyb, comp_row), "mf": _pack(mfr, comp_row),
                            "content": _pack(cont, comp_row),
                            "rerank_div": _pack(rdiv, comp_row), "rerank_cov": _pack(rcov, comp_row)}
            top_art = hyb[0]
            k0 = aidx.get(top_art)
            if k0 is not None:
                comp_for_expl[(cid, top_art)] = {"content": float(cr[k0]), "cf": float(fr[k0]),
                                                 "mf": float(mr[k0])}
        else:  # cold-start: honest popularity fallback across all variants
            packed = _pack(pop_top, None)
            recs_json[h] = {k: [dict(x) for x in packed] for k in
                            ["hybrid", "mf", "content", "rerank_div", "rerank_cov"]}
            comp_for_expl[(cid, pop_top[0])] = None

    # --- Explanations (Task 4) ---
    store = A.EvidenceStore(articles, el, decision_log, block_log,
                            component_scores={k: v for k, v in comp_for_expl.items() if v})
    gate = RuleGate(articles, el)
    soft = SoftVerifier()
    gen = A.TemplateGenerator()
    expl_json = {}
    adv_targets = set(picks[:4])   # a few customers get an adversarial showcase
    for cid in picks:
        h = handles[cid]
        top_art = recs_json[h]["hybrid"][0]["aid"]
        row = explain_and_verify(store, cid, top_art, gen, gate, soft)
        bundle = json.loads(row["evidence_bundle"])
        entry = {"top_article": top_art, "why": row["explanation_text"],
                 "bundle": _slim_bundle(bundle),
                 "fidelity": "passed" if not row["blocked"] else "blocked",
                 "block_reason": row["block_reason"]}
        if cid in adv_targets:
            entry["adversarial"] = _adversarial_example(store, cid, top_art, gate, soft)
        expl_json[h] = entry

    # --- Frontier (Task 5) ---
    frontier_json = [{"lambda": _r(r["lambda"], 2), "pop": _r(r["pop_penalty"], 2),
                      "recall12": _r(r["recall@12"], 5), "cov12": _r(r["coverage@12"], 2),
                      "hit12": _r(r["hit@12"], 5), "gini": _r(r["gini"], 3),
                      "dissim": _r(r["intra_list_dissim"], 3),
                      "mean_pop_rank": _r(r["mean_pop_rank"], 0),
                      "distinct_types": _r(r["distinct_types"], 2)}
                     for _, r in frontier.iterrows()]

    # --- Diagnostics (Task 5) ---
    recall_by = dict(zip(msum["model"], msum["recall@12"]))
    diag_json = []
    for _, r in diag.iterrows():
        rec12 = recall_by.get(_RECALL_MAP.get(r["model"]))
        diag_json.append({"model": r["model"], "hit12": _r(r["hit@12"], 5),
                          "recall12": _r(rec12, 5) if rec12 is not None else None,
                          "cov12": _r(r["coverage_pct@12"], 2),
                          "mean_pop_rank": _r(r["mean_pop_rank"], 0), "gini": _r(r["gini"], 3),
                          "dissim": _r(r["intra_list_dissim"], 3),
                          "distinct_types": _r(r["avg_distinct_types"], 2)})

    # --- Write + validate ---
    files = {"customers.json": customers_json, "recommendations.json": recs_json,
             "explanations.json": expl_json, "frontier.json": frontier_json,
             "diagnostics.json": diag_json}
    sizes = {}
    for name, obj in files.items():
        p = DEMO_DIR / name
        p.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        json.loads(p.read_text())          # validate it parses
        sizes[name] = p.stat().st_size

    _print_summary(picks, seg, handles, sizes, recs_json, expl_json)
    _write_report(picks, seg, handles, sizes, recs_json)
    print("\nDONE.")


def _slim_bundle(bundle):
    """Compact the evidence bundle for the browser (drop verbose id lists)."""
    b = dict(bundle)
    h = dict(b["customer_history"])
    h.pop("purchased_article_ids", None)   # not needed by the demo UI
    b["customer_history"] = h
    return b


def _adversarial_example(store, cid, article_id, gate, soft):
    """A pre-computed 'watch it refuse to lie' block for the showcase."""
    facts = store.get_article_facts(article_id) or {}
    true_colour = facts.get("colour_group_name") or "black"
    wrong = "purple" if true_colour != "purple" else "orange"
    advgen = _ScriptedGenerator(
        f"This is a {wrong} item and you bought it many times.",
        [{"text": f"This is a {wrong} item.", "type": "attribute",
          "field": "colour_group_name", "value": wrong}])
    row = explain_and_verify(store, cid, article_id, advgen, gate, soft, regenerate=False)
    rg = json.loads(row["rule_gate_result"])
    viol = rg["violations"][0] if rg["violations"] else None
    return {"attempted_claim": f"This is a {wrong} item.", "blocked": bool(row["blocked"]),
            "gate": "rule" if row["block_reason"] == "rule_violation" else row["block_reason"],
            "violated_field": viol["field"] if viol else None,
            "true_value": viol["true_value"] if viol else None}


def _print_summary(picks, seg, handles, sizes, recs_json, expl_json):
    print("\n" + "=" * 66)
    print("DEMO COHORT")
    print("=" * 66)
    for cid in picks:
        h = handles[cid]
        top = recs_json[h]["hybrid"][0]
        print(f"  {h} [{seg[cid]:11s}] top-rec: {top['colour']} {top['type']} "
              f"hit={top['hit']}  why={expl_json[h]['why'][:60]}…")
    print("\nFile sizes:")
    tot = 0
    for n, s in sizes.items():
        print(f"  {n:22s} {s / 1024:7.1f} KB")
        tot += s
    print(f"  {'TOTAL':22s} {tot / 1024:7.1f} KB")


def _write_report(picks, seg, handles, sizes, recs_json):
    tot_kb = sum(sizes.values()) / 1024
    counts = {s: sum(1 for v in seg.values() if v == s) for s in set(seg.values())}
    cohort_rows = "\n".join(
        f"| {handles[cid]} | {seg[cid]} |" for cid in picks)
    size_rows = "\n".join(f"| `{n}` | {s / 1024:.1f} KB |" for n, s in sizes.items())

    md = f"""# Phase 5a — Demo Data Export (for the Lovable web app)

A browser app can't run the pipeline or read parquet, so this phase pre-computes
everything the interactive demo shows and writes it to **`data/demo/`** as compact
JSON (committed — total **{tot_kb:.0f} KB**). No new models were trained; this reads
existing artifacts (Exp 5 models, Phase 3b re-ranker + frontier, Phase 4 explainer,
`customer_context`, `event_log`, `articles`, `labels_article`).

## Cohort ({len(picks)} customers, seed {config.SEED})

Selected from the **evaluable** set (customers with label-window ground truth) to
showcase variety: {counts}. Buckets:
- **high_freq** — warm customers with rich pre-cutoff history (top-decile frequency).
- **mid_history** — warm, middle-of-the-distribution frequency.
- **divergent** — warm customers whose taste is far from popularity (lowest
  `cosine_to_popularity` from `divergent_customers_exp3`); these best show the
  model beating a popularity baseline.
- **cold_start** — zero pre-cutoff history; every model falls back to popularity
  (the demo shows this honestly). **Data caveat:** in this dataset no zero-history
  customer has any label-window purchase (`cold_start ∩ evaluable = ∅`), so these
  customers carry an **empty `ground_truth`**. They are included specifically to
  demonstrate the popularity fallback, not to score hits — stated here rather than
  hidden.

Customers are anonymized to handles `C01…C{len(picks):02d}`; the same handle keys
every file.

| handle | segment |
|---|---|
{cohort_rows}

## File schemas (exact field names for the Lovable prompt)

All floats are rounded; keys are short. Handles (`C01`…) are the join key.

### `customers.json` — list of customer objects
```
{{ id, cid, seg,          # id = anonymized handle (frontend key); cid = source hash (join/test only)
   profile: {{ age_band, club, cold(bool), freq(int), recency_days,
              distinct_types(int), dominant_types:[str] }},
   history: [ {{ aid, name, type, colour }} ]            # top ~15, pre-cutoff, recent-first
   ground_truth: {{ n(int), articles:[ {{ aid, name, type }} ] }}  # LABEL WINDOW
}}
```
`history`/`profile` are **pre-cutoff** (`t_dat < {config.CUTOFF_DATE}`);
`ground_truth` is the **label window**. They never mix (leakage guard).

### `recommendations.json` — `{{ handle: {{ variant: [item x12] }} }}`
Five variants per customer: **`hybrid`** (triple hybrid, production), **`mf`**
(MF alone), **`content`** (content alone), **`rerank_div`** (re-ranked λ={DIV[0]},
pop={DIV[1]} — the diversity setting), **`rerank_cov`** (re-ranked λ={COV[0]},
pop={COV[1]} — max-coverage). Each item:
```
{{ aid, name, type, colour, dept,
   sc: {{ c, cf, mf }} | null,   # normalized triple-hybrid component scores
   rank(int 0-11), hit(bool) }}   # hit = article is in this customer's ground truth
```
`sc` is `null` for cold-start customers (not warm in the sub-models); all five
variants then equal the popularity fallback. Powers the **model toggle** and
**diversity slider**.

### `explanations.json` — `{{ handle: {{ ... }} }}`
```
{{ top_article, why(str), fidelity("passed"|"blocked"), block_reason,
   bundle: {{ ...evidence bundle (article_facts, customer_history summary,
              recommendation_context w/ component scores, constraint_decisions) }},
   adversarial?: {{ attempted_claim, blocked(bool), gate, violated_field, true_value }} }}
```
The Phase-4 grounded "why" + the evidence bundle it drew on + the hard-gate
fidelity result. A few customers carry a pre-computed **adversarial** block (a
false colour claim the rule gate rejects) for the "watch it refuse to lie" panel.

### `frontier.json` — list of `(lambda, pop)` operating points
```
{{ lambda, pop, recall12, cov12, hit12, gini, dissim, mean_pop_rank, distinct_types }}
```
Straight from `rerank_frontier.parquet` (24 points). Lets the slider map to real
recall-vs-coverage points and plot the tradeoff curve.

### `diagnostics.json` — headline per-model diagnostics
```
{{ model, hit12, recall12, cov12, mean_pop_rank, gini, dissim, distinct_types }}
```
From `diagnostics_results.parquet` (+ `recall@12` joined from `metrics_summary`).
The summary panel: **accuracy and coverage rank in opposite orders** — MF tops
pop-rank/accuracy tradeoffs but covers <1% of the catalog; neighborhood CF covers
78% but is least accurate.

## File sizes

| file | size |
|---|---|
{size_rows}
| **total** | **{tot_kb:.1f} KB** |

## Notes / honesty

- Recommendation lists mirror the production retrieval construction (blend of the
  min-max-normalized content/CF/MF scores; re-rank over the blend's top-{N_RETRIEVE}),
  so the demo's numbers are the real pipeline's, not a re-derivation.
- Cold-start customers legitimately collapse to the popularity fallback across all
  five variants — the demo does not hide this.
- Explanations are generated by the Phase-4 **offline** faithful generator and pass
  the deterministic rule gate; they describe the evidence, not the model's internal
  causal mechanism (see Phase 4 limitations).
"""
    REPORT_PATH.write_text(md)


if __name__ == "__main__":
    main()
