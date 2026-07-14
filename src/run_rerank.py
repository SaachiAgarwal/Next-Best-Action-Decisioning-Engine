"""Phase 3b runner: diversity + constraint re-ranking, the accuracy-coverage frontier.

Run with:  python -m src.run_rerank
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.eval import metrics, diagnostics as dg
from src.models.popularity_article import ArticlePopularityModel
from src.models.item_cf_article import ArticleItemCF
from src.models.content_based_exp4 import ContentModel, _row_minmax
from src.models.hybrid_content_cf_exp4 import cf_score_chunk
from src.models.mf_exp5 import MFModel
from src.rerank.reranker import ReRanker, _minmax

LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
POPPEN = [0.0, 0.1, 0.3, 0.5]
K = 12
FRONTIER_PATH = config.PROCESSED_DIR / "rerank_frontier.parquet"
LOG_PATH = config.PROCESSED_DIR / "rerank_decision_log.parquet"
BLOCK_PATH = config.PROCESSED_DIR / "rerank_block_log.parquet"


def _fe():
    return pd.read_parquet(config.PROCESSED_DIR / "features_events.parquet", engine="pyarrow")


def build():
    fe = _fe()
    articles = pd.read_parquet(config.PROCESSED_DIR / "articles.parquet", engine="pyarrow")
    articles["article_id"] = articles["article_id"].astype("string")
    la = pd.read_parquet(config.PROCESSED_DIR / "labels_article.parquet", engine="pyarrow")
    label_sets = {c: set(g) for c, g in la.groupby("customer_id", sort=False)["article_id"]}
    ev = list(label_sets)
    ctx = pd.read_parquet(config.PROCESSED_DIR / "customer_context.parquet", engine="pyarrow")

    pop = ArticlePopularityModel().fit(fe)
    cf = ArticleItemCF(popularity_model=pop).fit(fe, verbose=False)
    content = ContentModel(popularity_model=pop).fit(articles, fe, article_order=cf.article_ids)
    mf = MFModel(popularity_model=pop).fit(fe, article_order=cf.article_ids)
    w5 = json.loads((config.PROCESSED_DIR / "hybrid_weights_exp5.json").read_text())

    article_ids = cf.article_ids
    n_catalog = len(article_ids)
    product_type = np.array([None] * n_catalog, dtype=object)
    ptmap = dict(zip(articles["article_id"], articles["product_type_name"]))
    for i, a in enumerate(article_ids):
        product_type[i] = ptmap.get(a)
    # Headness in [0,1]: 1 = most popular.
    pop_score = np.zeros(n_catalog)
    rank = {a: i for i, a in enumerate(pop.ranked_articles)}
    for i, a in enumerate(article_ids):
        r = rank.get(a, n_catalog - 1)
        pop_score[i] = 1.0 - r / (n_catalog - 1)
    pop_rank = dg.popularity_ranks(pop.ranked_articles)

    # Stage-1 retrieval: top-N candidates per customer + relevance (triple blend).
    aidx = content.article_index
    N = config.N_RETRIEVE
    n_ev = len(ev)
    cand_idx = np.zeros((n_ev, N), dtype=np.int64)
    cand_rel = np.zeros((n_ev, N), dtype=np.float32)
    row_of = {c: i for i, c in enumerate(ev)}
    for s in range(0, n_ev, 1000):
        chunk = ev[s:s + 1000]
        warm = [c for c in chunk if c in content.customer_articles and c in cf.customer_articles and c in mf.customer_index]
        if not warm:
            continue
        Cn = _row_minmax(content.score_chunk(warm)); Fn = _row_minmax(cf_score_chunk(cf, warm))
        Mn = _row_minmax(mf.score_chunk(warm))
        blend = w5["w1"] * Cn + w5["w2"] * Fn + w5["w3"] * Mn
        for j, c in enumerate(warm):
            top = np.argpartition(-blend[j], N - 1)[:N]
            top = top[np.argsort(-blend[j][top])]
            gi = row_of[c]
            cand_idx[gi] = top
            cand_rel[gi] = blend[j][top]

    # Fatigue product types (bought within FATIGUE_DAYS pre-cutoff) + simulated OOS.
    fatigue = _fatigue_types(set(ev))
    rng = np.random.default_rng(config.SEED)
    oos = set(int(i) for i in rng.choice(n_catalog, int(0.05 * n_catalog), replace=False))

    seg_maps = _seg_maps(ev, ctx)
    return {
        "ev": ev, "row_of": row_of, "cand_idx": cand_idx, "cand_rel": cand_rel,
        "content": content, "article_ids": article_ids, "product_type": product_type,
        "pop_score": pop_score, "pop_rank": pop_rank, "pop": pop, "n_catalog": n_catalog,
        "label_sets": label_sets, "seg_maps": seg_maps, "fatigue": fatigue, "oos": oos,
        "ptmap": ptmap,
    }


def _fatigue_types(ev_set):
    el = pd.read_parquet(config.PROCESSED_DIR / "event_log.parquet",
                         columns=["customer_id", "t_dat", "action_id"], engine="pyarrow")
    # action_id is the product-type action; map to name via actions.parquet.
    actions = pd.read_parquet(config.PROCESSED_DIR / "actions.parquet", engine="pyarrow")
    aname = dict(zip(actions["action_id"], actions["product_type_name"]))
    cutoff = pd.Timestamp(config.CUTOFF_DATE)
    lo = cutoff - pd.Timedelta(days=config.FATIGUE_DAYS)
    recent = el[(el["t_dat"] >= lo) & (el["t_dat"] < cutoff) & (el["customer_id"].isin(ev_set))]
    out = {}
    for c, g in recent.groupby("customer_id", sort=False)["action_id"]:
        out[c] = {aname.get(a) for a in g}
    return out


def _seg_maps(ev, ctx):
    c = ctx[ctx["customer_id"].isin(set(ev))].copy()
    fq = dg.quartile_labels(c["frequency"].to_numpy(), ("low", "mid", "high", "top"))
    rq = dg.quartile_labels(-c["recency_days"].fillna(c["recency_days"].max()).to_numpy(),
                            ("lapsed", "cooling", "warm", "active"))
    cid = c["customer_id"].tolist()
    return {"age_band": dict(zip(cid, c["age_band"].astype("string"))),
            "club": dict(zip(cid, c["club_member_status"].astype("string"))),
            "freq_q": dict(zip(cid, fq)), "rec_q": dict(zip(cid, rq))}


def _evaluate(recs, D, top10_set, top1_cut, top10_cut):
    ls = D["label_sets"]
    acc = metrics.evaluate(recs, ls, ks=[6, 12, 24])
    a12 = acc[acc["k"] == 12].iloc[0]
    cov = dg.coverage(recs, D["n_catalog"], 12, top10_set)
    bias = dg.popularity_bias({c: v[:12] for c, v in recs.items()}, D["pop_rank"], D["n_catalog"], top1_cut, top10_cut)
    div = dg.intra_list_diversity(recs, D["content"].item_matrix, D["content"].article_index,
                                  {a: D["ptmap"].get(a) for a in D["article_ids"]}, 12, sample=3000)
    spread = max(dg.hit_by_segment(recs, ls, sm, 12)["_spread"] for sm in D["seg_maps"].values())
    return {"hit@6": float(acc[acc["k"] == 6]["hit_rate"].iloc[0]), "hit@12": float(a12["hit_rate"]),
            "hit@24": float(acc[acc["k"] == 24]["hit_rate"].iloc[0]),
            "recall@12": float(a12["recall"]), "precision@12": float(a12["precision"]),
            "coverage@12": cov["coverage_pct"], "mean_pop_rank": bias["mean_pop_rank"],
            "pct_top10": bias["pct_top10"], "gini": bias["gini"],
            "intra_list_dissim": div["intra_list_dissimilarity"],
            "distinct_types": div["avg_distinct_types"], "seg_spread": spread}


def main():
    D = build()
    ev, ci, cr = D["ev"], D["cand_idx"], D["cand_rel"]
    content, article_ids = D["content"], D["article_ids"]
    rr = ReRanker(article_ids, D["product_type"], D["pop_score"])
    top1_cut, top10_cut = max(1, D["n_catalog"] // 100), max(1, D["n_catalog"] // 10)
    top10_set = set(D["pop"].ranked_articles[:top10_cut])
    print(f"Two-stage re-rank: {len(ev):,} customers, retrieve N={config.N_RETRIEVE}, select k={K}")

    # --- FRONTIER (MMR + pop penalty; constraints OFF for a clean tradeoff curve) ---
    configs = [(lam, pp) for lam in LAMBDAS for pp in POPPEN]
    recs_by = {cfg: {} for cfg in configs}
    for gi, c in enumerate(ev):
        cand = ci[gi]
        M = content.item_matrix[cand]
        sim = (M @ M.T).toarray()
        for cfg in configs:
            sel, _ = rr.rerank(cand, cr[gi], sim, K, cfg[0], cfg[1])
            recs_by[cfg][c] = [str(article_ids[i]) for i in sel]

    frontier = []
    for cfg in configs:
        m = _evaluate(recs_by[cfg], D, top10_set, top1_cut, top10_cut)
        m.update({"lambda": cfg[0], "pop_penalty": cfg[1]})
        frontier.append(m)
    fdf = pd.DataFrame(frontier)
    fdf.to_parquet(FRONTIER_PATH, engine="pyarrow")

    baseline = fdf[(fdf["lambda"] == 1.0) & (fdf["pop_penalty"] == 0.0)].iloc[0]
    print(f"\nBaseline (λ=1, pop=0 = triple hybrid): recall@12={baseline['recall@12']:.4f} "
          f"coverage={baseline['coverage@12']:.1f}% gini={baseline['gini']:.3f} "
          f"dissim={baseline['intra_list_dissim']:.3f}")

    # --- Recommended operating point: max coverage within 15% recall loss ---
    ok = fdf[fdf["recall@12"] >= 0.85 * baseline["recall@12"]]
    rec = ok.loc[ok["coverage@12"].idxmax()]
    lam_star, pp_star = float(rec["lambda"]), float(rec["pop_penalty"])
    print(f"Recommended operating point: λ={lam_star}, pop_penalty={pp_star} "
          f"(max coverage within 15% recall loss)")

    # --- Operating point WITH business constraints + audit ---
    op_recs, blocks, log_rows, block_counts = _apply_operating_point(D, rr, lam_star, pp_star)
    op = _evaluate(op_recs, D, top10_set, top1_cut, top10_cut)
    _save_audit(log_rows, blocks)

    _print_frontier(fdf, baseline)
    print(f"\nConstraint blocks (operating point): {block_counts}")
    _write_report(fdf, baseline, rec, op, block_counts, D, lam_star, pp_star)
    print("\nDONE.")


def _apply_operating_point(D, rr, lam, pp):
    ev, ci, cr = D["ev"], D["cand_idx"], D["cand_rel"]
    content, article_ids = D["content"], D["article_ids"]
    op_recs, log_rows, block_rows = {}, [], []
    block_counts = {"fatigue": 0, "category_cap": 0, "out_of_stock": 0}
    cap = config.CATEGORY_CAP
    for gi, c in enumerate(ev):
        cand = ci[gi]
        M = content.item_matrix[cand]
        sim = (M @ M.T).toarray()
        fat = D["fatigue"].get(c, frozenset())
        sel, blocks = rr.rerank(cand, cr[gi], sim, K, lam, pp,
                                fatigue_types=fat, cap=cap, oos=D["oos"])
        op_recs[c] = [str(article_ids[i]) for i in sel]
        rel = _minmax(cr[gi])
        for pos, ai in enumerate(sel):
            local = int(np.where(cand == ai)[0][0])
            log_rows.append({"customer_id": c, "article_id": str(article_ids[ai]),
                             "stage1_relevance": round(float(rel[local]), 5),
                             "mmr_score": round(float(rel[local] - pp * D["pop_score"][ai]), 5),
                             "popularity_rank": int(D["pop_rank"].get(str(article_ids[ai]), D["n_catalog"])),
                             "final_position": pos})
        for ai, rule in blocks:
            block_counts[rule] += 1
            block_rows.append({"customer_id": c, "article_id": str(article_ids[ai]), "rule_violated": rule})
    return op_recs, block_rows, log_rows, block_counts


def _save_audit(log_rows, block_rows):
    ld = pd.DataFrame(log_rows); ld["customer_id"] = ld["customer_id"].astype("string")
    ld.to_parquet(LOG_PATH, engine="pyarrow")
    bd = pd.DataFrame(block_rows)
    if len(bd):
        bd["customer_id"] = bd["customer_id"].astype("string")
    bd.to_parquet(BLOCK_PATH, engine="pyarrow")


def _print_frontier(fdf, baseline):
    print("\n" + "=" * 88)
    print("FRONTIER — recall@12 vs coverage@12 (each λ × pop_penalty)")
    print("=" * 88)
    piv = fdf.pivot_table(index="lambda", columns="pop_penalty", values="coverage@12")
    print("coverage@12 (%):"); print(piv.to_string(float_format=lambda v: f"{v:.1f}"))
    pivr = fdf.pivot_table(index="lambda", columns="pop_penalty", values="recall@12")
    print("\nrecall@12:"); print(pivr.to_string(float_format=lambda v: f"{v:.4f}"))


def _write_report(fdf, baseline, rec, op, block_counts, D, lam, pp):
    path = config.REPORTS_DIR / "phase3b_reranker.md"
    b = baseline

    def d(col):
        return op[col] - b[col]
    # Frontier table rows sorted by coverage.
    fr = fdf.sort_values(["lambda", "pop_penalty"])
    frows = "\n".join(
        f"| {r['lambda']} | {r['pop_penalty']} | {r['recall@12']:.4f} | {r['hit@12']:.4f} | "
        f"{r['coverage@12']:.1f}% | {r['mean_pop_rank']:,.0f} | {r['gini']:.3f} | {r['intra_list_dissim']:.3f} |"
        for _, r in fr.iterrows())

    # Tuning (frontier, no constraints) at the operating point, and +constraints (op).
    t = rec  # frontier row for (lam, pp): the clean accuracy-coverage tuning
    tune_loss = 100 * (t["recall@12"] - b["recall@12"]) / b["recall@12"]
    tune_cov_gain = t["coverage@12"] / b["coverage@12"]
    op_loss = 100 * (op["recall@12"] - b["recall@12"]) / b["recall@12"]
    cov_word = "rises" if t["coverage@12"] > b["coverage@12"] + 0.1 else "is roughly flat"
    fair_word = ("MORE equitable" if op["seg_spread"] < b["seg_spread"] - 1e-4
                 else "LESS equitable" if op["seg_spread"] > b["seg_spread"] + 1e-4 else "no more/less equitable")

    content_md = f"""# Phase 3b — Diversity + Constraint Re-Ranking

## The pathology this fixes (Phase 3a)

The production triple hybrid covers only **40.6%** of the catalog, draws **69%** of
recommendations from the top-10% head (Gini **0.886**), has intra-list diversity
**0.453**, and serves at mean popularity rank **~10,015** while customers actually
buy at rank **~35,279**. Accuracy metrics are blind to all of this. This layer
trades a controlled slice of accuracy to fix it.

## Two-stage architecture

**Stage 1 (retrieval):** the triple hybrid scores and returns the top-{config.N_RETRIEVE}
candidates per customer — "what is relevant". **Stage 2 (re-rank):** selects the
final {K} — "what should we actually show". Different questions: relevance is
necessary but not sufficient; the shown list must also be diverse, expose
inventory, and obey business rules. Separating them lets us tune the second
without touching the first.

## MMR, and why diversity ≠ coverage

MMR builds the list greedily, each pick maximizing
`λ·rel(i) − (1−λ)·max_{{j∈selected}} sim(i,j)` — trading relevance against
similarity to what's already chosen (cosine in the Exp 4 content space). λ=1 is
pure relevance (reproduces stage-1); λ→0 is pure diversity.

But **MMR alone cannot fix coverage**: it makes each *list* varied, yet everyone
could still get a varied list from the same popular head. So we add an explicit
**popularity penalty** — `adj_rel(i) = rel(i) − POP_PENALTY·pop_score(i)` — that
pushes the whole system into the long tail. λ controls *within-list* diversity;
POP_PENALTY controls *catalog* coverage. Both are needed.

## Business constraints (the decisioning layer)

Applied as hard filters at the operating point, each logged with a reason:
- **fatigue:** drop a product type the customer bought within {config.FATIGUE_DAYS}
  days (the measured repurchase cadence) — a restock nudge days after purchase is
  wasted contact.
- **category cap:** at most {config.CATEGORY_CAP} of one product type in the final
  {K} — fixes the "12 near-identical items" failure (diagnostics found content
  lists averaged 1.4 distinct types).
- **eligibility:** drop out-of-stock articles. **⚠️ Inventory is SIMULATED** — a
  seeded random 5% marked out-of-stock; the dataset has no real stock data. This
  demonstrates the mechanism only.

## THE FRONTIER (the central deliverable)

Each row is one (λ, POP_PENALTY) setting; constraints off to isolate the
accuracy-coverage tradeoff. Baseline = **λ=1, pop=0** (the unmodified triple hybrid).

| λ | pop_pen | recall@12 | hit@12 | coverage@12 | mean pop rank | Gini | intra-list dissim |
|---|---|---|---|---|---|---|---|
{frows}

**Reading the frontier:** moving down/right (lower λ, higher POP_PENALTY) trades
recall for coverage, tail depth (higher mean pop rank), lower Gini, and higher
list diversity. Accuracy loss is **real, not free**.

## Recommended operating point — λ={lam}, POP_PENALTY={pp}

Chosen by a **product rule, not metric-maximization**: on the frontier (tuning
only), *maximize catalog coverage subject to ≤15% recall loss.* Two effects are
separated below: the **tuning** (MMR + pop-penalty, the accuracy-coverage trade)
and the **business constraints** applied on top (which enforce rules at a further
accuracy cost).

| metric | baseline (triple hybrid) | tuning only (λ={lam}, pop={pp}) | + business constraints |
|---|---|---|---|
| recall@12 | {b['recall@12']:.4f} | {t['recall@12']:.4f} ({tune_loss:+.1f}%) | {op['recall@12']:.4f} ({op_loss:+.1f}%) |
| hit@12 | {b['hit@12']:.4f} | {t['hit@12']:.4f} | {op['hit@12']:.4f} |
| coverage@12 | {b['coverage@12']:.1f}% | {t['coverage@12']:.1f}% ({tune_cov_gain:.2f}×) | {op['coverage@12']:.1f}% |
| mean pop rank | {b['mean_pop_rank']:,.0f} | {t['mean_pop_rank']:,.0f} | {op['mean_pop_rank']:,.0f} |
| top-10% head share | {b['pct_top10']:.1f}% | {t['pct_top10']:.1f}% | {op['pct_top10']:.1f}% |
| Gini | {b['gini']:.3f} | {t['gini']:.3f} | {op['gini']:.3f} |
| intra-list dissimilarity | {b['intra_list_dissim']:.3f} | {t['intra_list_dissim']:.3f} | {op['intra_list_dissim']:.3f} |
| distinct types / list | {b['distinct_types']:.1f} | {t['distinct_types']:.1f} | {op['distinct_types']:.1f} |
| segment fairness spread | {b['seg_spread']:.4f} | {t['seg_spread']:.4f} | {op['seg_spread']:.4f} |

**Plain language — the tuning:** at λ={lam}, POP_PENALTY={pp} (no constraints),
recall goes {b['recall@12']:.4f}→{t['recall@12']:.4f} ({tune_loss:+.1f}%) while
coverage {cov_word} {b['coverage@12']:.1f}%→{t['coverage@12']:.1f}%
({tune_cov_gain:.2f}×), Gini {b['gini']:.3f}→{t['gini']:.3f}, and lists carry
{t['distinct_types']:.1f} distinct product types (up from {b['distinct_types']:.1f}).
This is a **real, deliberate trade** — recall for discovery/diversity — that
hit-rate alone would never surface.

**The honest catch — coverage is retrieval-bounded.** Re-ranking only reorders each
customer's fixed top-{config.N_RETRIEVE} candidates, so it **cannot** reach
neighborhood CF's 77.8% coverage; the achievable gain here is modest
({b['coverage@12']:.1f}%→{t['coverage@12']:.1f}%). Broadening coverage further
requires a **more diverse retriever**, not a smarter re-ranker — a limitation the
frontier makes explicit.

**Adding the business constraints** (fatigue + category cap + simulated OOS) costs
*more* accuracy (recall {t['recall@12']:.4f}→{op['recall@12']:.4f}, now {op_loss:+.1f}%
vs baseline) and does not raise article coverage (they filter, not broaden), but
they sharply improve **list quality and fairness**: distinct types per list rise to
{op['distinct_types']:.1f}, intra-list dissimilarity to {op['intra_list_dissim']:.3f},
head share falls to {op['pct_top10']:.1f}%, and the **segment fairness spread**
goes {b['seg_spread']:.4f}→{op['seg_spread']:.4f} — re-ranking makes the model
**{fair_word}** across customer segments. Whether the large accuracy cost is worth
the rule-compliance and fairness gains is a genuine product call; this report gives
the numbers to make it, not a verdict.

## Constraint blocks (operating point)

Across all customers, the hard filters blocked: **fatigue {block_counts['fatigue']:,}**,
**category cap {block_counts['category_cap']:,}**, **out-of-stock (simulated)
{block_counts['out_of_stock']:,}** candidates. Every block is logged with its
`rule_violated` in `rerank_block_log.parquet` — the Responsible-AI audit trail.

## Honest limitations

- **Re-ranking cannot exceed the retrieval ceiling** — it only reorders/filters the
  top-{config.N_RETRIEVE}; it cannot surface a relevant article stage-1 missed.
- **The accuracy loss is real**, not free (tuning {tune_loss:+.1f}% recall,
  {op_loss:+.1f}% with constraints) — a deliberate product trade, not a Pareto
  improvement.
- **Inventory is simulated** (seeded 5% OOS) — the mechanism is real, the stock
  data is not.
- **Fatigue is a heuristic** (the measured 12-day cadence), not a learned policy.
"""
    path.write_text(content_md)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
