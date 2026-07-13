"""Beyond-accuracy recommender diagnostics — the failure modes hit-rate is blind to.

Pure functions over a recommendation set ``recs`` (dict: customer_id -> ordered
list of recommended article_ids) plus supporting structures. Covers popularity
bias, catalog coverage, intra-list diversity, cold-start scoring, and fairness by
segment. The runner assembles the inputs and calls these.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Concentration / popularity bias
# --------------------------------------------------------------------------
def gini(counts) -> float:
    """Gini coefficient of a non-negative array (0 = perfectly even, ->1 = all mass
    on one item). Computed over recommendation frequency across the catalog."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x) / (n * x.sum())) - (n + 1.0) / n)


def popularity_ranks(ranked_article_ids):
    """Map article_id -> popularity rank (1 = most popular pre-cutoff)."""
    return {a: i + 1 for i, a in enumerate(ranked_article_ids)}


def popularity_bias(recs, pop_rank, n_ranked, top1_cut, top10_cut) -> dict:
    """Mean/median popularity rank of recs, % in top-1%/top-10%, and Gini of
    recommendation frequency. Unranked (cold) articles get rank n_ranked+1."""
    ranks, counts = [], {}
    n_top1 = n_top10 = n_total = 0
    for arts in recs.values():
        for a in arts:
            r = pop_rank.get(a, n_ranked + 1)
            ranks.append(r)
            counts[a] = counts.get(a, 0) + 1
            n_total += 1
            if r <= top1_cut:
                n_top1 += 1
            if r <= top10_cut:
                n_top10 += 1
    ranks = np.array(ranks)
    freq = np.zeros(n_ranked, dtype=np.float64)
    for a, c in counts.items():
        r = pop_rank.get(a)
        if r is not None:
            freq[r - 1] = c
    return {
        "mean_pop_rank": float(ranks.mean()) if len(ranks) else float("nan"),
        "median_pop_rank": float(np.median(ranks)) if len(ranks) else float("nan"),
        "pct_top1": 100.0 * n_top1 / n_total if n_total else 0.0,
        "pct_top10": 100.0 * n_top10 / n_total if n_total else 0.0,
        "gini": gini(freq),
    }


def label_mean_pop_rank(label_sets, pop_rank, n_ranked) -> float:
    """Mean popularity rank of what customers ACTUALLY bought (the demand baseline)."""
    ranks = [pop_rank.get(a, n_ranked + 1) for s in label_sets.values() for a in s]
    return float(np.mean(ranks)) if ranks else float("nan")


# --------------------------------------------------------------------------
# Catalog coverage
# --------------------------------------------------------------------------
def coverage(recs, n_catalog, k, top10_set) -> dict:
    """Catalog coverage, aggregate diversity, and long-tail share."""
    recommended = set()
    n_total = long_tail = 0
    for arts in recs.values():
        topk = arts[:k]
        recommended.update(topk)
        for a in topk:
            n_total += 1
            if a not in top10_set:
                long_tail += 1
    n_cust = len(recs)
    return {
        "coverage_count": len(recommended),
        "coverage_pct": 100.0 * len(recommended) / n_catalog if n_catalog else 0.0,
        "aggregate_diversity": len(recommended) / (n_cust * k) if n_cust and k else 0.0,
        "long_tail_share_pct": 100.0 * long_tail / n_total if n_total else 0.0,
    }


# --------------------------------------------------------------------------
# Intra-list diversity
# --------------------------------------------------------------------------
def intra_list_diversity(recs, item_matrix, article_index, product_type, k,
                         sample=None, seed=42) -> dict:
    """Mean pairwise content dissimilarity (1 - cosine) within each top-k list, and
    the average number of distinct product types per list. ``item_matrix`` rows are
    L2-normalized so cosine = dot product."""
    cust_ids = list(recs)
    if sample and len(cust_ids) > sample:
        rng = np.random.default_rng(seed)
        cust_ids = list(rng.choice(cust_ids, sample, replace=False))
    dissims, distinct_types = [], []
    for c in cust_ids:
        arts = [a for a in recs[c][:k] if a in article_index]
        if len(arts) < 2:
            continue
        idx = [article_index[a] for a in arts]
        M = item_matrix[idx]
        sim = (M @ M.T).toarray()
        m = len(arts)
        iu = np.triu_indices(m, k=1)
        dissims.append(float(np.mean(1.0 - sim[iu])))
        distinct_types.append(len({product_type.get(a) for a in arts}))
    return {
        "intra_list_dissimilarity": float(np.mean(dissims)) if dissims else float("nan"),
        "avg_distinct_types": float(np.mean(distinct_types)) if distinct_types else float("nan"),
    }


# --------------------------------------------------------------------------
# Fairness by segment
# --------------------------------------------------------------------------
def hit_at_k(recs, label_sets, k) -> float:
    if not label_sets:
        return float("nan")
    hits = 0
    for c, labels in label_sets.items():
        if set(recs.get(c, [])[:k]) & labels:
            hits += 1
    return hits / len(label_sets)


def hit_by_segment(recs, label_sets, segment_of, k) -> dict:
    """hit@k within each segment. ``segment_of`` maps customer_id -> segment label.
    Every evaluated customer is counted in exactly one segment."""
    groups = {}
    for c in label_sets:
        seg = segment_of.get(c, "unknown")
        groups.setdefault(seg, {})[c] = label_sets[c]
    out = {seg: {"n": len(g), "hit": hit_at_k(recs, g, k)} for seg, g in groups.items()}
    hits = [v["hit"] for v in out.values() if v["n"] > 0]
    out["_spread"] = (max(hits) - min(hits)) if hits else 0.0
    return out


def quartile_labels(values, names=("Q1", "Q2", "Q3", "Q4")):
    """Assign each value to a quartile label (Q1 = lowest). Returns a list aligned
    to ``values``; ties handled by rank so groups are ~equal size."""
    v = np.asarray(values, dtype=np.float64)
    ranks = v.argsort().argsort()
    edges = (ranks / max(len(v), 1) * 4).astype(int).clip(0, 3)
    return [names[e] for e in edges]
