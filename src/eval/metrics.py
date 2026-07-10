"""Reusable ranking-evaluation metrics for the NBA Decisioning Engine.

Every model (popularity baseline, collaborative filtering, learned rankers)
plugs into this harness so results are directly comparable. Metrics are the
standard top-k recommendation measures, evaluated per customer and averaged over
the evaluable set.

Inputs (shared shape for all metrics):
    recommendations : dict[customer_id -> ordered list of action_ids]
                      (most-relevant first; may be longer than k)
    labels          : dict[customer_id -> set of action_ids the customer
                      actually purchased in the label window]

Customers are evaluated over the keys of ``labels`` — i.e. only the evaluable
set (customers with >=1 label action). A customer missing from
``recommendations`` is treated as having received an empty list.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_KS = [6, 12, 24]


def _topk(recommendations, customer_id, k):
    """Top-k recommended action_ids for a customer (order preserved, deduped)."""
    recs = recommendations.get(customer_id, [])
    seen, out = set(), []
    for a in recs:
        if a not in seen:
            seen.add(a)
            out.append(a)
        if len(out) == k:
            break
    return out


def hit_rate_at_k(recommendations, labels, k) -> float:
    """Mean over customers of 1 if any top-k rec is in the label set, else 0."""
    if not labels:
        return 0.0
    hits = 0
    for cid, label_set in labels.items():
        topk = _topk(recommendations, cid, k)
        if any(a in label_set for a in topk):
            hits += 1
    return hits / len(labels)


def recall_at_k(recommendations, labels, k) -> float:
    """Mean over customers of (label actions found in top-k) / (all label actions)."""
    if not labels:
        return 0.0
    total = 0.0
    for cid, label_set in labels.items():
        if not label_set:
            continue
        topk = set(_topk(recommendations, cid, k))
        total += len(topk & label_set) / len(label_set)
    return total / len(labels)


def precision_at_k(recommendations, labels, k) -> float:
    """Mean over customers of (label actions found in top-k) / k."""
    if not labels:
        return 0.0
    total = 0.0
    for cid, label_set in labels.items():
        topk = set(_topk(recommendations, cid, k))
        total += len(topk & label_set) / k
    return total / len(labels)


def evaluate(recommendations, labels, ks=None) -> pd.DataFrame:
    """Evaluate all three metrics at each k. Returns a tidy results table."""
    ks = list(ks) if ks is not None else list(DEFAULT_KS)
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "hit_rate": hit_rate_at_k(recommendations, labels, k),
            "recall": recall_at_k(recommendations, labels, k),
            "precision": precision_at_k(recommendations, labels, k),
        })
    return pd.DataFrame(rows)
