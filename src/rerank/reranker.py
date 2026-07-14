"""Diversity + constraint re-ranking layer (Phase 3b).

Stage 2 of a two-stage recommender: the triple hybrid (stage 1) answers "what is
relevant"; the re-ranker answers "what should we actually show" — trading a
controlled slice of accuracy for diversity, catalog coverage, and business-rule
compliance.

Core algorithm — **Maximal Marginal Relevance (MMR)**, built greedily:
    MMR(i) = LAMBDA · adj_rel(i)  -  (1 - LAMBDA) · max_{j∈selected} sim(i, j)
  - adj_rel(i) = rel(i) - POP_PENALTY · pop_score(i)
  - rel(i)       = stage-1 relevance, min-max normalized across the candidates
  - pop_score(i) = headness in [0,1] (1 = most popular) — the **coverage lever**
  - sim(i, j)    = cosine in the Exp 4 content space (the diversity space)
  - LAMBDA: 1 = pure relevance (reproduces stage-1 order), 0 = pure diversity

**Diversity ≠ coverage.** MMR makes each *list* varied but everyone could still get
a varied list from the same popular head. The POP_PENALTY term is what pushes the
whole system into the long tail (coverage).

Hard business constraints (the "decisioning" layer), each logged with a reason:
  - fatigue: drop candidates whose product type the customer bought within
    FATIGUE_DAYS (measured ~12-day repurchase cadence)
  - category cap: at most ``cap`` articles of one product_type in the final list
  - eligibility: drop out-of-stock articles (SIMULATED inventory)
"""

from __future__ import annotations

import numpy as np


def _minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class ReRanker:
    def __init__(self, article_ids, product_type, pop_score):
        """article_ids: index->article_id; product_type[idx]->type; pop_score[idx] in [0,1]."""
        self.article_ids = np.asarray(article_ids)
        self.product_type = product_type      # array indexed by article index
        self.pop_score = np.asarray(pop_score, dtype=np.float64)

    def rerank(self, cand_idx, cand_rel, sim, k, lam, pop_penalty,
               fatigue_types=frozenset(), cap=None, oos=frozenset()):
        """Return (selected article indices in order, block_log).

        ``cand_idx`` : article indices of the retrieved candidates (order = stage-1).
        ``cand_rel`` : stage-1 relevance scores for those candidates.
        ``sim``      : (n, n) content cosine similarity among the candidates.
        Constraints default to OFF (empty fatigue/oos, no cap) so LAMBDA=1,
        POP_PENALTY=0 reproduces the stage-1 top-k exactly.
        """
        cand_idx = np.asarray(cand_idx)
        n = len(cand_idx)
        rel = _minmax(cand_rel)
        adj = rel - pop_penalty * self.pop_score[cand_idx]
        ptypes = [self.product_type[a] for a in cand_idx]

        blocks = []
        alive = np.ones(n, dtype=bool)
        for i, a in enumerate(cand_idx):
            if a in oos:
                alive[i] = False; blocks.append((int(a), "out_of_stock"))
            elif ptypes[i] in fatigue_types:
                alive[i] = False; blocks.append((int(a), "fatigue"))

        selected = []          # positions into cand_idx
        tcount = {}
        NEG = -1e18
        while len(selected) < k:
            div = sim[:, selected].max(axis=1) if selected else np.zeros(n)
            score = lam * adj - (1.0 - lam) * div
            score[~alive] = NEG
            if selected:
                score[selected] = NEG
            # Take the best; if the cap displaces it, log and retry.
            chosen = -1
            while True:
                i = int(np.argmax(score))
                if score[i] <= NEG / 2:
                    break
                pt = ptypes[i]
                if cap is not None and tcount.get(pt, 0) >= cap:
                    blocks.append((int(cand_idx[i]), "category_cap"))
                    alive[i] = False; score[i] = NEG
                    continue
                chosen = i
                break
            if chosen < 0:
                break
            selected.append(chosen)
            tcount[ptypes[chosen]] = tcount.get(ptypes[chosen], 0) + 1

        return [int(cand_idx[i]) for i in selected], blocks

    def rerank_ids(self, cand_idx, cand_rel, sim, k, lam, pop_penalty, **kw):
        """Convenience: return selected article_ids (strings) instead of indices."""
        sel, blocks = self.rerank(cand_idx, cand_rel, sim, k, lam, pop_penalty, **kw)
        return [str(self.article_ids[i]) for i in sel], blocks
