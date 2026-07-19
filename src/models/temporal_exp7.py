"""Experiment 7 — temporal signals on top of the Exp 5 triple hybrid.

Three signal families, none explored before:

**A. Trend popularity** — article demand over only the last ``TREND_WINDOW_DAYS``
   before the reference date, vs all-time popularity. ``momentum`` = recent share
   ÷ all-time share (rising/falling), offered as an alternative to raw trend.

**B. Seasonality** — each article's historical month distribution (two years of
   pre-cutoff history), smoothed toward the global month distribution so thin
   articles don't spike. ``season_score`` = the article's propensity for the
   label-window month (September).

**C. Purchase timing ("due-ness")** — per customer, ``due_ratio =
   days_since_last / typical_gap`` where ``typical_gap`` is the MEDIAN
   inter-purchase gap. Used three ways: a ranking modifier (3a), an analysis
   dimension (3b), and — the higher-value use — a CONTACT-TIMING decision that
   assigns each customer a band (3c).

The ``TemporalHybrid`` adds ``w4*trend + w5*season`` to the triple blend. With
``w4=w5=0`` and the due modifier off it reproduces the triple hybrid EXACTLY — the
key regression guard. All signals are built from pre-cutoff data only (leakage
guard); at tuning time the reference date is the internal validation cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.models.content_based_exp4 import _row_minmax


# ---------------------------------------------------------------------------
# Signals A + B: article-level trend, momentum, season (aligned to article_ids)
# ---------------------------------------------------------------------------
class TemporalSignals:
    """Per-article temporal priors, aligned to a fixed ``article_ids`` order."""

    def __init__(self, article_ids):
        self.article_ids = np.asarray(article_ids)
        self.index = {a: i for i, a in enumerate(self.article_ids)}
        self.n = len(self.article_ids)
        self.trend_norm = None      # min-max recent-window popularity in [0,1]
        self.momentum_norm = None   # min-max (recent share / all-time share)
        self.season_norm = None     # min-max propensity for LABEL_MONTH
        self.pop_norm = None        # min-max all-time popularity (for due modifier)
        self.recent_counts = None
        self.alltime_counts = None
        self.article_month = None   # (n, 12) smoothed month distribution
        self.global_month = None    # (12,) global month distribution

    def _counts(self, ev):
        c = np.zeros(self.n, dtype=np.float64)
        g = ev["article_id"].map(self.index)
        vc = g.dropna().astype(int).value_counts()
        c[vc.index.to_numpy()] = vc.to_numpy()
        return c

    def fit(self, feature_events, reference_date,
            trend_window_days=None, label_month=None, smooth_k=None):
        ref = pd.Timestamp(reference_date)
        tw = trend_window_days or config.TREND_WINDOW_DAYS
        lm = label_month or config.LABEL_MONTH
        K = config.SEASON_SMOOTH_K if smooth_k is None else smooth_k
        ev = feature_events[feature_events["t_dat"] < ref]     # leakage guard
        ev = ev[["article_id", "t_dat"]].copy()
        ev["article_id"] = ev["article_id"].astype("string")

        # --- Signal A: trend + momentum ---
        recent = ev[ev["t_dat"] >= ref - pd.Timedelta(days=tw)]
        self.recent_counts = self._counts(recent)
        self.alltime_counts = self._counts(ev)
        self.trend_norm = _row_minmax(self.recent_counts.reshape(1, -1)).ravel()
        rt, at = self.recent_counts.sum(), self.alltime_counts.sum()
        recent_share = self.recent_counts / (rt + 1e-9)
        alltime_share = self.alltime_counts / (at + 1e-9)
        momentum = recent_share / (alltime_share + 1e-9)      # >1 rising, <1 falling
        # articles absent all-time can't have momentum; keep finite
        momentum = np.where(self.alltime_counts > 0, momentum, 0.0)
        self.momentum_norm = _row_minmax(momentum.reshape(1, -1)).ravel()

        # --- Signal B: seasonal month profile, smoothed toward the global shape ---
        months = ev["t_dat"].dt.month.to_numpy()
        aidx = ev["article_id"].map(self.index).to_numpy()
        keep = ~np.isnan(aidx)
        aidx = aidx[keep].astype(int)
        months = months[keep]
        M = np.zeros((self.n, 12), dtype=np.float64)
        np.add.at(M, (aidx, months - 1), 1.0)
        global_counts = M.sum(axis=0)
        self.global_month = global_counts / (global_counts.sum() + 1e-9)
        totals = M.sum(axis=1, keepdims=True)
        # profile = (article_month + K*global) / (total + K)  -> shrinks thin articles
        self.article_month = (M + K * self.global_month[None, :]) / (totals + K)
        season_score = self.article_month[:, lm - 1]
        self.season_norm = _row_minmax(season_score.reshape(1, -1)).ravel()

        self.pop_norm = _row_minmax(self.alltime_counts.reshape(1, -1)).ravel()
        return self

    def top_overlap(self, k=50):
        """Signal-A pre-model diagnostic: |recent top-k ∩ all-time top-k| / k."""
        r = set(np.argsort(-self.recent_counts)[:k])
        a = set(np.argsort(-self.alltime_counts)[:k])
        return len(r & a) / k

    def month_skew_by_type(self, product_type, month):
        """Per product-type lift for a month vs the global share of that month.

        Returns a Series indexed by product_type_name of (type month share /
        global month share). >1 = the type skews toward that month.
        """
        pt = np.array([product_type.get(a) for a in self.article_ids], dtype=object)
        counts = self.alltime_counts
        # month totals per type via the article_month profile weighted by counts
        raw = self.article_month * counts[:, None]     # (n,12) purchase mass by month
        df = pd.DataFrame(raw)
        df["pt"] = pt
        agg = df.dropna(subset=["pt"]).groupby("pt").sum()
        type_share = agg[month - 1] / (agg.sum(axis=1) + 1e-9)
        global_share = self.global_month[month - 1]
        return (type_share / (global_share + 1e-9)).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Signal C: purchase-timing "due-ness" + contact bands
# ---------------------------------------------------------------------------
def compute_due(event_log, evaluable_ids, reference_date, population_gap=None):
    """Per-customer due-ness from pre-cutoff events.

    typical_gap = MEDIAN inter-purchase gap between distinct purchase dates.
    due_ratio   = days_since_last / typical_gap.
    Single-purchase customers have no gap -> fall back to the population median
    gap and are flagged (``single_purchase``).
    Returns a DataFrame: customer_id, days_since_last, typical_gap, due_ratio,
    single_purchase, band.
    """
    ref = pd.Timestamp(reference_date)
    ev = event_log[(event_log["t_dat"] < ref)
                   & (event_log["customer_id"].isin(set(evaluable_ids)))]
    ev = ev[["customer_id", "t_dat"]].copy()

    # distinct purchase dates per customer (same-day multi-buys are one occasion)
    days = (ev.groupby("customer_id")["t_dat"]
            .apply(lambda s: np.sort(np.unique(s.values))))
    rows = []
    gaps_all = []
    for cid, arr in days.items():
        arr = pd.to_datetime(arr)
        last = arr[-1]
        dsl = (ref - last).days
        if len(arr) >= 2:
            g = np.diff(arr).astype("timedelta64[D]").astype(int)
            tg = float(np.median(g))
            single = False
            gaps_all.append(tg)
        else:
            tg = np.nan            # fill after we know the population median
            single = True
        rows.append((cid, dsl, tg, single))

    pop_gap = population_gap if population_gap is not None else (
        float(np.median(gaps_all)) if gaps_all else 30.0)
    out = []
    for cid, dsl, tg, single in rows:
        gap = pop_gap if (single or not np.isfinite(tg) or tg <= 0) else tg
        due = dsl / gap
        out.append({"customer_id": cid, "days_since_last": int(dsl),
                    "typical_gap": round(float(gap), 3), "due_ratio": round(float(due), 4),
                    "single_purchase": bool(single), "band": assign_band(due)})
    df = pd.DataFrame(out)
    df.attrs["population_gap"] = pop_gap
    return df


def assign_band(due_ratio, bands=None):
    for name, lo, hi in (bands or config.CONTACT_BANDS):
        if lo <= due_ratio < hi:
            return name
    return (bands or config.CONTACT_BANDS)[-1][0]   # due_ratio == inf edge


# ---------------------------------------------------------------------------
# The temporal-augmented hybrid
# ---------------------------------------------------------------------------
def topk_dense(score_row, article_ids, bought_idx, k, include_repeats, pop_ranked):
    from src.models.content_based_exp4 import topk_dense as _t
    return _t(score_row, article_ids, bought_idx, k, include_repeats, pop_ranked)


class TemporalHybrid:
    """triple blend + w4*trend + w5*season, with an optional due-ratio modifier.

    ``w4=w5=0`` and ``due_modifier=False`` -> identical to the wrapped TripleHybrid.
    """

    def __init__(self, triple, signals, due_ratios=None, trend_field="trend_norm"):
        self.triple = triple
        self.signals = signals
        self.article_ids = triple.article_ids
        self.popularity_model = triple.popularity_model
        self.content = triple.content
        self.due = due_ratios or {}
        self.trend_field = trend_field       # "trend_norm" or "momentum_norm"

    def _warm(self, customer_ids):
        return self.triple._warm(customer_ids)

    def blended_chunk(self, warm_ids, w1, w2, w3, w4=0.0, w5=0.0, due_modifier=False):
        S = self.triple.blended_chunk(warm_ids, w1, w2, w3)
        if w4:
            S = S + w4 * getattr(self.signals, self.trend_field)[None, :]
        if w5:
            S = S + w5 * self.signals.season_norm[None, :]
        if due_modifier:
            S = self._apply_due(S, warm_ids)
        return S

    def _apply_due(self, S, warm_ids):
        """Due customers lean on personalization; just-purchased lean on popularity."""
        due = np.array([self.due.get(c, 1.0) for c in warm_ids], dtype=np.float64)
        pers = np.clip(due, 0.0, 2.0) / 2.0            # 0..1
        w_pers = (0.5 + 0.5 * pers)[:, None]           # 0.5..1.0
        Sn = _row_minmax(S)
        return w_pers * Sn + (1.0 - w_pers) * self.signals.pop_norm[None, :]

    def recommend_all(self, customer_ids, k=24, w1=1.0, w2=0.5, w3=1.0, w4=0.0, w5=0.0,
                      due_modifier=False, include_repeats=True, batch_size=1000):
        customer_ids = list(customer_ids)
        warm = self._warm(customer_ids)
        warm_set = set(warm)
        recs = {}
        pop_topk = self.popularity_model.recommend(k=k)
        for c in customer_ids:
            if c not in warm_set:
                recs[c] = list(pop_topk)
        pop_ranked = self.popularity_model.ranked_articles
        for s in range(0, len(warm), batch_size):
            chunk = warm[s:s + batch_size]
            S = self.blended_chunk(chunk, w1, w2, w3, w4, w5, due_modifier)
            for c, row in zip(chunk, S):
                bought = self.content.customer_articles[c][0]
                recs[c] = topk_dense(row, self.article_ids, bought, k, include_repeats, pop_ranked)
        return recs
