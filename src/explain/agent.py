"""Agentic RAG explanation layer — evidence tools, bundle, and generation (Phase 4).

What makes this *agentic* rather than "an LLM writing a sentence": the agent does
not free-associate over a recommendation. It (1) GATHERS evidence by calling
discrete tools that each read one real source, (2) assembles a structured
EVIDENCE BUNDLE that is the *only* thing generation may draw on, (3) GENERATES a
grounded "why", and (4) hands the result to a two-stage fidelity guard
(``verifier.py``) that self-checks every claim. Every step is logged — auditable.

This module owns steps 1-3. Verification is in ``src/explain/verifier.py``.

Leakage guard: every tool reads **pre-cutoff** data only (``t_dat < CUTOFF_DATE``).
The recommendation is made "as of" the cutoff; an explanation that cited a
post-cutoff purchase would be using the future to justify the past.

LLM access is **injectable**. The agent takes a ``generator`` object with a
``.generate(bundle) -> {"explanation", "claims"}`` method:
  - ``LLMGenerator``       — the real Anthropic API path (claude-sonnet-4-6).
  - ``TemplateGenerator``  — a deterministic, offline, faithful-by-construction
                             generator used when no API key is available and as
                             the substrate tests build on. It only ever emits
                             facts already in the bundle, so it is the honest
                             floor, not a stand-in for model quality.
Tests inject their own mock generators, so the suite makes **no live API calls**.
"""

from __future__ import annotations

import json
import os
from datetime import date

import numpy as np
import pandas as pd

from src import config

# Article-fact fields exposed to the explanation (the ground-truth attributes).
ARTICLE_FIELDS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "graphical_appearance_name",
    "detail_desc",
]


# ---------------------------------------------------------------------------
# Evidence store + the four tools
# ---------------------------------------------------------------------------
class EvidenceStore:
    """Backs the four evidence tools. Reads only real, pre-cutoff sources.

    Constructed from dataframes so tests can pass tiny synthetic frames and the
    runner can pass the real parquets. ``component_scores`` (optional) maps
    ``(customer_id, article_id) -> {"content","cf","mf"}`` — the triple-hybrid
    component breakdown the runner computes from the models; when absent the
    recommendation context still carries the persisted blended score.
    """

    def __init__(self, articles, event_log, decision_log, block_log,
                 cutoff_date=None, component_scores=None):
        self.cutoff = pd.Timestamp(cutoff_date or config.CUTOFF_DATE)
        # Articles indexed by id (strings, leading zeros preserved).
        a = articles.copy()
        a["article_id"] = a["article_id"].astype("string")
        self.articles = a.set_index("article_id")
        # Event log — PRE-CUTOFF ONLY (the leakage guard, enforced once here).
        el = event_log.copy()
        el["article_id"] = el["article_id"].astype("string")
        el["customer_id"] = el["customer_id"].astype("string")
        el["t_dat"] = pd.to_datetime(el["t_dat"])
        self.events = el[el["t_dat"] < self.cutoff].reset_index(drop=True)
        # Product type / colour per article, for history summarisation.
        self._ptype = self.articles["product_type_name"]
        self._colour = self.articles["colour_group_name"]
        # Decision + block logs, grouped by customer for O(1) lookup.
        dl = decision_log.copy()
        dl["customer_id"] = dl["customer_id"].astype("string")
        dl["article_id"] = dl["article_id"].astype("string")
        self._dl_by_cust = {c: g for c, g in dl.groupby("customer_id", sort=False)}
        self._dl_key = {(r.customer_id, r.article_id): r for r in dl.itertuples(index=False)}
        if len(block_log):
            bl = block_log.copy()
            bl["customer_id"] = bl["customer_id"].astype("string")
            bl["article_id"] = bl["article_id"].astype("string")
            self._bl_by_cust = {c: g for c, g in bl.groupby("customer_id", sort=False)}
        else:
            self._bl_by_cust = {}
        self.component_scores = component_scores or {}

    # -- Tool 1 ------------------------------------------------------------
    def get_customer_history(self, customer_id):
        """Pre-cutoff purchase summary: top types (recency-weighted), colours, dates."""
        customer_id = str(customer_id)
        ev = self.events[self.events["customer_id"] == customer_id]
        if ev.empty:
            return {"total_purchases": 0, "distinct_product_types": 0,
                    "last_purchase_date": None, "top_product_types": [],
                    "colours_bought": [], "purchased_article_ids": []}
        ptype = ev["article_id"].map(self._ptype)
        colour = ev["article_id"].map(self._colour)
        # Recency weight: half-life decay measured back from the cutoff date.
        age = (self.cutoff - ev["t_dat"]).dt.days.to_numpy().astype(float)
        w = 0.5 ** (age / config.HALF_LIFE_DAYS)
        agg = {}
        for pt, wi in zip(ptype.tolist(), w.tolist()):
            if pt is None or (isinstance(pt, float) and np.isnan(pt)):
                continue
            d = agg.setdefault(pt, {"product_type": pt, "count": 0, "recency_weighted": 0.0})
            d["count"] += 1
            d["recency_weighted"] += wi
        top = sorted(agg.values(), key=lambda d: d["recency_weighted"], reverse=True)[:5]
        for d in top:
            d["recency_weighted"] = round(d["recency_weighted"], 4)
        colours = [c for c in colour.dropna().tolist()]
        colour_counts = pd.Series(colours).value_counts() if colours else pd.Series(dtype=int)
        return {
            "total_purchases": int(len(ev)),
            "distinct_product_types": int(ptype.dropna().nunique()),
            "last_purchase_date": ev["t_dat"].max().date().isoformat(),
            "top_product_types": top,
            "colours_bought": colour_counts.index.tolist()[:6],
            "purchased_article_ids": ev["article_id"].tolist(),
        }

    # -- Tool 2 ------------------------------------------------------------
    def get_article_facts(self, article_id):
        """The article's REAL record from articles.parquet (ground truth)."""
        article_id = str(article_id)
        if article_id not in self.articles.index:
            return None
        row = self.articles.loc[article_id]
        out = {"article_id": article_id}
        for f in ARTICLE_FIELDS:
            v = row[f]
            if pd.isna(v):
                out[f] = None
            else:
                out[f] = str(v)
        return out

    # -- Tool 3 ------------------------------------------------------------
    def get_recommendation_context(self, customer_id, article_id):
        """Why the model ranked it: blended relevance, MMR, rank, position, components."""
        customer_id, article_id = str(customer_id), str(article_id)
        r = self._dl_key.get((customer_id, article_id))
        ctx = {"found": r is not None}
        if r is not None:
            ctx.update({
                "final_position": int(r.final_position),
                "stage1_relevance": round(float(r.stage1_relevance), 5),
                "mmr_score": round(float(r.mmr_score), 5),
                "popularity_rank": int(r.popularity_rank),
            })
        comp = self.component_scores.get((customer_id, article_id))
        ctx["component_scores"] = (
            {k: round(float(v), 5) for k, v in comp.items()} if comp else None)
        return ctx

    # -- Tool 4 ------------------------------------------------------------
    def get_constraint_decisions(self, customer_id):
        """From the rerank BLOCK log: which candidates were blocked, and why."""
        customer_id = str(customer_id)
        g = self._bl_by_cust.get(customer_id)
        counts = {"fatigue": 0, "category_cap": 0, "out_of_stock": 0}
        examples = []
        if g is not None and len(g):
            vc = g["rule_violated"].value_counts()
            for rule in counts:
                counts[rule] = int(vc.get(rule, 0))
            for row in g.head(5).itertuples(index=False):
                examples.append({"article_id": str(row.article_id),
                                 "rule": str(row.rule_violated)})
        return {"blocked_counts": counts, "total_blocked": int(sum(counts.values())),
                "examples": examples}


# ---------------------------------------------------------------------------
# The evidence bundle (Task 2)
# ---------------------------------------------------------------------------
def build_bundle(store: EvidenceStore, customer_id, article_id) -> dict:
    """Call the four tools and assemble the structured evidence bundle.

    This dict is the ONLY thing the explanation may draw on, and the exact object
    the verifier checks against — persisting it makes each decision auditable.
    """
    customer_id, article_id = str(customer_id), str(article_id)
    hist = store.get_customer_history(customer_id)
    facts = store.get_article_facts(article_id)
    rec = store.get_recommendation_context(customer_id, article_id)
    cons = store.get_constraint_decisions(customer_id)

    hist_types = {d["product_type"] for d in hist["top_product_types"]}
    shares_type = facts is not None and facts.get("product_type_name") in hist_types
    shares_colour = facts is not None and facts.get("colour_group_name") in set(hist["colours_bought"])
    rec = dict(rec)
    rec["shares_product_type_with_history"] = bool(shares_type)
    rec["shares_colour_with_history"] = bool(shares_colour)

    return {
        "customer_id": customer_id,
        "article_id": article_id,
        "cutoff_date": store.cutoff.date().isoformat(),
        "customer_history": hist,
        "article_facts": facts,
        "recommendation_context": rec,
        "constraint_decisions": cons,
    }


# ---------------------------------------------------------------------------
# Generators (Task 3)
# ---------------------------------------------------------------------------
GEN_SYSTEM = (
    "You explain a product recommendation to a retail customer. You may state "
    "ONLY facts present in the EVIDENCE BUNDLE provided. Do not embellish, do not "
    "speculate about the customer's feelings, occasions, lifestyle, or intent, and "
    "do not mention any attribute not present in article_facts. Write 2-3 short "
    "sentences. Then decompose your explanation into discrete claims. Respond with "
    "STRICT JSON: {\"explanation\": str, \"claims\": [{\"text\": str, \"type\": "
    "\"attribute\"|\"history\"|\"recommendation\"|\"descriptor\"|\"reasoning\", "
    "\"field\": str|null, \"value\": str|null, \"product_type\": str|null, "
    "\"count\": int|null, \"window_days\": int|null, \"term\": str|null}]}. For an "
    "attribute claim set field to the article_facts field name and value to its "
    "value. For a history claim set product_type/count/window_days. Emit nothing "
    "outside the JSON."
)


def _bundle_to_prompt(bundle: dict) -> str:
    return "EVIDENCE BUNDLE (the only permissible source of facts):\n" + json.dumps(
        bundle, indent=2, default=str)


class TemplateGenerator:
    """Deterministic, offline, faithful-by-construction generator.

    Emits only facts already in the bundle, so its output passes the rule gate by
    construction. It is the honest floor when no API key is present — NOT a claim
    about model-quality parity. The prose is assembled from the same fields the
    claims cite, so prose and claims never drift.
    """

    name = "template-offline"

    def generate(self, bundle: dict) -> dict:
        facts = bundle["article_facts"] or {}
        hist = bundle["customer_history"]
        cons = bundle["constraint_decisions"]
        rec = bundle["recommendation_context"]
        claims = []
        sentences = []

        # Sentence 1 — article attributes (all grounded in article_facts).
        colour = facts.get("colour_group_name")
        ptype = facts.get("product_type_name")
        dept = facts.get("department_name")
        pieces = []
        if colour:
            pieces.append(colour)
            claims.append({"text": f"This is a {colour} item.", "type": "attribute",
                           "field": "colour_group_name", "value": colour})
        if ptype:
            pieces.append(ptype)
            claims.append({"text": f"It is a {ptype}.", "type": "attribute",
                           "field": "product_type_name", "value": ptype})
        if pieces and dept:
            sentences.append(f"This is a {' '.join(pieces)} from the {dept} department.")
            claims.append({"text": f"It is from the {dept} department.", "type": "attribute",
                           "field": "department_name", "value": dept})
        elif pieces:
            sentences.append(f"This is a {' '.join(pieces)}.")

        # Sentence 2 — history link (only if genuinely shared).
        if rec.get("shares_product_type_with_history") and ptype:
            match = next((d for d in hist["top_product_types"]
                          if d["product_type"] == ptype), None)
            cnt = match["count"] if match else 1
            sentences.append(
                f"You have purchased {ptype} {cnt} time(s) in your history before the cutoff.")
            claims.append({"text": f"You bought {ptype} {cnt} time(s).", "type": "history",
                           "product_type": ptype, "count": int(cnt), "window_days": None})
        elif rec.get("shares_colour_with_history") and colour:
            sentences.append(f"You have previously bought {colour} items.")
            claims.append({"text": f"You previously bought {colour} items.", "type": "history",
                           "product_type": None, "count": 1, "window_days": None,
                           "field": "colour_group_name", "value": colour})

        # Sentence 3 — constraint context ("why not the others").
        tb = cons.get("total_blocked", 0)
        if tb:
            parts = [f"{v} {k}" for k, v in cons["blocked_counts"].items() if v]
            sentences.append(
                f"{tb} other candidate(s) were filtered out ({', '.join(parts)}).")
            claims.append({"text": f"{tb} candidates were blocked by business rules.",
                           "type": "recommendation", "field": "total_blocked",
                           "value": str(tb)})

        if not sentences:
            sentences.append("Recommended based on your purchase history and catalog relevance.")
        return {"explanation": " ".join(sentences), "claims": claims}


class LLMGenerator:
    """Real Anthropic API generation (claude-sonnet-4-6). Lazily imported.

    Only constructed by the runner when ANTHROPIC_API_KEY is set. Kept minimal /
    single-call per explanation to stay within a modest budget.
    """

    name = "claude-sonnet-4-6"

    def __init__(self, model="claude-sonnet-4-6", client=None, max_tokens=1024):
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        else:
            import anthropic  # lazy: not a hard dependency of the module
            self.client = anthropic.Anthropic()

    def generate(self, bundle: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=GEN_SYSTEM,
            messages=[{"role": "user", "content": _bundle_to_prompt(bundle)}])
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        return parse_generation(text)


def parse_generation(text: str) -> dict:
    """Parse a generator's JSON response into {'explanation', 'claims'}.

    Tolerant of fenced code blocks / surrounding prose. On failure returns the raw
    text as the explanation with an empty claim list (the rule gate then has
    nothing structured to check — a conservative, not permissive, outcome).
    """
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    lo, hi = s.find("{"), s.rfind("}")
    if lo != -1 and hi != -1:
        try:
            obj = json.loads(s[lo:hi + 1])
            return {"explanation": str(obj.get("explanation", "")),
                    "claims": list(obj.get("claims", []))}
        except json.JSONDecodeError:
            pass
    return {"explanation": text.strip(), "claims": []}


def make_default_generator():
    """LLMGenerator if an API key + SDK are available, else the offline template."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMGenerator()
        except Exception:
            pass
    return TemplateGenerator()
