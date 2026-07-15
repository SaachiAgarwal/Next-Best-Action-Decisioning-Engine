"""Two-stage fidelity guard for generated explanations (Phase 4).

Stage 1 — RULE-BASED GATE (``RuleGate``): HARD, deterministic, no LLM. Every
ATTRIBUTE / DESCRIPTOR / HISTORY claim is checked against the *actual* source of
truth (``articles.parquet`` for attributes, the customer's pre-cutoff event log
for history). Any mismatch is a HARD BLOCK, logged with the exact violated field
and the true value. Because it is pure comparison against ground truth, this gate
gives a **100% guarantee on factual attribute claims** — it is the ONLY hard
guarantee in the system.

Stage 2 — LLM SOFT VERIFIER (``SoftVerifier``): a SEPARATE LLM call that sees only
the explanation + the evidence bundle and returns SUPPORTED / UNSUPPORTED per
claim. It catches what rules cannot: unsupported *reasoning* ("perfect for summer
parties", "you'll love this") — claims that are not factually false but have no
grounding in the evidence. Because the verifier is itself an LLM, it is a **SOFT**
control: it reduces unsupported claims, it does not guarantee their absence. Do
not present it as a guarantee.

Policy on failure: UNSUPPORTED or a rule violation ⇒ the explanation is BLOCKED
(rejected) and logged with a reason. The runner attempts a single regeneration;
a still-failing explanation stays blocked. (We block rather than silently ship —
a wrong "why" is worse than no "why".)
"""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.explain.agent import ARTICLE_FIELDS

# Categorical article fields a claim may assert an exact value for.
_CATEGORICAL = [
    "product_type_name", "product_group_name", "colour_group_name",
    "department_name", "graphical_appearance_name",
]


def _norm(x):
    return str(x).strip().lower() if x is not None else ""


# ---------------------------------------------------------------------------
# Stage 1 — deterministic rule gate
# ---------------------------------------------------------------------------
class RuleGate:
    """Deterministic claim checker. Reads ground truth directly, never the bundle.

    Constructed with ``articles`` and ``event_log`` dataframes so the guarantee is
    against the *source*, not against a bundle that could in principle be stale.
    """

    def __init__(self, articles, event_log, cutoff_date=None):
        a = articles.copy()
        a["article_id"] = a["article_id"].astype("string")
        self.articles = a.set_index("article_id")
        self.cutoff = pd.Timestamp(cutoff_date or config.CUTOFF_DATE)
        el = event_log.copy()
        el["article_id"] = el["article_id"].astype("string")
        el["customer_id"] = el["customer_id"].astype("string")
        el["t_dat"] = pd.to_datetime(el["t_dat"])
        self.events = el[el["t_dat"] < self.cutoff].reset_index(drop=True)
        self._ptype = self.articles["product_type_name"]

    def _article_text(self, article_id):
        """Concatenated ground-truth text of an article (for descriptor terms)."""
        if article_id not in self.articles.index:
            return ""
        row = self.articles.loc[article_id]
        vals = [row[f] for f in ARTICLE_FIELDS + ["garment_group_name", "index_name",
                                                   "perceived_colour_master_name"]
                if f in self.articles.columns]
        return " ".join(_norm(v) for v in vals if not pd.isna(v))

    def check(self, bundle: dict, claims: list) -> dict:
        """Return {passed: bool, violations: [...]}. A violation names the field
        and the true value, so the block is fully auditable."""
        article_id = str(bundle["article_id"])
        customer_id = str(bundle["customer_id"])
        facts = self.get_article_facts_row(article_id)
        violations = []

        for c in claims or []:
            ctype = c.get("type")
            if ctype == "attribute":
                v = self._check_attribute(article_id, facts, c)
            elif ctype == "descriptor":
                v = self._check_descriptor(article_id, c)
            elif ctype == "history":
                v = self._check_history(customer_id, c)
            else:
                v = None  # recommendation / reasoning: not a rule-gate concern
            if v is not None:
                violations.append(v)

        return {"passed": len(violations) == 0, "violations": violations}

    def get_article_facts_row(self, article_id):
        if article_id not in self.articles.index:
            return {}
        row = self.articles.loc[article_id]
        return {f: (None if pd.isna(row[f]) else str(row[f])) for f in ARTICLE_FIELDS}

    def _check_attribute(self, article_id, facts, c):
        field, value = c.get("field"), c.get("value")
        if not value:
            return None
        if field == "detail_desc":  # free text: substring containment
            true = facts.get("detail_desc") or ""
            if _norm(value) and _norm(value) not in _norm(true):
                return {"claim": c.get("text", ""), "field": "detail_desc",
                        "claimed": value, "true_value": true,
                        "reason": "attribute_not_in_detail_desc"}
            return None
        if field in _CATEGORICAL:
            true = facts.get(field)
            if _norm(value) != _norm(true):
                return {"claim": c.get("text", ""), "field": field,
                        "claimed": value, "true_value": true,
                        "reason": "attribute_mismatch"}
            return None
        # Unknown field with a value: verify the value appears somewhere real.
        if _norm(value) not in self._article_text(article_id):
            return {"claim": c.get("text", ""), "field": field or "unknown",
                    "claimed": value, "true_value": None,
                    "reason": "attribute_unverifiable"}
        return None

    def _check_descriptor(self, article_id, c):
        term = c.get("term") or c.get("value")
        if not term:
            return None
        if _norm(term) not in self._article_text(article_id):
            return {"claim": c.get("text", ""), "field": "material/descriptor",
                    "claimed": term, "true_value": None,
                    "reason": "descriptor_not_in_article"}
        return None

    def _check_history(self, customer_id, c):
        ev = self.events[self.events["customer_id"] == customer_id]
        ptype = c.get("product_type")
        window = c.get("window_days")
        claimed = c.get("count")
        if ptype:
            sub = ev[ev["article_id"].map(self._ptype).map(_norm) == _norm(ptype)]
            if window:
                lo = self.cutoff - pd.Timedelta(days=int(window))
                sub = sub[sub["t_dat"] >= lo]
            actual = int(len(sub))
            if actual == 0:
                return {"claim": c.get("text", ""), "field": "customer_history",
                        "claimed": f"{ptype} x{claimed}", "true_value": "never purchased",
                        "reason": "history_purchase_not_found"}
            if claimed is not None and int(claimed) > actual:
                return {"claim": c.get("text", ""), "field": "customer_history",
                        "claimed": f"{ptype} x{claimed}", "true_value": f"{ptype} x{actual}",
                        "reason": "history_count_overstated"}
        return None


# ---------------------------------------------------------------------------
# Stage 2 — LLM soft verifier
# ---------------------------------------------------------------------------
SOFT_SYSTEM = (
    "You are a strict fidelity verifier. You are given a generated EXPLANATION and "
    "the EVIDENCE BUNDLE it was supposed to be grounded in — and NOTHING ELSE. For "
    "each claim in the explanation, decide whether it is SUPPORTED by a specific "
    "field in the evidence bundle or UNSUPPORTED (no grounding — e.g. speculation "
    "about occasions, feelings, lifestyle, or any attribute not in the bundle). "
    "Respond with STRICT JSON: {\"verdicts\": [{\"claim\": str, \"verdict\": "
    "\"SUPPORTED\"|\"UNSUPPORTED\", \"evidence_field\": str}]}. Use evidence_field "
    "\"none\" for UNSUPPORTED claims. Emit nothing outside the JSON."
)

# Speculative markers the offline heuristic verifier treats as unsupported reasoning.
_SPECULATIVE = [
    "perfect for", "you'll love", "you will love", "great for", "ideal for",
    "summer", "winter", "party", "parties", "occasion", "vacation", "holiday",
    "feel", "style", "trend", "must-have", "wardrobe staple", "everyone", "lifestyle",
    "work", "office", "date night", "weekend",
]


class SoftVerifier:
    """LLM soft verifier. Sees ONLY the explanation + bundle (Task 5, Task 9).

    ``verify_fn(explanation, bundle) -> verdicts`` is injectable; the default is a
    deterministic offline heuristic that stands in for the LLM when no API key is
    present (and that tests replace with recorded responses). The heuristic flags
    ``reasoning``-type claims and any claim text containing speculative markers —
    an approximation of what the LLM verifier is for, clearly labelled as such.
    """

    def __init__(self, verify_fn=None):
        self.verify_fn = verify_fn or heuristic_soft_verify

    def verify(self, explanation_text: str, bundle: dict, claims: list) -> list:
        # NOTE: only explanation_text + bundle cross the boundary — claims are
        # passed for the offline heuristic's convenience but the LLM path (below)
        # is given the explanation + bundle only.
        return self.verify_fn(explanation_text, bundle, claims)

    def any_unsupported(self, verdicts) -> bool:
        return any(str(v.get("verdict")).upper() == "UNSUPPORTED" for v in verdicts)


def heuristic_soft_verify(explanation_text, bundle, claims):
    """Offline stand-in for the LLM verifier: flag speculative / ungrounded claims."""
    verdicts = []
    for c in claims or []:
        text = _norm(c.get("text"))
        spec = any(m in text for m in _SPECULATIVE)
        is_reasoning = c.get("type") == "reasoning"
        if spec or is_reasoning:
            verdicts.append({"claim": c.get("text", ""), "verdict": "UNSUPPORTED",
                             "evidence_field": "none"})
        else:
            verdicts.append({"claim": c.get("text", ""), "verdict": "SUPPORTED",
                             "evidence_field": c.get("field") or c.get("type") or "bundle"})
    # Also scan the prose itself for speculation the claim list may have omitted.
    if any(m in _norm(explanation_text) for m in _SPECULATIVE) and not any(
            str(v["verdict"]).upper() == "UNSUPPORTED" for v in verdicts):
        verdicts.append({"claim": explanation_text, "verdict": "UNSUPPORTED",
                         "evidence_field": "none"})
    return verdicts


class LLMSoftVerifier:
    """Real Anthropic soft verifier (claude-sonnet-4-6). Lazily imported."""

    name = "claude-sonnet-4-6"

    def __init__(self, model="claude-sonnet-4-6", client=None, max_tokens=1024):
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        else:
            import anthropic
            self.client = anthropic.Anthropic()

    def __call__(self, explanation_text, bundle, claims):
        # Boundary: send ONLY the explanation and the bundle (no claims, no extra).
        user = ("EXPLANATION:\n" + explanation_text + "\n\nEVIDENCE BUNDLE:\n"
                + json.dumps(bundle, indent=2, default=str))
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SOFT_SYSTEM,
            messages=[{"role": "user", "content": user}])
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        return parse_verdicts(text)


def parse_verdicts(text: str) -> list:
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    lo, hi = s.find("{"), s.rfind("}")
    if lo != -1 and hi != -1:
        try:
            return list(json.loads(s[lo:hi + 1]).get("verdicts", []))
        except json.JSONDecodeError:
            pass
    return []
