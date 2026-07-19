"""Phase 5c — export the Exp 7 contact-timing bands as demo JSON.

Adds ONE file, ``data/demo/contact_timing.json``, so the Lovable app can show
timing as an action cue ("Contact now" / "Hold — recently purchased"). No new
computation: this reads the existing ``contact_timing_exp7.parquet`` (written by
Exp 7), the C01–C38 handle mapping from the existing ``data/demo/customers.json``,
and the per-band hit@12 figures parsed from ``reports/exp7_temporal.md``.

STANDALONE BY DESIGN: this script never touches the six existing demo exports —
it only writes the seventh file. Run with:  python -m src.run_export_contact_timing
"""

from __future__ import annotations

import json
import re

import pandas as pd

from src import config

DEMO_DIR = config.PROCESSED_DIR.parent / "demo"
OUT = DEMO_DIR / "contact_timing.json"
BAND_PARQUET = config.PROCESSED_DIR / "contact_timing_exp7.parquet"
EXP7_REPORT = config.REPORTS_DIR / "exp7_temporal.md"
REPORT_PATH = config.REPORTS_DIR / "phase5a_export.md"

FINDING = {
    "headline": "Conversion falls as time since last purchase grows — recency beats rhythm.",
    "detail": ("We expected customers 'due' at their usual cadence to convert best. They "
               "don't. The highest-converting band ('just purchased', 9.4%) is exactly the "
               "one the fatigue rule suppresses."),
    "caveat": ("The conversion metric cannot see fatigue cost — there is no unsubscribe or "
               "annoyance signal in the data — and the bands are confounded by purchase "
               "frequency (frequent buyers cluster in 'just purchased'). Resolving this needs "
               "a live experiment measuring conversion AND unsubscribes against contact recency."),
}


def _range_str(lo, hi):
    if lo == 0.0:
        return f"<{hi:g}"
    if hi == float("inf"):
        return f">{lo:g}"
    return f"{lo:g}–{hi:g}"


def _band_hit12_from_report():
    """Parse per-band hit@12 from the Exp 7 report (the source of truth)."""
    text = EXP7_REPORT.read_text()
    hits = {}
    for name, _, _ in config.CONTACT_BANDS:
        # row like: | just purchased | 3,286 | 21.6% | 0.09434 | 10.9 | 100.0% |
        m = re.search(rf"\|\s*{re.escape(name)}\s*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|", text)
        hits[name] = round(float(m.group(1)), 4) if m else None
    return hits


def main():
    if not OUT.parent.exists():
        OUT.parent.mkdir(parents=True)

    df = pd.read_parquet(BAND_PARQUET, engine="pyarrow")
    df["customer_id"] = df["customer_id"].astype("string")
    total = len(df)
    assert total == 15246, f"expected 15,246 evaluable customers, got {total:,}"

    counts = df["band"].value_counts().to_dict()
    assert sum(counts.values()) == 15246, "band counts must sum to 15,246"
    hit12 = _band_hit12_from_report()

    bands = []
    for name, lo, hi in config.CONTACT_BANDS:
        n = int(counts.get(name, 0))
        bands.append({"band": name, "range": _range_str(lo, hi), "customers": n,
                      "share": round(n / total, 4), "hit12": hit12[name]})
    print("Band summary (sum = {:,}):".format(sum(b["customers"] for b in bands)))
    for b in bands:
        print(f"  {b['band']:15s} range {b['range']:8s} n={b['customers']:5,} "
              f"share={b['share']:.1%} hit@12={b['hit12']}")

    # Per-demo-customer join (handles from the EXISTING customers.json — read only).
    demo_customers = json.loads((DEMO_DIR / "customers.json").read_text())
    by_cid = {r.customer_id: r for r in df.itertuples(index=False)}
    customers = {}
    cold_handles = []
    for c in demo_customers:
        h, cid = c["id"], c["cid"]
        rec = by_cid.get(cid)
        # Cold-start (no history) OR simply absent from the evaluable band file ->
        # "no history": never fabricate a timing band for a customer with no purchases.
        if c["profile"].get("cold") or rec is None:
            customers[h] = {"band": "no history", "due_ratio": None,
                            "days_since_last": None, "typical_gap": None}
            cold_handles.append(h)
        else:
            customers[h] = {"band": rec.band, "due_ratio": round(float(rec.due_ratio), 2),
                            "days_since_last": int(rec.days_since_last),
                            "typical_gap": round(float(rec.typical_gap), 2)}
    assert len(customers) == len(demo_customers), "all demo customers must appear"

    obj = {"bands": bands, "customers": customers, "finding": FINDING}
    OUT.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    json.loads(OUT.read_text())          # validate it parses
    size = OUT.stat().st_size
    print(f"\nWrote {OUT}  ({size/1024:.1f} KB)  |  {len(customers)} demo customers "
          f"({len(cold_handles)} 'no history': {cold_handles})")

    _append_report(bands, size, cold_handles, hit12)
    print("DONE.")


def _append_report(bands, size, cold_handles, hit12):
    marker = "## Phase 5c — Contact-Timing Layer"
    existing = REPORT_PATH.read_text()
    if marker in existing:
        existing = existing[:existing.index(marker)].rstrip() + "\n\n"
    band_rows = "\n".join(
        f"| {b['band']} | {b['range']} | {b['customers']:,} | {b['share']:.1%} | {b['hit12']} |"
        for b in bands)
    section = f"""{marker} (`contact_timing.json`)

A **seventh** committed demo file (added by `src/run_export_contact_timing.py`, a
standalone script that does not touch the other six exports). It surfaces the Exp 7
contact-timing bands as an action cue ("Contact now" / "Hold — recently purchased").
Read-only inputs: `contact_timing_exp7.parquet` (band per customer), the existing
`customers.json` (C01–C38 handle mapping), and `exp7_temporal.md` (per-band hit@12).
Size: **{size/1024:.1f} KB**.

### Portfolio band summary (all 15,246 evaluable customers)

| band | due_ratio range | customers | share | hit@12 |
|---|---|---|---|---|
{band_rows}

`hit@12` is parsed from `reports/exp7_temporal.md` (the Exp 7 source of truth), not
recomputed. Counts sum to exactly 15,246.

### Schema — `contact_timing.json`
```
{{
  "bands": [ {{ band, range, customers, share, hit12 }} ],          # the 5 bands above
  "customers": {{ "C01": {{ band, due_ratio, days_since_last, typical_gap }} }},
  "finding": {{ headline, detail, caveat }}
}}
```
- `due_ratio = days_since_last / typical_gap` (rounded 2dp); `typical_gap` = median
  inter-purchase gap.
- **Cold-start handling:** the {len(cold_handles)} cold-start demo customers
  ({', '.join(cold_handles)}) have no purchase history, so a real band would be
  misleading. They are emitted as `band: "no history"` with
  `due_ratio/days_since_last/typical_gap = null` — never assigned "lapsed" or "due
  now". Any demo customer absent from the evaluable band file falls back to the same
  "no history" record, so all 38 handles always appear.
- The `finding` object carries the honest Exp 7 result verbatim for the UI: recency
  beats rhythm (the "due now converts better" hypothesis failed), plus the caveat
  that the conversion metric can't see fatigue cost and the bands are
  frequency-confounded — resolvable only by a live experiment.
"""
    REPORT_PATH.write_text(existing.rstrip() + "\n\n" + section)


if __name__ == "__main__":
    main()
