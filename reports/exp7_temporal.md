# Experiment 7 — Temporal Signals (trend, seasonality, contact timing)

Tests the one signal family the project hadn't touched: **temporal**. Three signals
are added to the production **Exp 5 triple hybrid** (content 1.0 + CF 0.5 + MF 1.0;
hit@12 = 0.0628, recall@12 = 0.0229, coverage =
40.6%), and — the higher-value idea — purchase timing is also
treated as a **contact-timing decision**, not just a ranking feature.

## Why temporal should matter in fashion

Fashion is seasonal (knitwear in autumn, swimwear in summer), trend-driven (a style
is "hot" for weeks), and rhythmic (customers repurchase on a cadence). The label
window here is **2020-08-26 → 2020-09-22** — the summer→autumn transition — and the
sample spans two full years, so seasonal cycles exist to learn from.

## Pre-model diagnostics (these predict the result before the model runs)

**1. Recent-30d vs all-time top-50 overlap: 16%.**
The recent and all-time heads diverge materially, so there is genuine "what is hot now" signal for trend to exploit.

**2. Seasonality — most September-skewed vs June-skewed product types** (lift =
type's share of that month ÷ the global share of that month; >1 = skews toward it):
- **September-skewed:** shoulder bag (2.6×) · tote bag (2.5×) · bootie (2.1×) · coat (1.8×) · keychain (1.8×) · cardigan (1.8×) · cushion (1.8×) · sweater (1.7×) · dog wear (1.6×) · hoodie (1.6×)
- **June-skewed:** sarong (2.3×) · shorts (1.9×) · wedge (1.7×) · flip flop (1.7×) · sandals (1.7×) · heeled sandals (1.7×) · bikini top (1.7×) · underdress (1.7×) · swimwear bottom (1.7×) · cap (1.7×)

The split is intuitive (autumn/knitwear-leaning types skew to September; summer/beachwear to June), so the seasonal signal is real at the product-type level — the open question is whether it survives at the article level within a 28-day window.

## Tuned temporal weights (the critical diagnostic)

Triple weights are held at the Exp 5 tuned values (content 1.0, CF
0.5, MF 1.0); only the new temporal weights are tuned, on the
same feature-side validation slice (train < 2020-07-29, mini-labels in
the last 28 pre-cutoff days). **Test labels are never
touched.** Trend vs momentum and the due-ratio modifier are selected on the same slice.

| weight | signal | tuned value |
|---|---|---|
| w4 | trend (trend) | **0.5** |
| w5 | season (September propensity) | **0.0** |
| due-ratio ranking modifier | 3a | **OFF** |

Validation hit@12 moved 0.06616 → 0.08051 with the tuned temporal weights.

## Ablation — each signal attributed

| variant | hit@12 | recall@12 |
|---|---|---|
| triple hybrid (Exp5 baseline) | 0.06284 | 0.02292 |
| + trend only | 0.07563 | 0.02684 |
| + season only | 0.06284 | 0.02292 |
| + trend + season | 0.07563 | 0.02684 |
| + trend + season + due mod | 0.06717 | 0.02228 |

## Exp 5 vs Exp 7 — accuracy AND beyond-accuracy

| metric | Exp 5 triple hybrid | Exp 7 temporal | Δ |
|---|---|---|---|
| hit@6 | 0.04716 | 0.05300 | +0.00584 |
| hit@12 | 0.06284 | 0.07563 | +0.01279 |
| hit@24 | 0.07759 | 0.10357 | +0.02597 |
| recall@12 | 0.02292 | 0.02684 | +0.00392 |
| precision@12 | 0.00609 | 0.00724 | +0.00115 |
| coverage@12 | 40.6% | 29.2% | -11.4 |
| mean pop rank | 10,015 | 6,306 | -3,709 |
| top-10% head share | 69.0% | 80.2% | +11.2 |
| Gini | 0.886 | 0.948 | +0.063 |
| intra-list dissim | 0.453 | 0.491 | +0.038 |
| distinct types/list | 3.47 | 3.87 | +0.40 |
| segment fairness spread | 0.0688 | 0.0938 | +0.0250 |

## 3b — hit@12 by due-ratio quartile (analysis dimension)

Even where timing doesn't improve ranking, does it correlate with conversion?

| due-ratio quartile | n | hit@12 |
|---|---|---|
| Q1_low | 3,812 | 0.09103 |
| Q2 | 3,811 | 0.06403 |
| Q3 | 3,812 | 0.04906 |
| Q4_high | 3,811 | 0.04723 |

## 3c — Contact-timing bands (the most actionable output)

`due_ratio = days_since_last / typical_gap` (typical_gap = **median** inter-purchase
gap; single-purchase customers fall back to the population median gap and are
flagged). Each of the 15,246 evaluable customers is assigned one band. Saved to
`contact_timing_exp7.parquet` so the demo can show it as an action cue.

| band (due_ratio) | customers | share | hit@12 | mean days→first label-window buy | % who bought in window |
|---|---|---|---|---|---|
| just purchased | 3,286 | 21.6% | 0.09434 | 10.9 | 100.0% |
| approaching | 2,394 | 15.7% | 0.06307 | 11.2 | 100.0% |
| due now | 2,818 | 18.5% | 0.06317 | 10.8 | 100.0% |
| overdue | 3,002 | 19.7% | 0.04763 | 11.1 | 100.0% |
| lapsed | 3,746 | 24.6% | 0.04698 | 11.8 | 100.0% |

*(The last two columns are trivially ~constant — every evaluable customer bought in
the window by definition, so "% bought" is 100% and days-to-first-buy is ~11 for all
bands. The informative column is **hit@12**: does the recommender find that purchase?)*

**Does "due now" convert better than "just purchased"?** due-now hit@12 =
0.06317 vs just-purchased 0.09434 — a **-33%** difference.
The hypothesis is **not** supported — and the failure is informative. hit@12 falls monotonically as due-ratio rises (just-purchased **0.0943** → lapsed **0.0470**): the "due now" band (0.0632) converts *worse* than "just purchased", not better. At this 28-day horizon **recency dominates** — the customers most likely to buy again are the ones who *just* bought, not the ones whose cadence says they are "due". Due-ness does not predict higher conversion here.

**This still pairs with the Phase 3b fatigue rule — and the result sharpens the
tension.** The highest-converting band is *"just purchased"*, which is exactly the
band the fatigue rule says to **suppress**: those customers are most likely to buy
again, yet re-contacting them so soon risks annoyance and wasted contact. So the two
signals are not "fatigue = stop / due-ness = go" as originally framed; the honest
picture is that **recency predicts conversion but not contactability**, and the
bands are most useful as a *segmentation* (hold vs nurture vs win-back) feeding a
policy, not as a conversion-ranking lever.

## Honest verdict

**Trend helps accuracy but costs coverage — it is a lever, not a free upgrade.** Adding recent-window popularity (w4=0.5) lifts hit@12 0.0628→0.0756 (+20% relative), exactly as the 16%-overlap pre-model diagnostic predicted (recent and all-time heads genuinely diverge). But it pulls recommendations toward the hot recent head, so coverage falls 40.6%→29.2% (-11.4 pts) and the list leans more popular. This is the **same accuracy↔coverage trade** the Phase 3b frontier makes explicit — trend is a new knob on that frontier, not a Pareto win. **Season is a clean null (w5=0)** — and the pre-model diagnostic already told us why it *isn't* noise: the September/June product-type split is intuitive (booties, coats, boots skew autumn; sarongs, flip-flops, sandals skew summer). The signal is real seasonally but **cannot manifest inside a 28-day label window** where the season barely turns. This is a limitation of the evaluation horizon, not the signal — a full-quarter horizon would be needed to test it, and is the natural follow-up. **Purchase timing does not validate as a conversion signal at this horizon** (recency dominates the bands), but it is a genuine *decisioning* output: the contact-timing bands segment every customer into hold/nurture/win-back states and are saved for the CRM policy layer — reported honestly, including the failed 'due-now converts better' hypothesis. **Should it replace the Exp 5 triple hybrid?** **No — keep Exp 5 as the default production model.** The only accuracy gain (trend) comes at a real coverage cost, so replacement is an objective-dependent trade rather than a strict improvement. Trend belongs as a *tunable signal on the frontier* (surface it where accuracy is the goal), and the durable contribution of Exp 7 is the contact-timing layer, not a new ranking model.

## Limitations

- **28-day horizon vs seasonality.** The label window is only 28 days, so the season
  barely changes within it — a September-heavy article and an August-heavy one both
  look "in season" across the window. If season shows little effect that is a
  limitation of the **evaluation horizon**, not proof the signal is worthless; a
  longer prediction horizon (a full quarter) would be needed to see it.
- **Trend at a 30-day scale matches the horizon**, so if any temporal signal helps
  it should be trend, not season.
- The due-ratio bands are a heuristic cadence model, not a learned hazard model.
