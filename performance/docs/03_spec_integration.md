# 03 — Spec integration map

Where each artifact slots into the FINAL merged MVP spec. Nothing here changes the
locked build (§4) or the latency spine (§6); these extend the eval + pitch.

## Artifact → spec section
| Artifact / figure | Spec hook | What it replaces / adds |
|---|---|---|
| Roofline (Fig 1) | new lead slide before §11 | the "why data-parallel + FP8 + large batch" justification, visualized |
| FLOP/MFU (Fig 2) | §11 Ladder A | theoretical floor under the ladder rungs; MFU instead of relative ratios |
| Suffix-only band | §6 #2; §9 "suffix-only vs full re-prefill" | predicted L/s band; measured plotted on it |
| Area under loop (Fig 3) | top of §9 ("measure the area under the loop") | the iteration money-shot as one picture |
| KV capacity (Fig 5) | §9 scaling-study "mark the KV crossover" row | predicted-then-confirmed crossover; the 4-bit-KV correction |
| Compute-vs-churn (Fig 4) | §3b; §14 ("you can't re-index a firehose") | the forward thesis as a number, with break-even D |
| Quantization split | §5/§6 quantization line; §11 ladder | corrects "4-bit→4× throughput"; splits capacity vs throughput levers |
| Calibration (ROC/ECE) | §9 quality row; §6 cascade; §1 threshold handle | "is the filter good?" → a curve; motivates the cascade |

## Metric-table additions (extends §9 / §11)
New rows worth adding, each derived not assumed:
- **MFU (cold scan)** — achieved ÷ theoretical peak; target 40–55% prefill on H100.
- **Arithmetic intensity (scan vs decode)** — confirms compute-bound regime.
- **Suffix-only speedup vs predicted L/s band** — report measured-on-band.
- **Cumulative compute @ turn k (scoped vs full vs RAG)** — the area under the loop.
- **Break-even data-churn D\*** — where recompute-over-store overtakes RAG.
- **Energy per query / per re-index (joules)** — the honest cost unit.
- **Score AUC + ECE vs gold** — quality + calibration before optimizing speed.

## Demo-script touch points (§8)
- Beat "drag in fresh file → query instantly": back the Footprint flip with the
  **compute-vs-churn** number ("RAG needs a re-index here; at this churn rate
  that's the crossover").
- Beat "drag the threshold": narrate it as **sliding along the ROC curve**.
- Eval slide (§9 close): lead with the **area-under-the-loop** graph — our cost
  flattens, RAG climbs and step-jumps.

## Risk-register additions (extends §12)
- **MFU under-reports** if measured on padded tokens — log real vs padded.
- **Cold floor contaminated** by warm-cache leakage — capture it first, reset
  explicitly.
- **KV formula over-/under-estimates** vs PagedAttention + prefix dedup — measure
  real allocation, treat the gap as a finding.
- **Quantization overclaim** — use the corrected capacity-vs-throughput split
  (artifact 6) so a judge can't catch "4-bit→4× speed" in the compute-bound regime.

## Build-plan touch points (§10)
- **H0–3 "stand up eval harness":** implement the cache counter + counterfactual
  replay + measured-vs-predicted overlays against `theory.py`. Capture the
  perishable naive cold floor *now*.
- **H14–19 "scale 10× / warm→scoped crossover":** the KV-capacity sweep confirms
  the predicted ~14k crossover; the area-under-loop curves get their distribution.
