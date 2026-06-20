# performance/ - theoretical compute layer for *Grep for Meaning*

This folder is the **performance-rigor layer** that sits underneath the measured
numbers in the MVP spec. It does not change the locked build or the latency spine
(§4/§6). It adds the "theoretical compute / graphs / math" that makes the demo
read as *predicted-then-measured* rather than merely instrumented.

The core move: for most of these graphs the x-axis is **compute, not seconds**.
We derive closed-form predictions from hardware constants + model shape
(`theory.py`), then `eval/bench.py` overlays measured points on the predicted
curves. Showing measured tracking a derivation is far more convincing than a
bare ratio.

## Contents
```
performance/
  README.md                       <- you are here
  theory.py                       <- closed-form models + figure generators
  figures/                        <- 5 figures, each .png (slides) + .svg (scalable)
  docs/
    00_overview.md                <- the thesis, and why a performance framing
    01_optimization_artifacts.md  <- the 6 theoretical artifacts, with the math
    02_benchmarking_methodology.md<- how to measure these without lying to yourself
    03_spec_integration.md        <- where each piece slots into the MVP spec
    04_constants_to_verify.md     <- hardware/model assumptions to confirm on the box
```

## Quick start
```bash
python -m pip install -r performance/requirements.txt
python performance/theory.py     # prints worked examples + regenerates performance/figures/
```

## The five figures (and the one-line claim each makes)
1. **roofline** — semantic scoring lives on the *compute* ceiling, not the bandwidth wall.
2. **mfu_waterfall** — theoretical floor vs achieved; the gap is MFU (Ladder A).
3. **area_under_loop** — cumulative compute over a refine session; our cost *flattens* while RAG climbs. **(the money shot)**
4. **compute_vs_churn** — recompute-over-store beats RAG above a break-even data-churn rate.
5. **kv_capacity** — warm-KV footprint vs corpus size; marks the KV crossover (#3 -> #4 handoff).

## Status / provenance
Derived from the FINAL merged MVP spec. The closed forms reproduce the spec's
own worked targets (e.g. ~36s cold scan @ 40% MFU lands in the §11 30–60s window).
One correction surfaced: at the top of the 10–20k corpus range, **4-bit KV is
required** for the warm path to fit 640 GB, not optional — see
`docs/01_optimization_artifacts.md` §4 and `docs/04_constants_to_verify.md`.

The hardware/model constants in `theory.py` are best-estimates. Verify them on
the real 8×H100 node before putting numbers on a slide — see `docs/04`.
