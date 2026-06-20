# 00 — Overview: why a theoretical-compute layer

## The problem this solves
The MVP spec already has a metrics table (§9/§11) and an optimization ladder
(§11), but they are mostly **systems** metrics — latency, throughput, ratios.
They tell you *what happened*, not *what should happen*. To judges who think in
compute, "we measured 200 ms" is an anecdote; "we predicted 200 ms from first
principles and measured 210" is engineering.

This folder adds the missing layer: a roofline, FLOP accounting, MFU, and a few
closed-form models that let us **predict the curves before measuring**, then show
measured-vs-predicted. That is the move that makes the demo read as rigorous.

## The single most important reframing
**The workload is almost pure prefill with a one-token decode.** Per chunk we
process `[instruction + chunk + predicate]` and read out one logprob. That is
structurally different from chatbot/generation serving, which is *decode-bound*
and lives on the memory-bandwidth roofline (it re-streams all weights per
generated token). Single-token scoring lives on the **compute roofline** — high
arithmetic intensity, tensor-core-saturating, MFU-friendly.

This single observation reorganizes everything else in the spec:
- It is why **data-parallel replicas** (not tensor-parallel) and **large
  batches** are correct.
- It is why **FP8 compute** (not 4-bit weights) is the real throughput lever.
- It is the honest reason **recompute-over-store** (§3b) is viable: we are
  compute-bound with bandwidth headroom, so re-streaming raw data each query is
  affordable.

## The forward thesis it makes quantitative (spec §3b)
RAG/GraphRAG/vector DBs exist because inference is expensive, so engineers spend
storage to buy latency (indexes, embeddings, caches, materialized aggregates —
often 2–4× the raw data). As inference cheapens, storage/memory becomes the
binding constraint and the optimal move inverts to **recompute over store**:
persist nothing derived, regenerate on demand. Under iterative refinement (each
turn reinterprets the raw data) or changing data (each change invalidates the
index), stored state is pure liability — stale, rebuilt, or wrong.

The `compute_vs_churn` model turns this from rhetoric into a number: a break-even
data-change rate `D*` above which recompute wins by construction (and for
streaming/log workloads `D → ∞`, so we always win).

## The honest next bottleneck (don't hide it)
Recompute-over-store trades **capacity for bandwidth** — every query re-streams
raw data. The roofline shows we sit *under* the 3.35 TB/s HBM ceiling with
headroom, which is *why* recompute is viable; that headroom is finite and is the
real next constraint. Residual fixes (future, not hackathon): keep raw resident
in fast memory, push compute near the data, cheap fast-pass semantic
data-skipping to re-read less.

## What's in scope here vs. the build
| In this folder (pitch + eval) | Not changed (locked build) |
|---|---|
| roofline, FLOP/MFU accounting, closed forms | single-token logprob scoring (§6 #1) |
| measured-vs-predicted overlays | prompt order / suffix-only re-prefill (§6 #2–3) |
| compute-for-storage break-even math | candidate-set scoping (§6 #4) |
| benchmarking methodology | 4-bit weights + data-parallel replicas (§6 #5) |

The build still caches to be fast *today*; the thesis is the **direction** as
inference → free. See `01_optimization_artifacts.md` for the math behind each
artifact and `03_spec_integration.md` for exactly where each slots in.
