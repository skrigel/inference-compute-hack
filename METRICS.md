# Performance Metrics Plan

The metric story is performance-first: predict the curves from compute models,
measure the system, then show measured points against the prediction. Timing
alone is not enough because wall-clock latency moves with batching, queueing, and
warm state. The primary unit is inference work.

## Metric Hierarchy

| Layer | Primary metric | Why it matters |
|---|---|---|
| Work | `chunks_scored`, `chunks_served_from_cache` | Deterministic compute count for query/refine turns |
| Compute | FLOPs, achieved TFLOP/s, MFU | Shows how close the scan is to H100 peak |
| Latency | p50/p95 query and refine time | The felt interactive experience |
| Cache | prefix hit rate, warm-KV footprint, cache-missing set | Explains cold/warm/scoped behavior |
| Quality | F1, precision/recall, AUC, ECE | Prevents optimizing speed while wrong |
| RAG exchange | index build, query cost, break-even churn `D*` | Quantifies recompute-over-store |
| Energy | joules/query, joules/index build | Optional but strongest cost unit |

## Required Counters

Every query/refine trace should include:

- `run_id`, `commit`, `corpus_id`, `model_id`, `scorer_backend`
- `turn`, `operation`, `threshold`
- `n_chunks_total`, `candidate_count`, `chunks_scored`, `chunks_served_from_cache`
- `survivor_count`, `rho`
- `elapsed_ms`, `model_ms`, `queue_ms` where available
- `cache_hit_rate`, `warm_state`, `latency_kind`
- `quality_slice` when the turn maps to gold labels

## Required Figures

The imported performance package provides the theoretical basis:

- [`performance/docs/00_overview.md`](performance/docs/00_overview.md) - performance thesis.
- [`performance/docs/01_optimization_artifacts.md`](performance/docs/01_optimization_artifacts.md) - math and figures.
- [`performance/docs/02_benchmarking_methodology.md`](performance/docs/02_benchmarking_methodology.md) - measurement hygiene.
- [`performance/docs/03_spec_integration.md`](performance/docs/03_spec_integration.md) - where the artifacts slot into the MVP.
- [`performance/docs/04_constants_to_verify.md`](performance/docs/04_constants_to_verify.md) - constants to verify on the H100 node.

Minimum slide set:

1. Roofline: our scan is compute-bound at large batch.
2. MFU waterfall: theoretical floor vs achieved time per optimization rung.
3. Area under loop: scoped refinement saturates while RAG climbs and re-indexes.
4. Compute-vs-churn: break-even data-change rate where recompute wins.
5. KV capacity: warm-cache crossover and why candidate scoping carries the long tail.

## Measurement Rules

- Count inference work before timing it.
- Capture the naive cold floor before enabling warm-state optimizations.
- Report real-token and padded-token MFU separately if both are available.
- Label every number as `predicted`, `measured`, or `projected`.
- Validate quality before speed sweeps.
- Treat 4-bit weights as a capacity lever and FP8 compute as the throughput lever.

## Phase Ownership

- Phase 00 captures constants, cold floor, and counters.
- Phase 01 proves query/dashboard metrics and zero-inference threshold drag.
- Phase 02 records refine traces and the area-under-loop inputs.
- Phase 04 freezes real vLLM numbers and regenerates figures.
- Phase 06 preserves artifacts with commit, model, corpus, and caveats.
