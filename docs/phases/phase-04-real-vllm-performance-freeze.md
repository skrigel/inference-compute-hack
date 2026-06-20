# Phase 04 - Real vLLM Performance Freeze

**Window:** H14-H19  
**Milestone:** M4 real scorer green  
**Theme:** swap the real scorer in, measure the claims, and freeze numbers for the eval slide.

## Goals

- Run 6 data-parallel vLLM replicas with the selected Tier-1 model.
- Validate score quality before speed sweeps.
- Capture warm vs cold, full vs suffix, scoped vs full, and RAG comparison numbers.
- Regenerate performance figures with measured overlays where available.
- Record canned SSE from the real vLLM path.

## Owner Work

| Owner | Work |
|---|---|
| A | `VLLMScorer`, serve script, score-quality gate, performance sweeps, figures |
| B | backend scorer swap, health checks, queue/backpressure tuning |
| C | live adapter fallback behavior, latency readout labels, real fixture playback |
| D | final corpus and gold labels for quality gate |

## Performance Metrics To Capture

- Quality: F1, precision/recall, ROC AUC, ECE if gold labels are sufficient.
- Throughput: chunks/sec, tokens/sec, achieved MFU, batch size, p50/p95 latency.
- Cache: prefix-cache hit rate, warm-KV footprint, `gpu_cache_usage_perc`.
- Refinement: `rho`, cumulative compute, scoped asymptote, chip-removal model calls.
- RAG: index build, query cost, re-index step cost, break-even churn `D*`.
- Energy if feasible: joules/query and joules/index-build from DCGM power samples.

## Exit Gate

- `SCORER_BACKEND=vllm` runs against the same contracts as mock.
- Score-quality gate passes or the fallback model decision is recorded.
- Metrics in the eval slide are frozen with date, commit, corpus size, and model id.
- Real vLLM canned SSE fixture exists.

## Fallback

If vLLM/AWQ/prefix caching burns time, keep the live demo on mock or replay and
use the real box only for the performance slide numbers that are actually verified.
Never put unverified constants on the slide.
