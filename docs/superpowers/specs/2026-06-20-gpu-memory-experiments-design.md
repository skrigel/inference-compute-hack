# GPU Memory Optimization Experiments Design

**Date:** 2026-06-20
**Status:** Approved
**Author:** Agent + User collaboration

## Overview

This document specifies 6 independent experiments to test GPU memory optimizations for the inference pipeline. Each experiment isolates a single independent variable against the current baseline to enable clean attribution of performance changes.

## Approach

**Strategy:** Sequential Independent Experiments (Approach A)

Each experiment changes exactly one variable from the current baseline. This provides:
- Clean causality - wins/losses are attributable to a single change
- Independent validation - experiments can run in any order
- Simple rollback - each optimization can be accepted or rejected independently

If multiple optimizations pass, a follow-up combined experiment can test interactions.

## Experiment Summary

| ID | Experiment | Independent Variable | Values Tested |
|---|---|---|---|
| EXP-FP8-001 | fp8 KV cache | `KV_CACHE_DTYPE` | `auto` → `fp8` |
| EXP-BATCH-001 | Query/refine batch sizes | `QUERY_BATCH_SIZE` | 64 → 128 |
| EXP-MBT-001 | Max batched tokens | `max_num_batched_tokens` | 8192 → 12288, 16384 |
| EXP-SCHED-001 | Time-window scheduling | `BATCH_ACCUMULATE_MS` | 0 → 15ms |
| EXP-LENBIN-001 | Input-length binning | `ROUTING_MODE` | `round_robin` → `length_bin` |
| EXP-OVERLAP-001 | Chunk overlap | `CHUNK_OVERLAP_RATIO` | 0.0 → 0.1, 0.2 |

## Common Baseline

All experiments compare against the current Phase 04 baseline:

| Parameter | Baseline Value | Source |
|---|---|---|
| `KV_CACHE_DTYPE` | `auto` | `inference/modal_app.py:71` |
| `QUERY_BATCH_SIZE` | 64 | `backend/streaming.py:17` |
| `max_num_batched_tokens` | 8192 | `inference/modal_app.py:79` |
| Batch accumulation | disabled (immediate dispatch) | N/A |
| Routing mode | `round_robin` | `inference/vllm_scorer.py` |
| Chunk overlap | 0.0 | N/A |

Baseline artifact: `eval/artifacts/phase04_h100_rag_matrix.json`

## Test Matrix

Each experiment runs the same matrix for comparability.

### GPU Configurations
- 1 H100 (single-GPU efficiency)
- 6 H100s (scaled production behavior)

### Scenarios

| Scenario | Dataset Mode | Concurrency | Requests |
|---|---|---|---|
| single_user_static | static | 1 | 32 |
| multi_user_static | static | 32 | 96 |
| single_user_dynamic | dynamic | 1 | 32 |
| multi_user_dynamic | dynamic | 32 | 96 |

### Dataset Sizes (RAG Ladder)
- 7, 100, 1,000, 10,000, 25,000, 100,000 docs

### Repetitions
- 5 runs per configuration
- Warmup run excluded from statistics
- Report: mean, std, min, max, 95% CI, p-value vs baseline

### Total Runs Per Experiment
- 4 scenarios × 2 GPU configs × 6 dataset sizes × 5 reps = **240 runs**
- Plus RAG baseline ladder: 6 sizes × 5 reps = 30 runs

## Per-Experiment Details

### EXP-FP8-001: fp8 KV Cache

| Field | Value |
|---|---|
| **Hypothesis** | fp8 KV cache halves KV memory → enables either longer contexts or larger batches without OOM |
| **Change** | `KV_CACHE_DTYPE=fp8` in `modal_app.py` |
| **Expected mechanism** | H100 native fp8 support reduces memory per KV entry from 16 bits to 8 bits |
| **Risk** | Potential quality degradation from reduced precision |
| **Success criteria** | No quality regression (F1 ≥ 0.7), throughput neutral or improved |

### EXP-BATCH-001: Increased Batch Sizes

| Field | Value |
|---|---|
| **Hypothesis** | Larger batches (128 vs 64) improve GPU utilization by amortizing kernel launch overhead |
| **Change** | `QUERY_BATCH_SIZE=128`, `REFINE_BATCH_SIZE=128` in `streaming.py` |
| **Expected mechanism** | More chunks per vLLM batch → better tensor-core saturation |
| **Risk** | Higher memory pressure, potential latency increase for interactive workloads |
| **Success criteria** | ≥5% throughput improvement, p95 latency regression <10% |

### EXP-MBT-001: Max Batched Tokens

| Field | Value |
|---|---|
| **Hypothesis** | Larger `max_num_batched_tokens` (12288/16384 vs 8192) improves prefill throughput |
| **Change** | `--max-num-batched-tokens 12288` and `16384` in `modal_app.py` |
| **Expected mechanism** | Prefill-bound workload benefits from processing more tokens per batch |
| **Risk** | OOM at higher values if KV cache memory insufficient |
| **Success criteria** | ≥5% throughput improvement without OOM |

### EXP-SCHED-001: Time-Window Scheduling

| Field | Value |
|---|---|
| **Hypothesis** | Accumulating requests for 15ms before dispatch improves batch efficiency |
| **Change** | New `BatchAccumulator` class with `BATCH_ACCUMULATE_MS=15` |
| **Expected mechanism** | Instead of immediate dispatch, wait up to 15ms to fill batch → better GPU utilization |
| **Risk** | Adds 0-15ms latency to every request |
| **Success criteria** | ≥10% throughput improvement to justify latency cost |

### EXP-LENBIN-001: Input-Length Binning

| Field | Value |
|---|---|
| **Hypothesis** | Routing similar-length inputs together reduces padding waste and straggler latency |
| **Change** | New `length_bin` routing mode: short (<512 tokens), medium (512-2048), long (>2048) |
| **Expected mechanism** | Homogeneous batches avoid padding shorter sequences to match longest |
| **Risk** | Load imbalance if length distribution is skewed |
| **Success criteria** | ≥5% throughput improvement, p95 latency improvement |

### EXP-OVERLAP-001: Chunk Overlap

| Field | Value |
|---|---|
| **Hypothesis** | 10-20% overlap between chunks improves recall at boundaries |
| **Change** | New `CHUNK_OVERLAP_RATIO` parameter in chunking pipeline (0.1, 0.2) |
| **Expected mechanism** | Semantic content spanning chunk boundaries is captured in both chunks |
| **Risk** | Increases corpus size proportionally (10% overlap → ~10% more chunks) |
| **Success criteria** | Recall improvement with acceptable throughput cost |

## Implementation Changes Required

### EXP-FP8-001 - Minimal changes
- `inference/modal_app.py`: Already supports `KV_CACHE_DTYPE` env var
- Run with `KV_CACHE_DTYPE=fp8`

### EXP-BATCH-001 - Minimal changes
- `backend/streaming.py`: Already supports `QUERY_BATCH_SIZE` env var
- Add `REFINE_BATCH_SIZE` env var support (currently hardcoded)
- Run with `QUERY_BATCH_SIZE=128 REFINE_BATCH_SIZE=128`

### EXP-MBT-001 - Minimal changes
- `inference/modal_app.py`: Already parameterized via CLI
- Run with `--max-num-batched-tokens 12288` and `16384`

### EXP-SCHED-001 - New code required
```
backend/batch_accumulator.py (new file)
├── BatchAccumulator class
│   ├── __init__(max_wait_ms, max_batch_size)
│   ├── add(request) → triggers dispatch if full
│   └── _dispatch() → sends accumulated batch to scorer
```
- Integrate into `backend/streaming.py` query path
- Env var: `BATCH_ACCUMULATE_MS` (default 0 = disabled)

### EXP-LENBIN-001 - New code required
```
inference/vllm_scorer.py
├── _estimate_tokens(text) → rough token count
├── _length_bin(token_count) → "short" | "medium" | "long"
└── _route_replica() updated to use length bins
```
- Env var: `VLLM_ROUTING_MODE=length_bin`
- Bin thresholds: `LENBIN_SHORT_MAX=512`, `LENBIN_MEDIUM_MAX=2048`

### EXP-OVERLAP-001 - New code required
```
data/chunker.py (new or extend existing)
├── chunk_with_overlap(text, chunk_size, overlap_ratio)
└── Returns overlapping chunks
```
- Env var: `CHUNK_OVERLAP_RATIO` (default 0.0)
- Affects corpus generation, not inference path

## Artifacts

### Per-Experiment Artifacts
Generated by `eval/standard_benchmark.py`:

```
eval/artifacts/experiment_results/EXP-XXX-001/
├── config.json              # Exact configuration and command
├── runs/*.json              # Individual run results
├── aggregated.json          # Statistics and comparisons
├── scaling_analysis.json    # Dataset size scaling behavior
└── ledger_entry.md          # Paste-ready for optimization-results-ledger.md
```

### Summaries
```
eval/artifacts/experiment_summaries/
├── EXP-FP8-001_summary.md
├── EXP-BATCH-001_summary.md
├── EXP-MBT-001_summary.md
├── EXP-SCHED-001_summary.md
├── EXP-LENBIN-001_summary.md
└── EXP-OVERLAP-001_summary.md
```

### Ledger Updates
- Each experiment gets an entry in `docs/optimization-results-ledger.md`
- Status progression: `proposed` → `running` → `applied` | `rejected`

## Success Criteria Summary

| Experiment | Primary Metric | Threshold | Quality Gate |
|---|---|---|---|
| EXP-FP8-001 | throughput | neutral or improved | F1 ≥ 0.7 |
| EXP-BATCH-001 | throughput | ≥5% improvement | p95 regression <10% |
| EXP-MBT-001 | throughput | ≥5% improvement | no OOM |
| EXP-SCHED-001 | throughput | ≥10% improvement | latency tradeoff justified |
| EXP-LENBIN-001 | throughput | ≥5% improvement | p95 latency improved |
| EXP-OVERLAP-001 | recall | improved | throughput cost acceptable |

## Next Steps

1. Write implementation plan via `superpowers:writing-plans` skill
2. Implement code changes for each experiment
3. Run experiments sequentially
4. Document results in ledger
5. Accept/reject each optimization independently
