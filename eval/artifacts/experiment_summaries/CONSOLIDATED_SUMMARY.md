# GPU Memory Optimization Experiments - Consolidated Summary

Generated: 2026-06-20T17:35:48Z

## Overview

| Experiment | Name | Status | Hypothesis |
|---|---|---|---|
| EXP-FP8-001 | fp8 KV cache | config only | fp8 KV cache halves memory, enables larger batches without O... |
| EXP-BATCH-001 | increased batch sizes | config only | Larger batches (128 vs 64) improve GPU utilization... |
| EXP-MBT-001 | max batched tokens 12288 | config only | Larger max_num_batched_tokens improves prefill throughput... |
| EXP-MBT-002 | max batched tokens 16384 | config only | Larger max_num_batched_tokens improves prefill throughput... |
| EXP-SCHED-001 | time-window scheduling 15ms | config only | Accumulating requests for 15ms improves batch efficiency... |
| EXP-LENBIN-001 | input-length binning | config only | Routing similar-length inputs together reduces padding waste... |
| EXP-OVERLAP-001 | chunk overlap 10% | config only | 10% overlap improves recall at chunk boundaries... |
| EXP-OVERLAP-002 | chunk overlap 20% | config only | 20% overlap improves recall at chunk boundaries... |

## Experiment Details

### EXP-FP8-001: fp8 KV cache

**Hypothesis:** fp8 KV cache halves memory, enables larger batches without OOM

**Success Criteria:** F1 >= 0.7, throughput neutral or improved

**Configuration:**
- Env vars: `{'KV_CACHE_DTYPE': 'fp8'}`
- Modal args: `none`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-FP8-001/config.json`
- Results: `eval/artifacts/experiment_results/EXP-FP8-001/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-FP8-001_summary.md`

---

### EXP-BATCH-001: increased batch sizes

**Hypothesis:** Larger batches (128 vs 64) improve GPU utilization

**Success Criteria:** >=5% throughput improvement, p95 regression <10%

**Configuration:**
- Env vars: `{'QUERY_BATCH_SIZE': '128', 'REFINE_BATCH_SIZE': '128'}`
- Modal args: `none`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-BATCH-001/config.json`
- Results: `eval/artifacts/experiment_results/EXP-BATCH-001/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-BATCH-001_summary.md`

---

### EXP-MBT-001: max batched tokens 12288

**Hypothesis:** Larger max_num_batched_tokens improves prefill throughput

**Success Criteria:** >=5% throughput improvement without OOM

**Configuration:**
- Env vars: `none`
- Modal args: `['--max-num-batched-tokens', '12288']`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-MBT-001/config.json`
- Results: `eval/artifacts/experiment_results/EXP-MBT-001/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-MBT-001_summary.md`

---

### EXP-MBT-002: max batched tokens 16384

**Hypothesis:** Larger max_num_batched_tokens improves prefill throughput

**Success Criteria:** >=5% throughput improvement without OOM

**Configuration:**
- Env vars: `none`
- Modal args: `['--max-num-batched-tokens', '16384']`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-MBT-002/config.json`
- Results: `eval/artifacts/experiment_results/EXP-MBT-002/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-MBT-002_summary.md`

---

### EXP-SCHED-001: time-window scheduling 15ms

**Hypothesis:** Accumulating requests for 15ms improves batch efficiency

**Success Criteria:** >=10% throughput improvement to justify latency

**Configuration:**
- Env vars: `{'BATCH_ACCUMULATE_MS': '15'}`
- Modal args: `none`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-SCHED-001/config.json`
- Results: `eval/artifacts/experiment_results/EXP-SCHED-001/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-SCHED-001_summary.md`

---

### EXP-LENBIN-001: input-length binning

**Hypothesis:** Routing similar-length inputs together reduces padding waste

**Success Criteria:** >=5% throughput improvement, p95 latency improved

**Configuration:**
- Env vars: `{'VLLM_ROUTING_MODE': 'length_bin'}`
- Modal args: `none`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-LENBIN-001/config.json`
- Results: `eval/artifacts/experiment_results/EXP-LENBIN-001/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-LENBIN-001_summary.md`

---

### EXP-OVERLAP-001: chunk overlap 10%

**Hypothesis:** 10% overlap improves recall at chunk boundaries

**Success Criteria:** Recall improvement with acceptable throughput cost

**Configuration:**
- Env vars: `{'CHUNK_OVERLAP_RATIO': '0.1'}`
- Modal args: `none`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-OVERLAP-001/config.json`
- Results: `eval/artifacts/experiment_results/EXP-OVERLAP-001/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-OVERLAP-001_summary.md`

---

### EXP-OVERLAP-002: chunk overlap 20%

**Hypothesis:** 20% overlap improves recall at chunk boundaries

**Success Criteria:** Recall improvement with acceptable throughput cost

**Configuration:**
- Env vars: `{'CHUNK_OVERLAP_RATIO': '0.2'}`
- Modal args: `none`

**Results:** Not yet available

**Artifacts:**
- Config: `eval/artifacts/experiment_results/EXP-OVERLAP-002/config.json`
- Results: `eval/artifacts/experiment_results/EXP-OVERLAP-002/aggregated.json`
- Summary: `eval/artifacts/experiment_summaries/EXP-OVERLAP-002_summary.md`

---

## Recommendations

Based on experiment results:

**Pending execution:** EXP-FP8-001, EXP-BATCH-001, EXP-MBT-001, EXP-MBT-002, EXP-SCHED-001, EXP-LENBIN-001, EXP-OVERLAP-001, EXP-OVERLAP-002

## Next Steps

1. Review individual experiment summaries in `eval/artifacts/experiment_summaries/`
2. Apply successful optimizations incrementally
3. Re-run quality gates after applying changes
4. Update `docs/optimization-results-ledger.md` with final decisions
