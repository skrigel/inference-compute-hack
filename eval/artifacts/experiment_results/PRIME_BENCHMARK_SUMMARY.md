# GPU Memory Optimization Benchmark Results

**Run Date:** 2026-06-20
**Infrastructure:** Prime Intellect (lambdalabs provider)
**GPU:** NVIDIA H100 80GB
**Model:** Qwen/Qwen2.5-3B-Instruct-AWQ
**vLLM Version:** 0.6.6.post1

## Quick Reference

Results are stored in individual experiment folders:
```
eval/artifacts/experiment_results/
├── EXP-MBT-001/run_001.json    # Baseline (max_batched_tokens=12288)
├── EXP-MBT-002/run_001.json    # max_batched_tokens=16384
├── EXP-FP8-001/run_001.json    # FP8 KV cache
├── EXP-SCHED-001/run_001.json  # Time-window scheduling (15ms)
├── EXP-LENBIN-001/run_001.json # Input-length binning
└── PRIME_BENCHMARK_SUMMARY.md  # This file
```

## Results Summary

| Experiment | Config Change | Throughput | P50 | P95 | P99 | Verdict |
|------------|--------------|-----------|-----|-----|-----|---------|
| **MBT-001** | baseline (12288 tokens) | 132.74 req/s | 182.9ms | 503.9ms | 649.0ms | Baseline |
| **MBT-002** | 16384 tokens | 133.37 req/s | 193.1ms | 475.7ms | 683.2ms | ✅ Slight win |
| **FP8-001** | fp8 KV cache | 122.98 req/s | 219.6ms | 501.4ms | 557.5ms | ❌ Slower |
| **SCHED-001** | 15ms batch accumulation | 124.07 req/s | 164.6ms | 231.2ms | 240.2ms | ✅ Best latency consistency |
| **LENBIN-001** | length-binned requests | 128.47 req/s | 198.8ms | 384.1ms | 583.6ms | ✅ Better tail latencies |

## Key Findings

### 1. Max Batched Tokens (MBT-001 vs MBT-002)
- Increasing `max_num_batched_tokens` from 12288 → 16384 gives marginal throughput improvement (+0.5%)
- **Recommendation:** Use 16384 if memory allows

### 2. FP8 KV Cache (FP8-001)
- FP8 KV cache is **7% slower** with AWQ-quantized models
- Likely due to dtype conversion overhead between AWQ (int4) and FP8
- **Recommendation:** Skip FP8 KV cache when using AWQ quantization

### 3. Time-Window Scheduling (SCHED-001)
- 15ms batch accumulation reduces throughput by 6.5%
- But dramatically improves latency consistency (P95: 231ms vs 504ms!)
- **Recommendation:** Use for latency-sensitive workloads; skip for throughput-critical

### 4. Input-Length Binning (LENBIN-001)
- Grouping similar-length prompts reduces throughput by 3%
- Improves P95 latency by 24% (384ms vs 504ms)
- **Recommendation:** Consider for production workloads with mixed prompt lengths

## Not Tested (Application-Level)

These experiments require application-level changes, not vLLM config:
- **EXP-BATCH-001:** Application batch size (QUERY_BATCH_SIZE, REFINE_BATCH_SIZE)
- **EXP-OVERLAP-001/002:** Document chunk overlap ratios (CHUNK_OVERLAP_RATIO)

## Reproducing Results

1. Spin up Prime pod with H100:
   ```bash
   prime pods create --id <resource_id> --disk-size 100
   ```

2. Install vLLM 0.6.6:
   ```bash
   pip install vllm==0.6.6.post1 httpx
   ```

3. Run benchmark scripts from `/tmp/benchmark_*.py`

## Next Steps

1. Apply MBT-002 config (16384 max_batched_tokens) as new baseline
2. Consider SCHED-001 pattern for latency-sensitive endpoints
3. Test EXP-BATCH-001 and EXP-OVERLAP experiments at application level
4. Run quality gates to verify no accuracy regression
