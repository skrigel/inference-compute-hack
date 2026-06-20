# Experiment Summary: OPT-MODAL-001-REPLICATE - modal-throughput-baseline

Generated: 2026-06-20T15:36:02Z
Agent: agent
Commit: e722980

## Executive Summary

Compared `candidate` against `baseline` across 40 matrix metric comparisons and 0 RAG dataset sizes.

**Verdict:** REJECTED
**Confidence:** LOW until repeated candidate/baseline runs and quality reruns are present.

## Dataset Configuration

| size tier | doc count | corpus description |
|---|---:|---|
| small | 7 | scaled demo corpus for RAG timing |
| small | 100 | scaled demo corpus for RAG timing |
| medium | 1000 | scaled demo corpus for RAG timing |
| large | 10000 | scaled demo corpus for RAG timing |
| xlarge | 25000 | scaled demo corpus for RAG timing |
| xxlarge | 100000 | scaled demo corpus for RAG timing |

## Aggregated Results

| workload | metric | baseline mean | candidate mean | improvement % | verdict |
|---|---|---:|---:|---:|---|
| multi_user_dynamic (1 H100) | requests_per_s | 301.459 | 292.167 | -3.082 | neutral |
| multi_user_dynamic (1 H100) | latency_ms_p50 | 86.463 | 92.100 | -6.519 | regression |
| multi_user_dynamic (1 H100) | latency_ms_p95 | 160.617 | 205.427 | -27.899 | regression |
| multi_user_dynamic (1 H100) | gpu_utilization_pct_mean | 32.000 | 24.500 | -23.438 | regression |
| multi_user_dynamic (1 H100) | derived_mfu_bf16_peak | 0.083 | 0.080 | -3.082 | neutral |
| multi_user_dynamic (6 H100) | requests_per_s | 1762.192 | 1769.026 | 0.388 | neutral |
| multi_user_dynamic (6 H100) | latency_ms_p50 | 92.042 | 89.012 | 3.291 | neutral |
| multi_user_dynamic (6 H100) | latency_ms_p95 | 212.250 | 213.867 | -0.762 | neutral |
| multi_user_dynamic (6 H100) | gpu_utilization_pct_mean | 19.417 | 28.583 | 47.210 | improved |
| multi_user_dynamic (6 H100) | derived_mfu_bf16_peak | 0.081 | 0.081 | 0.388 | neutral |
| multi_user_static (1 H100) | requests_per_s | 208.341 | 237.655 | 14.070 | improved |
| multi_user_static (1 H100) | latency_ms_p50 | 158.134 | 129.795 | 17.921 | improved |
| multi_user_static (1 H100) | latency_ms_p95 | 205.087 | 212.674 | -3.700 | neutral |
| multi_user_static (1 H100) | gpu_utilization_pct_mean | 10.000 | 28.000 | 180.000 | improved |
| multi_user_static (1 H100) | derived_mfu_bf16_peak | 0.009 | 0.011 | 14.070 | improved |
| multi_user_static (6 H100) | requests_per_s | 1351.971 | 1891.140 | 39.880 | improved |
| multi_user_static (6 H100) | latency_ms_p50 | 132.197 | 79.326 | 39.995 | improved |
| multi_user_static (6 H100) | latency_ms_p95 | 244.472 | 190.243 | 22.182 | improved |
| multi_user_static (6 H100) | gpu_utilization_pct_mean | 11.417 | 15.667 | 37.226 | improved |
| multi_user_static (6 H100) | derived_mfu_bf16_peak | 0.010 | 0.014 | 39.880 | improved |
| single_user_dynamic (1 H100) | requests_per_s | 72.296 | 77.354 | 6.996 | improved |
| single_user_dynamic (1 H100) | latency_ms_p50 | 13.542 | 12.807 | 5.432 | improved |
| single_user_dynamic (1 H100) | latency_ms_p95 | 15.570 | 14.008 | 10.037 | improved |
| single_user_dynamic (1 H100) | gpu_utilization_pct_mean | 4.500 | 16.500 | 266.667 | improved |
| single_user_dynamic (1 H100) | derived_mfu_bf16_peak | 0.020 | 0.021 | 6.996 | improved |
| single_user_dynamic (6 H100) | requests_per_s | 419.784 | 493.339 | 17.522 | improved |
| single_user_dynamic (6 H100) | latency_ms_p50 | 14.665 | 11.987 | 18.263 | improved |
| single_user_dynamic (6 H100) | latency_ms_p95 | 27.157 | 14.434 | 46.851 | improved |
| single_user_dynamic (6 H100) | gpu_utilization_pct_mean | 15.639 | 12.000 | -23.268 | regression |
| single_user_dynamic (6 H100) | derived_mfu_bf16_peak | 0.019 | 0.022 | 17.522 | improved |
| single_user_static (1 H100) | requests_per_s | 78.544 | 82.821 | 5.445 | improved |
| single_user_static (1 H100) | latency_ms_p50 | 12.237 | 11.893 | 2.815 | neutral |
| single_user_static (1 H100) | latency_ms_p95 | 14.573 | 13.404 | 8.017 | improved |
| single_user_static (1 H100) | gpu_utilization_pct_mean | 15.500 | 18.500 | 19.355 | improved |
| single_user_static (1 H100) | derived_mfu_bf16_peak | 0.004 | 0.004 | 5.445 | improved |
| single_user_static (6 H100) | requests_per_s | 441.056 | 490.557 | 11.223 | improved |
| single_user_static (6 H100) | latency_ms_p50 | 13.947 | 11.982 | 14.094 | improved |
| single_user_static (6 H100) | latency_ms_p95 | 21.098 | 14.486 | 31.339 | improved |
| single_user_static (6 H100) | gpu_utilization_pct_mean | 12.750 | 15.250 | 19.608 | improved |
| single_user_static (6 H100) | derived_mfu_bf16_peak | 0.003 | 0.004 | 11.223 | improved |

## Scaling Factors

| transition | retrieve latency factor | fresh-file latency factor |
|---|---:|---:|

## Command to Reproduce

```bash
C:\Users\Mukun\Projects\inference-compute-hack\eval\standard_benchmark.py --opt-id OPT-MODAL-001-REPLICATE --name modal-throughput-baseline --run-modal --skip-rag
```

## Artifacts

| artifact | path |
|---|---|
| config | `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/config.json` |
| aggregated results | `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/aggregated.json` |
| scaling analysis | `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/scaling_analysis.json` |
| ledger entry | `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/ledger_entry.md` |
