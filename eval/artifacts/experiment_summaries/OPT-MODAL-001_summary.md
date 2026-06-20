# Experiment Summary: OPT-MODAL-001 - modal-throughput-baseline

Generated: 2026-06-20T12:19:38Z
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
| multi_user_dynamic (1 H100) | requests_per_s | 301.459 | 295.538 | -1.964 | neutral |
| multi_user_dynamic (1 H100) | latency_ms_p50 | 86.463 | 86.767 | -0.351 | neutral |
| multi_user_dynamic (1 H100) | latency_ms_p95 | 160.617 | 177.897 | -10.758 | regression |
| multi_user_dynamic (1 H100) | gpu_utilization_pct_mean | 32.000 | 31.000 | -3.125 | neutral |
| multi_user_dynamic (1 H100) | derived_mfu_bf16_peak | 0.083 | 0.081 | -1.964 | neutral |
| multi_user_dynamic (6 H100) | requests_per_s | 1762.192 | 1786.504 | 1.380 | neutral |
| multi_user_dynamic (6 H100) | latency_ms_p50 | 92.042 | 88.079 | 4.305 | neutral |
| multi_user_dynamic (6 H100) | latency_ms_p95 | 212.250 | 223.916 | -5.496 | regression |
| multi_user_dynamic (6 H100) | gpu_utilization_pct_mean | 19.417 | 15.000 | -22.747 | regression |
| multi_user_dynamic (6 H100) | derived_mfu_bf16_peak | 0.081 | 0.082 | 1.380 | neutral |
| multi_user_static (1 H100) | requests_per_s | 208.341 | 217.375 | 4.336 | neutral |
| multi_user_static (1 H100) | latency_ms_p50 | 158.134 | 139.621 | 11.707 | improved |
| multi_user_static (1 H100) | latency_ms_p95 | 205.087 | 214.643 | -4.660 | neutral |
| multi_user_static (1 H100) | gpu_utilization_pct_mean | 10.000 | 4.000 | -60.000 | regression |
| multi_user_static (1 H100) | derived_mfu_bf16_peak | 0.009 | 0.010 | 4.336 | neutral |
| multi_user_static (6 H100) | requests_per_s | 1351.971 | 1812.446 | 34.060 | improved |
| multi_user_static (6 H100) | latency_ms_p50 | 132.197 | 81.151 | 38.614 | improved |
| multi_user_static (6 H100) | latency_ms_p95 | 244.472 | 224.609 | 8.125 | improved |
| multi_user_static (6 H100) | gpu_utilization_pct_mean | 11.417 | 13.250 | 16.058 | improved |
| multi_user_static (6 H100) | derived_mfu_bf16_peak | 0.010 | 0.014 | 34.060 | improved |
| single_user_dynamic (1 H100) | requests_per_s | 72.296 | 70.400 | -2.624 | neutral |
| single_user_dynamic (1 H100) | latency_ms_p50 | 13.542 | 14.161 | -4.568 | neutral |
| single_user_dynamic (1 H100) | latency_ms_p95 | 15.570 | 15.510 | 0.385 | neutral |
| single_user_dynamic (1 H100) | gpu_utilization_pct_mean | 4.500 | 4.000 | -11.111 | regression |
| single_user_dynamic (1 H100) | derived_mfu_bf16_peak | 0.020 | 0.019 | -2.624 | neutral |
| single_user_dynamic (6 H100) | requests_per_s | 419.784 | 459.391 | 9.435 | improved |
| single_user_dynamic (6 H100) | latency_ms_p50 | 14.665 | 12.750 | 13.064 | improved |
| single_user_dynamic (6 H100) | latency_ms_p95 | 27.157 | 15.670 | 42.298 | improved |
| single_user_dynamic (6 H100) | gpu_utilization_pct_mean | 15.639 | 16.833 | 7.638 | improved |
| single_user_dynamic (6 H100) | derived_mfu_bf16_peak | 0.019 | 0.021 | 9.435 | improved |
| single_user_static (1 H100) | requests_per_s | 78.544 | 82.184 | 4.635 | neutral |
| single_user_static (1 H100) | latency_ms_p50 | 12.237 | 11.808 | 3.510 | neutral |
| single_user_static (1 H100) | latency_ms_p95 | 14.573 | 13.271 | 8.933 | improved |
| single_user_static (1 H100) | gpu_utilization_pct_mean | 15.500 | 15.500 | 0.000 | neutral |
| single_user_static (1 H100) | derived_mfu_bf16_peak | 0.004 | 0.004 | 4.635 | neutral |
| single_user_static (6 H100) | requests_per_s | 441.056 | 479.991 | 8.828 | improved |
| single_user_static (6 H100) | latency_ms_p50 | 13.947 | 12.221 | 12.378 | improved |
| single_user_static (6 H100) | latency_ms_p95 | 21.098 | 14.827 | 29.722 | improved |
| single_user_static (6 H100) | gpu_utilization_pct_mean | 12.750 | 14.333 | 12.418 | improved |
| single_user_static (6 H100) | derived_mfu_bf16_peak | 0.003 | 0.004 | 8.828 | improved |

## Scaling Factors

| transition | retrieve latency factor | fresh-file latency factor |
|---|---:|---:|

## Command to Reproduce

```bash
C:\Users\Mukun\Projects\inference-compute-hack\eval\standard_benchmark.py --opt-id OPT-MODAL-001 --name modal-throughput-baseline --run-modal --skip-rag
```

## Artifacts

| artifact | path |
|---|---|
| config | `eval/artifacts/experiment_results/OPT-MODAL-001/config.json` |
| aggregated results | `eval/artifacts/experiment_results/OPT-MODAL-001/aggregated.json` |
| scaling analysis | `eval/artifacts/experiment_results/OPT-MODAL-001/scaling_analysis.json` |
| ledger entry | `eval/artifacts/experiment_results/OPT-MODAL-001/ledger_entry.md` |
