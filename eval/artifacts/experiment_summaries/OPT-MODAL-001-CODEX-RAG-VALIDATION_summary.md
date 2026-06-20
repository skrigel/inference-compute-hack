# Experiment Summary: OPT-MODAL-001-CODEX-RAG-VALIDATION - post-merge Modal optimization validation with RAG ladder

Generated: 2026-06-20T16:19:09Z
Agent: codex
Commit: 7a090a9

## Executive Summary

Compared `candidate` against `baseline` across 40 matrix metric comparisons and 6 RAG dataset sizes.

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
| multi_user_dynamic (1 H100) | requests_per_s | 301.459 | 260.994 | -13.423 | regression |
| multi_user_dynamic (1 H100) | latency_ms_p50 | 86.463 | 91.680 | -6.033 | regression |
| multi_user_dynamic (1 H100) | latency_ms_p95 | 160.617 | 188.563 | -17.399 | regression |
| multi_user_dynamic (1 H100) | gpu_utilization_pct_mean | 32.000 | 21.000 | -34.375 | regression |
| multi_user_dynamic (1 H100) | derived_mfu_bf16_peak | 0.083 | 0.072 | -13.423 | regression |
| multi_user_dynamic (6 H100) | requests_per_s | 1762.192 | 1520.485 | -13.716 | regression |
| multi_user_dynamic (6 H100) | latency_ms_p50 | 92.042 | 111.883 | -21.557 | regression |
| multi_user_dynamic (6 H100) | latency_ms_p95 | 212.250 | 253.881 | -19.614 | regression |
| multi_user_dynamic (6 H100) | gpu_utilization_pct_mean | 19.417 | 22.167 | 14.163 | improved |
| multi_user_dynamic (6 H100) | derived_mfu_bf16_peak | 0.081 | 0.070 | -13.716 | regression |
| multi_user_static (1 H100) | requests_per_s | 208.341 | 226.626 | 8.776 | improved |
| multi_user_static (1 H100) | latency_ms_p50 | 158.134 | 143.460 | 9.280 | improved |
| multi_user_static (1 H100) | latency_ms_p95 | 205.087 | 202.462 | 1.280 | neutral |
| multi_user_static (1 H100) | gpu_utilization_pct_mean | 10.000 | 5.000 | -50.000 | regression |
| multi_user_static (1 H100) | derived_mfu_bf16_peak | 0.009 | 0.010 | 8.776 | improved |
| multi_user_static (6 H100) | requests_per_s | 1351.971 | 1247.423 | -7.733 | regression |
| multi_user_static (6 H100) | latency_ms_p50 | 132.197 | 153.669 | -16.242 | regression |
| multi_user_static (6 H100) | latency_ms_p95 | 244.472 | 287.155 | -17.459 | regression |
| multi_user_static (6 H100) | gpu_utilization_pct_mean | 11.417 | 8.500 | -25.547 | regression |
| multi_user_static (6 H100) | derived_mfu_bf16_peak | 0.010 | 0.009 | -7.733 | regression |
| single_user_dynamic (1 H100) | requests_per_s | 72.296 | 66.572 | -7.919 | regression |
| single_user_dynamic (1 H100) | latency_ms_p50 | 13.542 | 14.993 | -10.711 | regression |
| single_user_dynamic (1 H100) | latency_ms_p95 | 15.570 | 16.994 | -9.141 | regression |
| single_user_dynamic (1 H100) | gpu_utilization_pct_mean | 4.500 | 15.000 | 233.333 | improved |
| single_user_dynamic (1 H100) | derived_mfu_bf16_peak | 0.020 | 0.018 | -7.919 | regression |
| single_user_dynamic (6 H100) | requests_per_s | 419.784 | 448.919 | 6.940 | improved |
| single_user_dynamic (6 H100) | latency_ms_p50 | 14.665 | 13.212 | 9.911 | improved |
| single_user_dynamic (6 H100) | latency_ms_p95 | 27.157 | 15.809 | 41.786 | improved |
| single_user_dynamic (6 H100) | gpu_utilization_pct_mean | 15.639 | 16.083 | 2.842 | neutral |
| single_user_dynamic (6 H100) | derived_mfu_bf16_peak | 0.019 | 0.020 | 6.940 | improved |
| single_user_static (1 H100) | requests_per_s | 78.544 | 68.831 | -12.366 | regression |
| single_user_static (1 H100) | latency_ms_p50 | 12.237 | 14.329 | -17.089 | regression |
| single_user_static (1 H100) | latency_ms_p95 | 14.573 | 15.974 | -9.614 | regression |
| single_user_static (1 H100) | gpu_utilization_pct_mean | 15.500 | 14.000 | -9.677 | regression |
| single_user_static (1 H100) | derived_mfu_bf16_peak | 0.004 | 0.003 | -12.366 | regression |
| single_user_static (6 H100) | requests_per_s | 441.056 | 499.236 | 13.191 | improved |
| single_user_static (6 H100) | latency_ms_p50 | 13.947 | 11.817 | 15.276 | improved |
| single_user_static (6 H100) | latency_ms_p95 | 21.098 | 15.471 | 26.670 | improved |
| single_user_static (6 H100) | gpu_utilization_pct_mean | 12.750 | 15.833 | 24.183 | improved |
| single_user_static (6 H100) | derived_mfu_bf16_peak | 0.003 | 0.004 | 13.191 | improved |

## Scaling Factors

| transition | retrieve latency factor | fresh-file latency factor |
|---|---:|---:|
| 7 -> 100 docs | 4.295x | 10.405x |
| 100 -> 1000 docs | 8.427x | 8.083x |
| 1000 -> 10000 docs | 10.115x | 9.594x |
| 10000 -> 25000 docs | 2.555x | 2.495x |
| 25000 -> 100000 docs | 3.999x | 4.036x |

## Command to Reproduce

```bash
/private/tmp/inference-compute-hack-modal-prompt/eval/standard_benchmark.py --opt-id OPT-MODAL-001-CODEX-RAG-VALIDATION --name post-merge Modal optimization validation with RAG ladder --agent codex --run-modal --gpu-counts 1,6 --single-requests 32 --multi-requests 96 --single-concurrency 1 --multi-concurrency 32 --dataset-sizes 7 100 1000 10000 25000 100000 --rag-runs 3 --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --prompt-variant compact --notes Post-merge validation of perf/vllm scheduling and H100 replication optimization on current main, including standardized RAG scaling ladder.
```

## Artifacts

| artifact | path |
|---|---|
| config | `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/config.json` |
| aggregated results | `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/aggregated.json` |
| scaling analysis | `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/scaling_analysis.json` |
| ledger entry | `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/ledger_entry.md` |
