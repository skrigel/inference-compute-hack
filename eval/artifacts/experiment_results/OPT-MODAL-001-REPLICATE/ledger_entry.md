### OPT-MODAL-001-REPLICATE: modal-throughput-baseline

- status: proposed | running | applied | rejected | reverted
- owner: agent
- date: 2026-06-20
- commit: `e722980`
- artifacts:
  - `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/config.json`
  - `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/aggregated.json`
  - `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/scaling_analysis.json`
  - `eval/artifacts/experiment_summaries/OPT-MODAL-001-REPLICATE_summary.md`
- Weave run/eval: upload with `python -m eval.upload_weave_results` after freezing a matrix artifact
- hypothesis: fill in before accepting this optimization
- change: fill in exact code/config change
- expected mechanism: fill in why the metric should move

#### Experiment Configuration
- repetitions: baseline artifacts n=1; candidate artifacts n=1; RAG runs=3
- warmup excluded: no
- dataset sizes tested:
  | size tier | doc count | notes |
  |---|---:|---|
  | small | 7 | standardized RAG scaling ladder |
  | small | 100 | standardized RAG scaling ladder |
  | medium | 1000 | standardized RAG scaling ladder |
  | large | 10000 | standardized RAG scaling ladder |
  | xlarge | 25000 | standardized RAG scaling ladder |
  | xxlarge | 100000 | standardized RAG scaling ladder |

#### Quality Gate
- precision: carry forward from `phase04_quality_gate.json` unless rerun
- recall: carry forward from `phase04_quality_gate.json` unless rerun
- F1: must remain >= 0.7
- threshold: carry forward or document new threshold
- verdict: pass | fail | not rerun

#### Performance Delta (with variance)
| workload | dataset | baseline mean +/- std | candidate mean +/- std | delta | p-value | verdict |
|---|---|---:|---:|---:|---:|---|
| multi_user_dynamic (1 H100) | matrix | 301.459 +/- 0.000 | 292.167 +/- 0.000 | -3.082% | N/A | neutral |
| multi_user_dynamic (6 H100) | matrix | 1762.192 +/- 0.000 | 1769.026 +/- 0.000 | 0.388% | N/A | neutral |
| multi_user_static (1 H100) | matrix | 208.341 +/- 0.000 | 237.655 +/- 0.000 | 14.070% | N/A | improved |
| multi_user_static (6 H100) | matrix | 1351.971 +/- 0.000 | 1891.140 +/- 0.000 | 39.880% | N/A | improved |
| single_user_dynamic (1 H100) | matrix | 72.296 +/- 0.000 | 77.354 +/- 0.000 | 6.996% | N/A | improved |
| single_user_dynamic (6 H100) | matrix | 419.784 +/- 0.000 | 493.339 +/- 0.000 | 17.522% | N/A | improved |
| single_user_static (1 H100) | matrix | 78.544 +/- 0.000 | 82.821 +/- 0.000 | 5.445% | N/A | improved |
| single_user_static (6 H100) | matrix | 441.056 +/- 0.000 | 490.557 +/- 0.000 | 11.223% | N/A | improved |

#### Utilization (with variance)
| workload | dataset | metric | baseline mean +/- std | candidate mean +/- std | delta | verdict |
|---|---|---|---:|---:|---:|---|
| multi_user_dynamic (1 H100) | matrix | gpu_utilization_pct_mean | 32.000 +/- 0.000 | 24.500 +/- 0.000 | -23.438% | regression |
| multi_user_dynamic (1 H100) | matrix | derived_mfu_bf16_peak | 0.083 +/- 0.000 | 0.080 +/- 0.000 | -3.082% | neutral |
| multi_user_dynamic (6 H100) | matrix | gpu_utilization_pct_mean | 19.417 +/- 0.000 | 28.583 +/- 0.000 | 47.210% | improved |
| multi_user_dynamic (6 H100) | matrix | derived_mfu_bf16_peak | 0.081 +/- 0.000 | 0.081 +/- 0.000 | 0.388% | neutral |
| multi_user_static (1 H100) | matrix | gpu_utilization_pct_mean | 10.000 +/- 0.000 | 28.000 +/- 0.000 | 180.000% | improved |
| multi_user_static (1 H100) | matrix | derived_mfu_bf16_peak | 0.009 +/- 0.000 | 0.011 +/- 0.000 | 14.070% | improved |
| multi_user_static (6 H100) | matrix | gpu_utilization_pct_mean | 11.417 +/- 0.000 | 15.667 +/- 0.000 | 37.226% | improved |
| multi_user_static (6 H100) | matrix | derived_mfu_bf16_peak | 0.010 +/- 0.000 | 0.014 +/- 0.000 | 39.880% | improved |
| single_user_dynamic (1 H100) | matrix | gpu_utilization_pct_mean | 4.500 +/- 0.000 | 16.500 +/- 0.000 | 266.667% | improved |
| single_user_dynamic (1 H100) | matrix | derived_mfu_bf16_peak | 0.020 +/- 0.000 | 0.021 +/- 0.000 | 6.996% | improved |
| single_user_dynamic (6 H100) | matrix | gpu_utilization_pct_mean | 15.639 +/- 0.000 | 12.000 +/- 0.000 | -23.268% | regression |
| single_user_dynamic (6 H100) | matrix | derived_mfu_bf16_peak | 0.019 +/- 0.000 | 0.022 +/- 0.000 | 17.522% | improved |
| single_user_static (1 H100) | matrix | gpu_utilization_pct_mean | 15.500 +/- 0.000 | 18.500 +/- 0.000 | 19.355% | improved |
| single_user_static (1 H100) | matrix | derived_mfu_bf16_peak | 0.004 +/- 0.000 | 0.004 +/- 0.000 | 5.445% | improved |
| single_user_static (6 H100) | matrix | gpu_utilization_pct_mean | 12.750 +/- 0.000 | 15.250 +/- 0.000 | 19.608% | improved |
| single_user_static (6 H100) | matrix | derived_mfu_bf16_peak | 0.003 +/- 0.000 | 0.004 +/- 0.000 | 11.223% | improved |

#### Scaling Analysis
| transition | retrieve latency factor | fresh-file latency factor | retrieve exponent | fresh-file exponent |
|---|---:|---:|---:|---:|

#### Summary
- regression threshold: 5.0%
- decision: regression
- caveats: fill in infra anomalies, OOMs, JIT/warmup, or missing quality reruns
- next action: fill in
- rollback: fill in
