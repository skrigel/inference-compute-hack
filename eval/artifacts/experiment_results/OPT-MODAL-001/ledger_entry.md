### OPT-MODAL-001: modal-throughput-baseline

- status: proposed | running | applied | rejected | reverted
- owner: agent
- date: 2026-06-20
- commit: `e722980`
- artifacts:
  - `eval/artifacts/experiment_results/OPT-MODAL-001/config.json`
  - `eval/artifacts/experiment_results/OPT-MODAL-001/aggregated.json`
  - `eval/artifacts/experiment_results/OPT-MODAL-001/scaling_analysis.json`
  - `eval/artifacts/experiment_summaries/OPT-MODAL-001_summary.md`
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
| multi_user_dynamic (1 H100) | matrix | 301.459 +/- 0.000 | 295.538 +/- 0.000 | -1.964% | N/A | neutral |
| multi_user_dynamic (6 H100) | matrix | 1762.192 +/- 0.000 | 1786.504 +/- 0.000 | 1.380% | N/A | neutral |
| multi_user_static (1 H100) | matrix | 208.341 +/- 0.000 | 217.375 +/- 0.000 | 4.336% | N/A | neutral |
| multi_user_static (6 H100) | matrix | 1351.971 +/- 0.000 | 1812.446 +/- 0.000 | 34.060% | N/A | improved |
| single_user_dynamic (1 H100) | matrix | 72.296 +/- 0.000 | 70.400 +/- 0.000 | -2.624% | N/A | neutral |
| single_user_dynamic (6 H100) | matrix | 419.784 +/- 0.000 | 459.391 +/- 0.000 | 9.435% | N/A | improved |
| single_user_static (1 H100) | matrix | 78.544 +/- 0.000 | 82.184 +/- 0.000 | 4.635% | N/A | neutral |
| single_user_static (6 H100) | matrix | 441.056 +/- 0.000 | 479.991 +/- 0.000 | 8.828% | N/A | improved |

#### Utilization (with variance)
| workload | dataset | metric | baseline mean +/- std | candidate mean +/- std | delta | verdict |
|---|---|---|---:|---:|---:|---|
| multi_user_dynamic (1 H100) | matrix | gpu_utilization_pct_mean | 32.000 +/- 0.000 | 31.000 +/- 0.000 | -3.125% | neutral |
| multi_user_dynamic (1 H100) | matrix | derived_mfu_bf16_peak | 0.083 +/- 0.000 | 0.081 +/- 0.000 | -1.964% | neutral |
| multi_user_dynamic (6 H100) | matrix | gpu_utilization_pct_mean | 19.417 +/- 0.000 | 15.000 +/- 0.000 | -22.747% | regression |
| multi_user_dynamic (6 H100) | matrix | derived_mfu_bf16_peak | 0.081 +/- 0.000 | 0.082 +/- 0.000 | 1.380% | neutral |
| multi_user_static (1 H100) | matrix | gpu_utilization_pct_mean | 10.000 +/- 0.000 | 4.000 +/- 0.000 | -60.000% | regression |
| multi_user_static (1 H100) | matrix | derived_mfu_bf16_peak | 0.009 +/- 0.000 | 0.010 +/- 0.000 | 4.336% | neutral |
| multi_user_static (6 H100) | matrix | gpu_utilization_pct_mean | 11.417 +/- 0.000 | 13.250 +/- 0.000 | 16.058% | improved |
| multi_user_static (6 H100) | matrix | derived_mfu_bf16_peak | 0.010 +/- 0.000 | 0.014 +/- 0.000 | 34.060% | improved |
| single_user_dynamic (1 H100) | matrix | gpu_utilization_pct_mean | 4.500 +/- 0.000 | 4.000 +/- 0.000 | -11.111% | regression |
| single_user_dynamic (1 H100) | matrix | derived_mfu_bf16_peak | 0.020 +/- 0.000 | 0.019 +/- 0.000 | -2.624% | neutral |
| single_user_dynamic (6 H100) | matrix | gpu_utilization_pct_mean | 15.639 +/- 0.000 | 16.833 +/- 0.000 | 7.638% | improved |
| single_user_dynamic (6 H100) | matrix | derived_mfu_bf16_peak | 0.019 +/- 0.000 | 0.021 +/- 0.000 | 9.435% | improved |
| single_user_static (1 H100) | matrix | gpu_utilization_pct_mean | 15.500 +/- 0.000 | 15.500 +/- 0.000 | 0.000% | neutral |
| single_user_static (1 H100) | matrix | derived_mfu_bf16_peak | 0.004 +/- 0.000 | 0.004 +/- 0.000 | 4.635% | neutral |
| single_user_static (6 H100) | matrix | gpu_utilization_pct_mean | 12.750 +/- 0.000 | 14.333 +/- 0.000 | 12.418% | improved |
| single_user_static (6 H100) | matrix | derived_mfu_bf16_peak | 0.003 +/- 0.000 | 0.004 +/- 0.000 | 8.828% | improved |

#### Scaling Analysis
| transition | retrieve latency factor | fresh-file latency factor | retrieve exponent | fresh-file exponent |
|---|---:|---:|---:|---:|

#### Summary
- regression threshold: 5.0%
- decision: regression
- caveats: fill in infra anomalies, OOMs, JIT/warmup, or missing quality reruns
- next action: fill in
- rollback: fill in
