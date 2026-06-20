### OPT-MODAL-001-CODEX-RAG-VALIDATION: post-merge Modal optimization validation with RAG ladder

- status: rejected
- owner: codex
- date: 2026-06-20
- commit: `7a090a9`
- artifacts:
  - `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/config.json`
  - `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/aggregated.json`
  - `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/scaling_analysis.json`
  - `eval/artifacts/experiment_summaries/OPT-MODAL-001-CODEX-RAG-VALIDATION_summary.md`
- Weave run/eval: upload with `python -m eval.upload_weave_results` after freezing a matrix artifact
- hypothesis: after the merged vLLM scheduling/replication work, the current
  Modal path should preserve or improve the Phase 04 baseline, with the clearest
  wins expected on 6-H100 static and concurrent workloads where data-parallel
  replicas and prefix caching can amortize prefill work.
- change: validated current `main` (`7a090a9`) with the standard benchmark runner
  against the frozen Phase 04 baseline, using Modal vLLM `0.22.1`,
  `--enable-prefix-caching`, `--enable-mfu-metrics`,
  `--gpu-memory-utilization 0.92`, `--max-num-batched-tokens 8192`, compact
  prompts, and the 1-H100/6-H100 static/dynamic workload matrix plus the RAG
  size ladder.
- expected mechanism: six replicas should increase aggregate request capacity;
  prefix caching should reduce repeated static prompt prefill; compact prompts
  and larger batching limits should reduce per-request prefill overhead; MFU,
  GPU utilization, queue, and cache metrics should expose whether the run is
  actually saturating the GPUs.

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
- precision: not rerun in this benchmark
- recall: not rerun in this benchmark
- F1: not rerun in this benchmark; Phase 04 gate remains the last recorded gate
- threshold: not changed
- verdict: not rerun

#### Performance Delta (with variance)
| workload | dataset | baseline mean +/- std | candidate mean +/- std | delta | p-value | verdict |
|---|---|---:|---:|---:|---:|---|
| multi_user_dynamic (1 H100) | matrix | 301.459 +/- 0.000 | 260.994 +/- 0.000 | -13.423% | N/A | regression |
| multi_user_dynamic (6 H100) | matrix | 1762.192 +/- 0.000 | 1520.485 +/- 0.000 | -13.716% | N/A | regression |
| multi_user_static (1 H100) | matrix | 208.341 +/- 0.000 | 226.626 +/- 0.000 | 8.776% | N/A | improved |
| multi_user_static (6 H100) | matrix | 1351.971 +/- 0.000 | 1247.423 +/- 0.000 | -7.733% | N/A | regression |
| single_user_dynamic (1 H100) | matrix | 72.296 +/- 0.000 | 66.572 +/- 0.000 | -7.919% | N/A | regression |
| single_user_dynamic (6 H100) | matrix | 419.784 +/- 0.000 | 448.919 +/- 0.000 | 6.940% | N/A | improved |
| single_user_static (1 H100) | matrix | 78.544 +/- 0.000 | 68.831 +/- 0.000 | -12.366% | N/A | regression |
| single_user_static (6 H100) | matrix | 441.056 +/- 0.000 | 499.236 +/- 0.000 | 13.191% | N/A | improved |

#### Utilization (with variance)
| workload | dataset | metric | baseline mean +/- std | candidate mean +/- std | delta | verdict |
|---|---|---|---:|---:|---:|---|
| multi_user_dynamic (1 H100) | matrix | gpu_utilization_pct_mean | 32.000 +/- 0.000 | 21.000 +/- 0.000 | -34.375% | regression |
| multi_user_dynamic (1 H100) | matrix | derived_mfu_bf16_peak | 0.083 +/- 0.000 | 0.072 +/- 0.000 | -13.423% | regression |
| multi_user_dynamic (6 H100) | matrix | gpu_utilization_pct_mean | 19.417 +/- 0.000 | 22.167 +/- 0.000 | 14.163% | improved |
| multi_user_dynamic (6 H100) | matrix | derived_mfu_bf16_peak | 0.081 +/- 0.000 | 0.070 +/- 0.000 | -13.716% | regression |
| multi_user_static (1 H100) | matrix | gpu_utilization_pct_mean | 10.000 +/- 0.000 | 5.000 +/- 0.000 | -50.000% | regression |
| multi_user_static (1 H100) | matrix | derived_mfu_bf16_peak | 0.009 +/- 0.000 | 0.010 +/- 0.000 | 8.776% | improved |
| multi_user_static (6 H100) | matrix | gpu_utilization_pct_mean | 11.417 +/- 0.000 | 8.500 +/- 0.000 | -25.547% | regression |
| multi_user_static (6 H100) | matrix | derived_mfu_bf16_peak | 0.010 +/- 0.000 | 0.009 +/- 0.000 | -7.733% | regression |
| single_user_dynamic (1 H100) | matrix | gpu_utilization_pct_mean | 4.500 +/- 0.000 | 15.000 +/- 0.000 | 233.333% | improved |
| single_user_dynamic (1 H100) | matrix | derived_mfu_bf16_peak | 0.020 +/- 0.000 | 0.018 +/- 0.000 | -7.919% | regression |
| single_user_dynamic (6 H100) | matrix | gpu_utilization_pct_mean | 15.639 +/- 0.000 | 16.083 +/- 0.000 | 2.842% | neutral |
| single_user_dynamic (6 H100) | matrix | derived_mfu_bf16_peak | 0.019 +/- 0.000 | 0.020 +/- 0.000 | 6.940% | improved |
| single_user_static (1 H100) | matrix | gpu_utilization_pct_mean | 15.500 +/- 0.000 | 14.000 +/- 0.000 | -9.677% | regression |
| single_user_static (1 H100) | matrix | derived_mfu_bf16_peak | 0.004 +/- 0.000 | 0.003 +/- 0.000 | -12.366% | regression |
| single_user_static (6 H100) | matrix | gpu_utilization_pct_mean | 12.750 +/- 0.000 | 15.833 +/- 0.000 | 24.183% | improved |
| single_user_static (6 H100) | matrix | derived_mfu_bf16_peak | 0.003 +/- 0.000 | 0.004 +/- 0.000 | 13.191% | improved |

#### Scaling Analysis
| transition | retrieve latency factor | fresh-file latency factor | retrieve exponent | fresh-file exponent |
|---|---:|---:|---:|---:|
| 7 -> 100 docs | 4.295x | 10.405x | 0.548 | 0.881 |
| 100 -> 1000 docs | 8.427x | 8.083x | 0.926 | 0.908 |
| 1000 -> 10000 docs | 10.115x | 9.594x | 1.005 | 0.982 |
| 10000 -> 25000 docs | 2.555x | 2.495x | 1.024 | 0.998 |
| 25000 -> 100000 docs | 3.999x | 4.036x | 1.000 | 1.006 |

#### Summary
- regression threshold: 5.0%
- decision: regression
- caveats:
  - baseline and candidate are both n=1, so variance and p-values are not useful
    yet.
  - warmup was not excluded; several 6-H100 workers spent roughly 83-99 seconds
    in vLLM engine initialization, CUDA graph capture, and KV-cache setup before
    serving requests.
  - vLLM logged Triton JIT compilation for `_compute_slot_mapping_kernel` during
    measured inference, which can inflate latency tails.
  - Modal worker startup variance and `resource_tracker` leaked-semaphore warnings
    appeared in logs; no request errors were observed, but timing confidence is low.
  - quality, precision, recall, and F1 were not rerun.
- next action: do not accept this optimization as a global replacement. Re-run the
  matrix with at least 3 repetitions, prewarmed workers, warmup excluded, and an
  explicit shape warmup that covers the measured prompt/concurrency mix; then tune
  the multi-user dynamic path and GPU utilization before claiming a win.
- rollback: no runtime rollback is required for this artifact-only validation
  entry. If the current runtime knobs are suspected in production, compare against
  the frozen Phase 04 baseline config and disable the scheduling/routing knobs
  before re-testing.
