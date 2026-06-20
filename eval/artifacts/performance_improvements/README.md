# Consolidated Performance Improvement Ledger

Generated: 2026-06-20

This file congregates the repo's chart and benchmark artifacts in one place and
summarizes every artifact that shows a performance improvement. It separates
measured wins from projections and theory-only figures so slide claims can stay
traceable.

## Best Measured Claims To Use

| claim | improvement | source artifacts | caveat |
|---|---:|---|---|
| 2x H100 multi-user dynamic beats 100k-doc RAG retrieve throughput | 36.843x higher throughput, 313.896 req/s vs 8.520 RAG qps | `eval/artifacts/prime_final_slide_metrics.md`, `eval/artifacts/prime_final_2xh100_matrix.json`, `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png` | Measured against repo numpy-fallback RAG retrieve-only baseline. |
| 2x H100 multi-user static beats 100k-doc RAG retrieve throughput | 36.110x higher throughput, 307.648 req/s vs 8.520 RAG qps | `eval/artifacts/prime_final_slide_metrics.md`, `eval/artifacts/prime_final_2xh100_matrix.json`, `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png` | Best clean measured large-corpus slide row; RAG is retrieve-only. |
| 1 active H100 on a 2x pod improved over the dedicated 1x H100 run | +37.081% req/s and 22.668% lower p50 latency on multi-user dynamic | `eval/artifacts/prime_final_1xh100_rag_matrix.json`, `eval/artifacts/prime_final_2xh100_matrix.json`, `eval/artifacts/prime_final_charts/01_measured_h100_throughput.png` | Same selected config, different Prime pod context; treat as measured run comparison, not a universal architectural law. |
| 2 active H100s improved over 1 active H100 on static multi-user workload | +67.977% req/s and 17.837% lower p50 latency | `eval/artifacts/prime_final_2xh100_matrix.json`, `eval/artifacts/prime_final_charts/01_measured_h100_throughput.png`, `eval/artifacts/prime_final_charts/02_measured_latency_p50_p95.png` | Dynamic multi-user scaling was weak; use static multi-user for the scaling story. |
| 6x H100 phase-04 run improved throughput over 1x H100 across all workloads | 5.615x to 6.489x higher throughput | `eval/artifacts/phase04_h100_rag_matrix.md`, `eval/artifacts/phase04_h100_rag_matrix.json` | Earlier Modal/vLLM 0.22.1 run; latency improves only for static multi-user. |
| Candidate-set scoping reduced refine-loop compute | 11 chunks scored vs 21 full-rescore chunks; 47.619% less compute, 1.909x work reduction | `eval/artifacts/cut_line_trace.json`, `eval/artifacts/area_under_loop.png` | Measured on deterministic mock scorer over the pinned 7-chunk cut-line corpus. |
| Agent iteration loop beat human-visible query refinement time estimate | 34,025.392x speed estimate, 13.225 ms agent loop vs 450,000 ms human estimate | `eval/artifacts/extension3_agent_loop.md`, `eval/artifacts/extension3_agent_loop.json` | This is an iteration-loop speed estimate from a deterministic environment, not a live user study. |

## Top 4 Focus Claims For Agents

If another agent is making slides, a demo, or a narrative pass, focus on these
four claims first. They are the strongest measured claims because they combine
large visible deltas with a clear reason the architecture matters.

| priority | claim | why it is compelling | source artifacts |
|---:|---|---|---|
| 1 | **2x H100 multi-user dynamic vs 100k-doc RAG:** 36.843x higher throughput, 313.896 req/s vs 8.520 RAG qps. | Best headline throughput win; use this for the largest measured speedup against large-corpus RAG. | `eval/artifacts/prime_final_slide_metrics.md`, `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png`, `eval/artifacts/prime_final_2xh100_matrix.json` |
| 2 | **2x H100 multi-user static vs 100k-doc RAG:** 36.110x higher throughput and 17.565% lower p50 latency. | Cleanest large-corpus comparison because both throughput and latency improve. | `eval/artifacts/prime_final_slide_metrics.md`, `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png`, `eval/artifacts/prime_final_2xh100_matrix.json` |
| 3 | **6x H100 phase-04 scaling vs 1x H100:** 5.615x to 6.489x higher throughput depending on workload. | Best "compute scaling works" claim; it directly supports the project thesis that the approach improves as compute scales. | `eval/artifacts/phase04_h100_rag_matrix.md`, `eval/artifacts/phase04_h100_rag_matrix.json` |
| 4 | **Candidate-set scoping reduces refine-loop compute:** 11 chunks scored vs 21 full-rescore chunks, 47.619% less compute. | Best architecture-specific claim; it shows the system improves by avoiding repeated work, not only by adding GPUs. | `eval/artifacts/cut_line_trace.json`, `eval/artifacts/area_under_loop.png` |

## Chart Inventory And Improvement Summary

| chart | what it shows | improvement-bearing summary | measured? |
|---|---|---|---|
| `eval/artifacts/area_under_loop.png` | Refine-loop cumulative work. | Scoped loop scores 11 chunks vs 21 for full rescore, a 47.619% compute reduction. | Yes, mock scorer. |
| `eval/artifacts/prime_final_charts/01_measured_h100_throughput.png` | Prime 1x, 1-active-on-2x, and 2-active H100 throughput. | Best measured row is 2x multi-user dynamic at 313.896 req/s; best static scaling row is 1-active to 2-active multi-user static, +67.977% req/s. | Yes. |
| `eval/artifacts/prime_final_charts/02_measured_latency_p50_p95.png` | p50/p95 latency by Prime workload. | 2-active multi-user static lowers p50 from 117.763 ms to 96.757 ms vs the 1-active-on-2x row, a 17.837% p50 improvement. | Yes. |
| `eval/artifacts/prime_final_charts/03_rag_scaling.png` | RAG retrieve and fresh-file scaling by corpus size. | Shows why the strongest story is large corpora: RAG retrieve qps falls from 80,269.260 at 7 docs to 8.520 at 100k docs. | Yes, RAG baseline. |
| `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png` | H100 throughput multiples vs 100k-doc RAG. | Measured 2x rows reach 36.843x dynamic and 36.110x static throughput vs 100k-doc RAG. | Yes for 1x/2x; 8x entries projected. |
| `eval/artifacts/prime_final_charts/05_projected_8x_throughput.png` | Measured 2x vs projected 8x throughput. | Projected 8x multi-user dynamic is 1,255.584 req/s and projected static is 1,230.592 req/s. | Projection, not measured. |
| `performance/figures/1_roofline.png` and `.svg` | Theoretical compute roofline. | Supports the prefill-as-compute-bound framing; not a benchmark win by itself. | Theory only. |
| `performance/figures/2_mfu_waterfall.png` and `.svg` | Theoretical MFU ladder framing. | Helps explain where throughput should improve as MFU rises; measured MFU lives in H100 artifacts instead. | Theory only. |
| `performance/figures/3_area_under_loop.png` and `.svg` | Theoretical/refine-loop area model. | Same improvement mechanism as `eval/artifacts/area_under_loop.png`: scoped cumulative work flattens. | Theory/generated. |
| `performance/figures/4_compute_vs_churn.png` and `.svg` | Recompute-over-store vs RAG churn model. | Explains why dynamic/fresh data is the favorable regime; not a measured benchmark win. | Theory only. |
| `performance/figures/5_kv_capacity.png` and `.svg` | Warm-KV capacity model. | Shows capacity constraints and why lower-precision KV matters; not a measured runtime win yet. | Theory only. |

## Benchmark Improvements

### Final Prime H100/RAG Matrix

Sources:

- `eval/artifacts/prime_final_slide_metrics.md`
- `eval/artifacts/prime_final_1xh100_rag_matrix.md`
- `eval/artifacts/prime_final_1xh100_rag_matrix.json`
- `eval/artifacts/prime_final_2xh100_matrix.md`
- `eval/artifacts/prime_final_2xh100_matrix.json`
- `eval/artifacts/prime_final_charts/*.png`

Selected optimization:

- Use `max_num_batched_tokens=16384`.
- Keep AWQ Marlin and prefix caching.
- Do not use FP8 KV cache with AWQ for final throughput numbers.

Measured large-corpus wins vs 100k-doc RAG:

| comparison | our req/s | RAG qps | throughput multiple | our p50 ms | RAG p50 ms | p50 improvement |
|---|---:|---:|---:|---:|---:|---:|
| 2x H100 multi-user dynamic vs 100k-doc RAG | 313.896 | 8.520 | 36.843x | 109.089 | 117.374 | 7.059% lower |
| 2x H100 multi-user static vs 100k-doc RAG | 307.648 | 8.520 | 36.110x | 96.757 | 117.374 | 17.565% lower |
| 1 active H100 on 2x pod multi-user dynamic vs 100k-doc RAG | 303.746 | 8.520 | 35.652x | 68.299 | 117.374 | 41.811% lower |
| 1x dedicated H100 multi-user dynamic vs 100k-doc RAG | 221.581 | 8.520 | 26.008x | 88.319 | 117.374 | 24.755% lower |

Measured Prime run-to-run improvements:

| comparison | workload | throughput improvement | latency improvement | utilization note |
|---|---|---:|---:|---|
| 1 active H100 on 2x pod vs 1x dedicated H100 | multi-user dynamic | +37.081% req/s | 22.668% lower p50 | GPU util mean improved from 44.0% to 75.0%. |
| 1 active H100 on 2x pod vs 1x dedicated H100 | multi-user static | +23.697% req/s | 30.724% lower p50 | GPU util mean slightly lower, 28.0% to 26.5%. |
| 1 active H100 on 2x pod vs 1x dedicated H100 | single-user dynamic | +20.724% req/s | 15.632% lower p50 | GPU util mean 45.0% to 46.0%. |
| 1 active H100 on 2x pod vs 1x dedicated H100 | single-user static | +20.940% req/s | 15.634% lower p50 | GPU util mean fell; use for latency/throughput, not utilization. |
| 2 active H100s vs 1 active H100 on same 2x pod | multi-user static | +67.977% req/s | 17.837% lower p50 | Strongest measured 2x scaling row. |
| 2 active H100s vs 1 active H100 on same 2x pod | single-user dynamic | +84.248% req/s | p50 roughly flat | Good throughput scaling; latency neutral/slightly worse. |
| 2 active H100s vs 1 active H100 on same 2x pod | single-user static | +84.587% req/s | p50 roughly flat | Good throughput scaling; latency neutral/slightly worse. |

Projected, not measured:

| projection | projected req/s | vs 100k-doc RAG qps |
|---|---:|---:|
| 8x H100 multi-user dynamic | 1,255.584 | 147.37x |
| 8x H100 multi-user static | 1,230.592 | 144.44x |

### Sasha/Prime vLLM Config Experiments

Sources:

- `eval/artifacts/experiment_results/PRIME_BENCHMARK_SUMMARY.md`
- `eval/artifacts/experiment_results/EXP-MBT-001/run_001.json`
- `eval/artifacts/experiment_results/EXP-MBT-002/run_001.json`
- `eval/artifacts/experiment_results/EXP-SCHED-001/run_001.json`
- `eval/artifacts/experiment_results/EXP-LENBIN-001/run_001.json`
- `eval/artifacts/experiment_results/EXP-FP8-001/run_001.json`

Baseline: `EXP-MBT-001`, `max_num_batched_tokens=12288`, 132.744 req/s, p50
182.936 ms, p95 503.858 ms, p99 648.963 ms.

| experiment | improvement shown | regression/caveat | decision |
|---|---|---|---|
| `EXP-MBT-002` max batched tokens 16384 | Throughput improves +0.471% to 133.369 req/s; p95 improves 5.586% to 475.713 ms. | p50 worsens 5.558%; p99 worsens 5.279%. | Use as final throughput config because it is the only global throughput-positive pushed vLLM setting. |
| `EXP-SCHED-001` 15 ms batch accumulation | p50 improves 10.007%; p95 improves 54.106%; p99 improves 62.981%. | Throughput regresses 6.530%. | Useful for latency-sensitive endpoints, not final throughput config. |
| `EXP-LENBIN-001` input-length binning | p95 improves 23.767%; p99 improves 10.067%. | Throughput regresses 3.216%; p50 worsens 8.688%. | Useful when tail latency matters with mixed prompt lengths. |
| `EXP-FP8-001` FP8 KV cache | p99 improves 14.094%; p95 is roughly neutral (+0.497%). | Throughput regresses 7.354%; p50 worsens 20.057%. | Do not use with AWQ for final numbers. |

Config-only/no-result entries:

- `EXP-BATCH-001/config.json`: application batch-size experiment configured, no run result.
- `EXP-OVERLAP-001/config.json`: 10% overlap experiment configured, no run result.
- `EXP-OVERLAP-002/config.json`: 20% overlap experiment configured, no run result.

### Phase 04 1x vs 6x H100 Matrix

Sources:

- `eval/artifacts/phase04_h100_rag_matrix.md`
- `eval/artifacts/phase04_h100_rag_matrix.json`
- `eval/artifacts/phase04_rag_vs_6xh100.md`
- `eval/artifacts/phase04_rag_vs_6xh100.json`

Measured 1x to 6x scaling:

| workload | 1 H100 req/s | 6 H100 req/s | throughput scale | p50 change | utilization note |
|---|---:|---:|---:|---:|---|
| multi-user dynamic | 301.459 | 1,762.192 | 5.846x | p50 6.452% worse | GPU util mean lower per GPU, 32.0% to 19.4%. |
| multi-user static | 208.341 | 1,351.971 | 6.489x | p50 16.402% better | Best phase-04 scaling and latency row. |
| single-user dynamic | 72.296 | 419.784 | 5.806x | p50 8.295% worse | GPU util mean improves 4.5% to 15.6%. |
| single-user static | 78.544 | 441.056 | 5.615x | p50 13.974% worse | Throughput scales, latency does not. |

Large-corpus RAG comparisons from the same phase-04 matrix:

| comparison | improvement |
|---|---:|
| 6x H100 multi-user dynamic vs 25k-doc RAG fresh-file path | 887.223x higher qps; RAG fresh-file latency is 5.470x the H100 p50. |
| 6x H100 multi-user dynamic vs 10k-doc RAG fresh-file path | 394.281x higher qps; RAG fresh-file latency is 2.431x the H100 p50. |
| 6x H100 single-user dynamic vs 25k-doc RAG fresh-file path | 211.352x higher qps; RAG fresh-file latency is 34.331x the H100 p50. |
| 6x H100 multi-user static vs 25k-doc RAG retrieve-only path | 187.620x higher qps; RAG retrieve latency is 1.050x the H100 p50. |

Older focused 6x run:

- `eval/artifacts/phase04_rag_vs_6xh100.md` records 1,578.154 req/s,
  91,286.318 tok/s, p50 45.765 ms, p95 182.296 ms.
- RAG wins on tiny corpora; ours crosses over for large/fresh-file settings.

### Standard Benchmark Optimization Runs

Sources:

- `eval/artifacts/experiment_results/OPT-MODAL-001/ledger_entry.md`
- `eval/artifacts/experiment_results/OPT-MODAL-001-REPLICATE/ledger_entry.md`
- `eval/artifacts/experiment_results/OPT-MODAL-001-CODEX-RAG-VALIDATION/ledger_entry.md`
- matching `aggregated.json` and `eval/artifacts/experiment_summaries/*_summary.md`

These runs are mixed and marked low-confidence or rejected overall, but they do
contain improved sub-rows worth knowing about.

| run | improved performance rows | utilization improvements | caveat |
|---|---|---|---|
| `OPT-MODAL-001` | 6-H100 multi-user static +34.060% req/s, 38.614% lower p50, 8.125% lower p95; 6-H100 single-user dynamic +9.435% req/s, 13.064% lower p50, 42.298% lower p95; 6-H100 single-user static +8.828% req/s, 12.378% lower p50, 29.722% lower p95. | 6-H100 static/dynamic MFU and GPU util improved in several rows. | Overall summary says rejected because other rows regressed and n=1. |
| `OPT-MODAL-001-REPLICATE` | 6-H100 multi-user static +39.880% req/s, 39.995% lower p50, 22.182% lower p95; 1-H100 multi-user static +14.070% req/s; 1-H100 single-user dynamic +6.996% req/s; 6-H100 single-user dynamic +17.522% req/s; 1-H100 single-user static +5.445% req/s; 6-H100 single-user static +11.223% req/s. | Best utility lift: 1-H100 single-user dynamic GPU util +266.667%; 1-H100 multi-user static GPU util +180.000%; 6-H100 multi-user dynamic GPU util +47.210%. | Overall summary says rejected/low confidence because n=1 and quality reruns absent. |
| `OPT-MODAL-001-CODEX-RAG-VALIDATION` | 1-H100 multi-user static +8.776% req/s and 9.280% lower p50; 6-H100 single-user dynamic +6.940% req/s, 9.911% lower p50, 41.786% lower p95; 6-H100 single-user static +13.191% req/s, 15.276% lower p50, 26.670% lower p95. | 1-H100 single-user dynamic GPU util +233.333%; 6-H100 single-user static GPU util +24.183%. | Overall verdict rejected because major multi-user dynamic/static rows regressed and quality was not rerun. |

### Refine Loop / Fresh Data Benchmarks

Sources:

- `eval/artifacts/cut_line_trace.json`
- `eval/artifacts/area_under_loop.png`
- `eval/SLIDE.md`

Improvement summary:

| mechanism | improvement | note |
|---|---:|---|
| Candidate-set scoping over full rescore | 11 chunks scored vs 21, 47.619% less compute | Deterministic mock scorer over 3-turn loop. |
| Click-NOT refine | 0 chunks scored on that turn | Uses existing candidate/cache state rather than re-querying all chunks. |
| Threshold drag | 0 chunks scored | Demonstrates zero-inference recut of cached scores. |
| Fresh-file path | 0 derived bytes written | Structural advantage vs RAG re-indexing; real timing is in phase-04 RAG artifacts. |

### Extension 3 Agent Iteration Loop

Sources:

- `eval/artifacts/extension3_agent_loop.md`
- `eval/artifacts/extension3_agent_loop.json`
- `eval/artifacts/extension3_agent_loop_modal_smoke.md`
- `eval/artifacts/extension3_agent_loop_modal_smoke.json`

Improvement summary:

| run | speed / quality result | selectivity result | caveat |
|---|---|---|---|
| `extension3_agent_loop` | 34,025.392x agent-vs-human iteration speed estimate; mean best F1 1.000; pass rate 1.000. | Mean memory/movement selectivity 9.537%. | Deterministic environment; use as agent-iteration story, not live human timing. |
| `extension3_agent_loop_modal_smoke` | 37,014.776x speed estimate; mean best F1 1.000; pass rate 1.000. | Mean memory/movement selectivity 7.965%. | Smaller 60-doc-per-task smoke run. |

### Quality Threshold Calibration

Sources:

- `eval/artifacts/phase04_quality_gate_default_threshold.md`
- `eval/artifacts/phase04_quality_gate.md`

This is a quality improvement rather than throughput, but it protects the
performance story from optimizing a bad operating point.

| threshold | precision | recall | F1 | improvement |
|---:|---:|---:|---:|---|
| default 0.5 | 1.000 | 0.111 | 0.200 | baseline |
| calibrated ~0.0164 | 1.000 | 0.889 | 0.941 | Recall improves 8.0x; F1 improves 4.706x. |

## Artifacts Scanned With No Global Improvement Claim

| artifact | reason not used as an improvement claim |
|---|---|
| `eval/artifacts/phase04_modal_openai_server_benchmark.json` | Baseline OpenAI-server benchmark with vLLM/MFU metrics; not a comparison run. |
| `eval/artifacts/phase0_h100_smoke.md` | Deferred status note; no measured H100 result. |
| `eval/artifacts/prime_readiness/*` | Readiness/config schema artifacts, not benchmark results. |
| `eval/artifacts/experiment_summaries/CONSOLIDATED_SUMMARY.md` | Stale relative to raw `EXP-* /run_001.json`; says some results are unavailable even when raw runs exist. |
| `eval/artifacts/experiment_results/EXP-BATCH-001/config.json` | Config-only; no run result. |
| `eval/artifacts/experiment_results/EXP-OVERLAP-001/config.json` | Config-only; no run result. |
| `eval/artifacts/experiment_results/EXP-OVERLAP-002/config.json` | Config-only; no run result. |

## Recommended Slide Ordering

1. `eval/artifacts/area_under_loop.png`: show why iteration gets cheaper.
2. `eval/artifacts/prime_final_charts/01_measured_h100_throughput.png`: show measured throughput.
3. `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png`: show the large-corpus RAG win.
4. `eval/artifacts/prime_final_charts/02_measured_latency_p50_p95.png`: show latency is still controlled.
5. `eval/artifacts/extension3_agent_loop.md`: show the agent-iteration use case.
6. `performance/figures/4_compute_vs_churn.png`: use as theory framing for dynamic corpora.

## Claim Hygiene

- Label 8x H100 rows as projected, not measured.
- Say RAG is faster on tiny corpora and lead with 25k/100k-doc or fresh-file
  settings.
- Do not use FP8 KV cache with AWQ as a positive config despite one p99
  improvement row; throughput and p50 regress.
- Do not claim accepted optimization status for `OPT-MODAL-001*` runs; they have
  useful improved sub-rows but were rejected or low-confidence overall.
- Treat `performance/figures/*` as theory/supporting figures unless paired with
  measured eval artifacts.
