# Prime Final Slide Metrics

Generated from:

- `eval/artifacts/prime_final_1xh100_rag_matrix.json`
- `eval/artifacts/prime_final_2xh100_matrix.json`
- `eval/artifacts/experiment_results/PRIME_BENCHMARK_SUMMARY.md`

Charts:

- `eval/artifacts/prime_final_charts/01_measured_h100_throughput.png`
- `eval/artifacts/prime_final_charts/02_measured_latency_p50_p95.png`
- `eval/artifacts/prime_final_charts/03_rag_scaling.png`
- `eval/artifacts/prime_final_charts/04_speedup_vs_100k_rag.png`
- `eval/artifacts/prime_final_charts/05_projected_8x_throughput.png`

## Availability

- 8x H100: unavailable at run time (`total_count=0`).
- 4x H100: unavailable at run time (`total_count=0`).
- Actual measured runs completed on 1x H100 and 2x H100.
- 8x rows below are projections from the measured 1-active-GPU and 2-active-GPU rows on the same 2x H100 pod. Do not label them as measured.

## Selected Optimization

Use the Sasha-pushed throughput config:

- `max_num_batched_tokens=16384`
- keep AWQ Marlin
- keep prefix caching enabled
- do not use FP8 KV cache with AWQ for final throughput numbers

This follows `EXP-MBT-002`, which was the only global throughput-positive pushed vLLM setting. `EXP-SCHED-001` remains useful for latency-sensitive tail consistency, but it is not the final throughput config.

## Measured H100 Results

| Run | scenario | req/s | p50 ms | p95 ms | GPU util mean/max | memory max MB |
|---|---|---:|---:|---:|---:|---:|
| 1x dedicated H100 | multi_user_dynamic | 221.581 | 88.319 | 205.335 | 44.0/62.0 | 74017 |
| 1x dedicated H100 | multi_user_static | 148.062 | 169.992 | 291.407 | 28.0/47.0 | 74015 |
| 1 active H100 on 2x pod | multi_user_dynamic | 303.746 | 68.299 | 176.323 | 75.0/82.0 | 73847 |
| 1 active H100 on 2x pod | multi_user_static | 183.149 | 117.763 | 280.518 | 26.5/53.0 | 73845 |
| 2 active H100s | multi_user_dynamic | 313.896 | 109.089 | 350.412 | 49.2/74.0 | 73933 |
| 2 active H100s | multi_user_static | 307.648 | 96.757 | 282.529 | 33.7/54.0 | 73847 |
| 2 active H100s | single_user_dynamic | 115.992 | 14.718 | 15.177 | 32.5/69.0 | 73847 |
| 2 active H100s | single_user_static | 116.385 | 14.647 | 15.178 | 38.5/82.0 | 73847 |

## RAG Baseline

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|
| 7 | 0.012 | 4.885 | 80269.260 |
| 100 | 0.048 | 130.137 | 21034.428 |
| 1000 | 0.425 | 1209.255 | 2353.406 |
| 10000 | 8.980 | 14279.888 | 111.360 |
| 25000 | 27.807 | 45915.450 | 35.963 |
| 100000 | 117.374 | 188308.985 | 8.520 |

## Best Slide Comparisons

Use large dynamic/static corpus settings for the demo story. RAG is much faster on tiny corpora, so do not lead with 7, 100, or 1000 docs.

| comparison | our measured req/s | RAG retrieve qps | throughput multiple | our p50 ms | RAG retrieve p50 ms | note |
|---|---:|---:|---:|---:|---:|---|
| 2x H100 multi_user_static vs 100k-doc RAG | 307.648 | 8.520 | 36.110x | 96.757 | 117.374 | strongest measured throughput comparison |
| 2x H100 multi_user_dynamic vs 100k-doc RAG | 313.896 | 8.520 | 36.843x | 109.089 | 117.374 | best measured req/s row |
| 1x dedicated H100 multi_user_dynamic vs 100k-doc RAG | 221.581 | 8.520 | 26.008x | 88.319 | 117.374 | clean actual 1x H100 comparison |
| 2x H100 multi_user_static vs 25k-doc RAG | 307.648 | 35.963 | 8.555x | 96.757 | 27.807 | throughput win, latency not a win |
| 2x H100 multi_user_static vs 10k-doc RAG | 307.648 | 111.360 | 2.763x | 96.757 | 8.980 | throughput win, latency not a win |

## Projected 8x H100

Projection source: measured 1-active-GPU and 2-active-GPU rows on the same 2x H100 pod.

| scenario | observed 2x efficiency | projected 8x req/s | p50 ms assumption | p95 ms assumption |
|---|---:|---:|---:|---:|
| multi_user_dynamic | 0.517 | 1255.584 | 109.089 | 350.412 |
| multi_user_static | 0.840 | 1230.592 | 96.757 | 282.529 |
| single_user_dynamic | 0.921 | 463.966 | 14.718 | 15.177 |
| single_user_static | 0.923 | 465.540 | 14.647 | 15.178 |

Against 100k-doc RAG retrieve QPS (`8.520`), the projected 8x static row is `144.44x` higher throughput. Label this as projected, not measured.

## Interpretation

- Best measured slide row: `2x H100 multi_user_dynamic`, `313.896 req/s`.
- Best clean RAG comparison row: `2x H100 multi_user_static vs 100k-doc RAG`, `36.110x` higher throughput.
- Best actual 1x H100 row: `1 active H100 on 2x pod multi_user_dynamic`, `303.746 req/s`; the dedicated 1x pod measured `221.581 req/s`.
- Scaling is workload-sensitive. Static multi-user improves from `183.149 req/s` on 1 active H100 to `307.648 req/s` on 2 active H100s. Dynamic multi-user only improves from `303.746 req/s` to `313.896 req/s`, so do not use that row to claim strong 2x scaling.
- GPU utilization improved materially over the earlier 6% concern. Best observed mean/max utilization was `75.0/82.0%` on the 1-active-H100 multi-user dynamic row inside the 2x pod.
