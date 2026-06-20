# Phase 04 H100 vs RAG Scenario Matrix

- run_id: `phase04-h100-rag-matrix-1781969758`
- model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- vLLM: `0.22.1`
- prompt_variant: `compact`
- gpu_memory_utilization: `0.92`

| scenario | H100s | req/s | p50 ms | p95 max ms | MFU BF16 | GPU util mean/max | power mean/max W | memory max MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_user_static | 1 | 82.821 | 11.893 | 13.404 | 0.003776 | 18.5/34.0 | 125.5/130.8 | 75309.0 |
| single_user_static | 6 | 490.557 | 11.982 | 14.486 | 0.003727 | 15.2/34.0 | 126.3/137.1 | 75309.0 |
| multi_user_static | 1 | 237.655 | 129.795 | 212.674 | 0.010834 | 28.0/56.0 | 130.5/136.6 | 75309.0 |
| multi_user_static | 6 | 1891.140 | 79.326 | 190.243 | 0.014369 | 15.7/63.0 | 129.1/145.1 | 75309.0 |
| single_user_dynamic | 1 | 77.354 | 12.807 | 14.008 | 0.021088 | 16.5/33.0 | 130.6/140.0 | 75309.0 |
| single_user_dynamic | 6 | 493.339 | 11.987 | 14.434 | 0.022415 | 12.0/34.0 | 129.8/149.2 | 75309.0 |
| multi_user_dynamic | 1 | 292.167 | 92.100 | 205.427 | 0.080334 | 24.5/49.0 | 144.4/163.7 | 75309.0 |
| multi_user_dynamic | 6 | 1769.026 | 89.012 | 213.867 | 0.081068 | 28.6/72.0 | 139.9/163.4 | 75309.0 |

## 1 vs 6 H100 Scaling

| scenario | 1 H100 req/s | 6 H100 req/s | throughput scale | 1 H100 p50 ms | 6 H100 p50 ms | p50 ratio |
|---|---:|---:|---:|---:|---:|---:|
| single_user_static | 82.821 | 490.557 | 5.923x | 11.893 | 11.982 | 1.007x |
| multi_user_static | 237.655 | 1891.140 | 7.958x | 129.795 | 79.326 | 0.611x |
| single_user_dynamic | 77.354 | 493.339 | 6.378x | 12.807 | 11.987 | 0.936x |
| multi_user_dynamic | 292.167 | 1769.026 | 6.055x | 92.100 | 89.012 | 0.966x |

## RAG Reference

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|
| 7 | 0.090 | 0.432 | 11111.113 |
| 100 | 0.202 | 3.426 | 4945.598 |
| 1000 | 0.722 | 21.405 | 1385.425 |
| 10000 | 7.461 | 150.700 | 134.032 |
| 25000 | 18.703 | 383.338 | 53.469 |
| 100000 | 81.136 | 1488.488 | 12.325 |

## Biggest Difference

- `single_user_dynamic` with `6` H100(s) vs RAG at `100000` docs: RAG latency is `124.175x` the H100 p50 latency.

## Refinement Overlap

- **Prefill performance / TTFT / prefill throughput**: Direct. The benchmark is max_tokens=1, so it is effectively prefill-only; server TTFT/e2e/prefill metrics and compact prompts target this.
- **KV cache, prefix caching, and reusing computed data**: Partial. vLLM prefix caching is enabled and static scenarios exercise shared prompts; the current app does not yet implement a prefix tree or persistent document-prefix KV reuse.
- **Hardware utilization, GPU power, GPU memory**: Direct. The matrix samples nvidia-smi during load for GPU utilization, memory used, and power draw in addition to vLLM MFU.
- **Batch size, continuous batching, balancing across chips**: Direct for data-parallel balancing across 1 vs 6 replicas and concurrent-user scenarios; not yet single-document parallel merge.
- **Scheduling optimizations / shortest-job-first**: Measured but not implemented. Dynamic/static and single/multi scenarios expose queue/prefill behavior; JCT-aware scheduling remains a next optimization.
