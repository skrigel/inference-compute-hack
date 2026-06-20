# Phase 04 H100 vs RAG Scenario Matrix

- run_id: `phase04-h100-rag-matrix-1781972347`
- model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- vLLM: `0.22.1`
- prompt_variant: `compact`
- gpu_memory_utilization: `0.92`

| scenario | H100s | req/s | p50 ms | p95 max ms | MFU BF16 | GPU util mean/max | power mean/max W | memory max MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_user_static | 1 | 68.831 | 14.329 | 15.974 | 0.003138 | 14.0/27.0 | 123.9/129.2 | 75309.0 |
| single_user_static | 6 | 499.236 | 11.817 | 15.471 | 0.003793 | 15.8/36.0 | 128.2/138.8 | 75309.0 |
| multi_user_static | 1 | 226.626 | 143.460 | 202.462 | 0.010331 | 5.0/5.0 | 128.4/132.3 | 75309.0 |
| multi_user_static | 6 | 1247.423 | 153.669 | 287.155 | 0.009478 | 8.5/43.0 | 127.1/147.7 | 75309.0 |
| single_user_dynamic | 1 | 66.572 | 14.993 | 16.994 | 0.018148 | 15.0/30.0 | 126.8/134.8 | 75309.0 |
| single_user_dynamic | 6 | 448.919 | 13.212 | 15.809 | 0.020397 | 16.1/37.0 | 128.8/144.1 | 75309.0 |
| multi_user_dynamic | 1 | 260.994 | 91.680 | 188.563 | 0.071763 | 21.0/42.0 | 133.4/150.1 | 75309.0 |
| multi_user_dynamic | 6 | 1520.485 | 111.883 | 253.881 | 0.069679 | 22.2/69.0 | 136.1/168.2 | 75309.0 |

## 1 vs 6 H100 Scaling

| scenario | 1 H100 req/s | 6 H100 req/s | throughput scale | 1 H100 p50 ms | 6 H100 p50 ms | p50 ratio |
|---|---:|---:|---:|---:|---:|---:|
| single_user_static | 68.831 | 499.236 | 7.253x | 14.329 | 11.817 | 0.825x |
| multi_user_static | 226.626 | 1247.423 | 5.504x | 143.460 | 153.669 | 1.071x |
| single_user_dynamic | 66.572 | 448.919 | 6.743x | 14.993 | 13.212 | 0.881x |
| multi_user_dynamic | 260.994 | 1520.485 | 5.826x | 91.680 | 111.883 | 1.220x |

## RAG Reference

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|
| 7 | 0.028 | 0.251 | 35346.907 |
| 100 | 0.141 | 2.052 | 7079.648 |
| 1000 | 0.879 | 17.020 | 1137.979 |
| 10000 | 3.813 | 94.462 | 262.275 |
| 25000 | 9.454 | 155.402 | 105.778 |
| 100000 | 38.418 | 628.217 | 26.029 |

## Biggest Difference

- `single_user_dynamic` with `6` H100(s) vs RAG at `100000` docs: RAG latency is `47.549x` the H100 p50 latency.

## Refinement Overlap

- **Prefill performance / TTFT / prefill throughput**: Direct. The benchmark is max_tokens=1, so it is effectively prefill-only; server TTFT/e2e/prefill metrics and compact prompts target this.
- **KV cache, prefix caching, and reusing computed data**: Partial. vLLM prefix caching is enabled and static scenarios exercise shared prompts; the current app does not yet implement a prefix tree or persistent document-prefix KV reuse.
- **Hardware utilization, GPU power, GPU memory**: Direct. The matrix samples nvidia-smi during load for GPU utilization, memory used, and power draw in addition to vLLM MFU.
- **Batch size, continuous batching, balancing across chips**: Direct for data-parallel balancing across 1 vs 6 replicas and concurrent-user scenarios; not yet single-document parallel merge.
- **Scheduling optimizations / shortest-job-first**: Measured but not implemented. Dynamic/static and single/multi scenarios expose queue/prefill behavior; JCT-aware scheduling remains a next optimization.
