# Phase 04 H100 vs RAG Scenario Matrix

- run_id: `phase04-h100-rag-matrix-1781957974`
- model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- vLLM: `0.22.1`
- prompt_variant: `compact`
- gpu_memory_utilization: `0.92`

| scenario | H100s | req/s | p50 ms | p95 max ms | MFU BF16 | GPU util mean/max | power mean/max W | memory max MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_user_static | 1 | 82.184 | 11.808 | 13.271 | 0.003747 | 15.5/31.0 | 119.7/124.4 | 75439.0 |
| single_user_static | 6 | 479.991 | 12.221 | 14.827 | 0.003647 | 14.3/33.0 | 120.5/129.2 | 75309.0 |
| multi_user_static | 1 | 217.375 | 139.621 | 214.643 | 0.009910 | 4.0/8.0 | 117.9/120.8 | 75309.0 |
| multi_user_static | 6 | 1812.446 | 81.151 | 224.609 | 0.013771 | 13.2/55.0 | 120.9/133.6 | 75309.0 |
| single_user_dynamic | 1 | 70.400 | 14.161 | 15.510 | 0.019192 | 4.0/8.0 | 122.3/127.7 | 75309.0 |
| single_user_dynamic | 6 | 459.391 | 12.750 | 15.670 | 0.020873 | 16.8/33.0 | 124.2/135.0 | 75309.0 |
| multi_user_dynamic | 1 | 295.538 | 86.767 | 177.897 | 0.081261 | 31.0/62.0 | 140.4/165.6 | 75309.0 |
| multi_user_dynamic | 6 | 1786.504 | 88.079 | 223.916 | 0.081869 | 15.0/69.0 | 134.2/166.2 | 75309.0 |

## 1 vs 6 H100 Scaling

| scenario | 1 H100 req/s | 6 H100 req/s | throughput scale | 1 H100 p50 ms | 6 H100 p50 ms | p50 ratio |
|---|---:|---:|---:|---:|---:|---:|
| single_user_static | 82.184 | 479.991 | 5.840x | 11.808 | 12.221 | 1.035x |
| multi_user_static | 217.375 | 1812.446 | 8.338x | 139.621 | 81.151 | 0.581x |
| single_user_dynamic | 70.400 | 459.391 | 6.525x | 14.161 | 12.750 | 0.900x |
| multi_user_dynamic | 295.538 | 1786.504 | 6.045x | 86.767 | 88.079 | 1.015x |

## RAG Reference

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|
| 7 | 0.018 | 0.135 | 54054.068 |
| 100 | 0.084 | 1.430 | 11947.432 |
| 1000 | 0.758 | 15.363 | 1320.132 |
| 10000 | 7.336 | 136.194 | 136.305 |
| 25000 | 20.656 | 358.079 | 48.412 |
| 100000 | 79.020 | 1501.068 | 12.655 |

## Biggest Difference

- `single_user_dynamic` with `6` H100(s) vs RAG at `100000` docs: RAG latency is `117.734x` the H100 p50 latency.

## Refinement Overlap

- **Prefill performance / TTFT / prefill throughput**: Direct. The benchmark is max_tokens=1, so it is effectively prefill-only; server TTFT/e2e/prefill metrics and compact prompts target this.
- **KV cache, prefix caching, and reusing computed data**: Partial. vLLM prefix caching is enabled and static scenarios exercise shared prompts; the current app does not yet implement a prefix tree or persistent document-prefix KV reuse.
- **Hardware utilization, GPU power, GPU memory**: Direct. The matrix samples nvidia-smi during load for GPU utilization, memory used, and power draw in addition to vLLM MFU.
- **Batch size, continuous batching, balancing across chips**: Direct for data-parallel balancing across 1 vs 6 replicas and concurrent-user scenarios; not yet single-document parallel merge.
- **Scheduling optimizations / shortest-job-first**: Measured but not implemented. Dynamic/static and single/multi scenarios expose queue/prefill behavior; JCT-aware scheduling remains a next optimization.
