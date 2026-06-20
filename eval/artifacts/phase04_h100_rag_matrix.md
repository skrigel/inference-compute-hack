# Phase 04 H100 vs RAG Scenario Matrix

- run_id: `phase04-h100-rag-matrix-1781946340`
- model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- vLLM: `0.22.1`
- prompt_variant: `compact`
- gpu_memory_utilization: `0.92`

| scenario | H100s | req/s | p50 ms | p95 max ms | MFU BF16 | GPU util mean/max | power mean/max W | memory max MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_user_dynamic | 1 | 301.459 | 86.463 | 160.617 | 0.082889 | 32.0/64.0 | 134.7/153.1 | 75309.0 |
| multi_user_dynamic | 6 | 1762.192 | 92.042 | 212.250 | 0.080755 | 19.4/63.0 | 138.5/168.0 | 75309.0 |
| multi_user_static | 1 | 208.341 | 158.134 | 205.087 | 0.009498 | 10.0/20.0 | 122.6/128.4 | 75309.0 |
| multi_user_static | 6 | 1351.971 | 132.197 | 244.472 | 0.010272 | 11.4/52.0 | 123.4/134.8 | 75309.0 |
| single_user_dynamic | 1 | 72.296 | 13.542 | 15.570 | 0.019709 | 4.5/9.0 | 122.0/128.2 | 75309.0 |
| single_user_dynamic | 6 | 419.784 | 14.665 | 27.157 | 0.019073 | 15.6/38.0 | 129.0/145.1 | 75309.0 |
| single_user_static | 1 | 78.544 | 12.237 | 14.573 | 0.003581 | 15.5/31.0 | 119.7/124.1 | 75309.0 |
| single_user_static | 6 | 441.056 | 13.947 | 21.098 | 0.003351 | 12.8/34.0 | 123.4/131.9 | 75309.0 |

## 1 vs 6 H100 Scaling

| scenario | 1 H100 req/s | 6 H100 req/s | throughput scale | 1 H100 p50 ms | 6 H100 p50 ms | p50 ratio |
|---|---:|---:|---:|---:|---:|---:|
| multi_user_dynamic | 301.459 | 1762.192 | 5.846x | 86.463 | 92.042 | 1.065x |
| multi_user_static | 208.341 | 1351.971 | 6.489x | 158.134 | 132.197 | 0.836x |
| single_user_dynamic | 72.296 | 419.784 | 5.806x | 13.542 | 14.665 | 1.083x |
| single_user_static | 78.544 | 441.056 | 5.615x | 12.237 | 13.947 | 1.140x |

## RAG Reference

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|
| 7 | 0.188 | 0.650 | 5327.423 |
| 1000 | 6.674 | 40.297 | 149.824 |
| 10000 | 55.398 | 223.745 | 18.051 |
| 25000 | 138.775 | 503.477 | 7.206 |

## Biggest Difference

- `single_user_dynamic` with `1` H100(s) vs RAG at `25000` docs: RAG latency is `37.178x` the H100 p50 latency.

## Refinement Overlap

- **Prefill performance / TTFT / prefill throughput**: Direct. The benchmark is max_tokens=1, so it is effectively prefill-only; server TTFT/e2e/prefill metrics and compact prompts target this.
- **KV cache, prefix caching, and reusing computed data**: Partial. vLLM prefix caching is enabled and static scenarios exercise shared prompts; the current app does not yet implement a prefix tree or persistent document-prefix KV reuse.
- **Hardware utilization, GPU power, GPU memory**: Direct. The matrix samples nvidia-smi during load for GPU utilization, memory used, and power draw in addition to vLLM MFU.
- **Batch size, continuous batching, balancing across chips**: Direct for data-parallel balancing across 1 vs 6 replicas and concurrent-user scenarios; not yet single-document parallel merge.
- **Scheduling optimizations / shortest-job-first**: Measured but not implemented. Dynamic/static and single/multi scenarios expose queue/prefill behavior; JCT-aware scheduling remains a next optimization.
