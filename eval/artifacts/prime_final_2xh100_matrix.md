# Prime Final H100/RAG Matrix

- model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- vLLM: `0.6.6.post1`
- selected optimization: `max_num_batched_tokens=16384`
- source summary: `eval/artifacts/experiment_results/PRIME_BENCHMARK_SUMMARY.md`

## Actual Prime H100 Results

| H100s | scenario | req/s | p50 ms | p95 ms | GPU util mean/max | power mean/max W | memory max MB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | multi_user_dynamic | 303.746 | 68.299 | 176.323 | 75.0/82.0 | 196.0/206.7 | 73847.0 |
| 1 | multi_user_static | 183.149 | 117.763 | 280.518 | 26.5/53.0 | 154.1/157.1 | 73845.0 |
| 1 | single_user_dynamic | 62.954 | 14.654 | 15.121 | 46.0/53.0 | 178.2/184.0 | 73847.0 |
| 1 | single_user_static | 63.052 | 14.528 | 14.990 | 22.0/44.0 | 128.8/135.9 | 73747.0 |
| 2 | multi_user_dynamic | 313.896 | 109.089 | 350.412 | 49.2/74.0 | 184.1/211.4 | 73933.0 |
| 2 | multi_user_static | 307.648 | 96.757 | 282.529 | 33.7/54.0 | 177.5/206.0 | 73847.0 |
| 2 | single_user_dynamic | 115.992 | 14.718 | 15.177 | 32.5/69.0 | 173.3/183.4 | 73847.0 |
| 2 | single_user_static | 116.385 | 14.647 | 15.178 | 38.5/82.0 | 168.0/214.9 | 73847.0 |

## Projected 8xH100 From Actual 1x/2x

These rows are projections, not measured 8xH100 results.

| scenario | observed 2x efficiency | projected 8x req/s linear | projected 8x req/s efficiency-capped | p50 ms assumption | p95 ms assumption |
|---|---:|---:|---:|---:|---:|
| multi_user_dynamic | 0.517 | 1255.584 | 1255.584 | 109.089 | 350.412 |
| multi_user_static | 0.840 | 1230.592 | 1230.592 | 96.757 | 282.529 |
| single_user_dynamic | 0.921 | 463.966 | 463.966 | 14.718 | 15.177 |
| single_user_static | 0.923 | 465.540 | 465.540 | 14.647 | 15.178 |

## RAG Reference

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|

## Teammate TL;DR

- Use max_num_batched_tokens=16384: slight throughput win.
- Skip FP8 KV cache with AWQ models: it is about 7% slower in Sasha's run.
- For latency-sensitive endpoints, consider 15ms batch accumulation (SCHED-001 pattern).
- EXP-BATCH-001 and EXP-OVERLAP-001/002 still need application-layer testing.
