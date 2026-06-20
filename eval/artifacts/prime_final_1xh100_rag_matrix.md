# Prime Final H100/RAG Matrix

- model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- vLLM: `0.6.6.post1`
- selected optimization: `max_num_batched_tokens=16384`
- source summary: `eval/artifacts/experiment_results/PRIME_BENCHMARK_SUMMARY.md`

## Actual Prime H100 Results

| H100s | scenario | req/s | p50 ms | p95 ms | GPU util mean/max | power mean/max W | memory max MB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | multi_user_dynamic | 221.581 | 88.319 | 205.335 | 44.0/62.0 | 185.6/186.2 | 74017.0 |
| 1 | multi_user_static | 148.062 | 169.992 | 291.407 | 28.0/47.0 | 161.2/170.4 | 74015.0 |
| 1 | single_user_dynamic | 52.147 | 17.369 | 18.081 | 45.0/62.0 | 171.8/178.2 | 74017.0 |
| 1 | single_user_static | 52.135 | 17.220 | 18.152 | 27.7/48.0 | 134.5/148.4 | 73917.0 |

## RAG Reference

| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |
|---:|---:|---:|---:|
| 7 | 0.012 | 4.885 | 80269.260 |
| 100 | 0.048 | 130.137 | 21034.428 |
| 1000 | 0.425 | 1209.255 | 2353.406 |
| 10000 | 8.980 | 14279.888 | 111.360 |
| 25000 | 27.807 | 45915.450 | 35.963 |
| 100000 | 117.374 | 188308.985 | 8.520 |

## Teammate TL;DR

- Use max_num_batched_tokens=16384: slight throughput win.
- Skip FP8 KV cache with AWQ models: it is about 7% slower in Sasha's run.
- For latency-sensitive endpoints, consider 15ms batch accumulation (SCHED-001 pattern).
- EXP-BATCH-001 and EXP-OVERLAP-001/002 still need application-layer testing.
