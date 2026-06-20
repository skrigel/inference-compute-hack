# Extension 3 Agent Loop Experiment

- run_id: `agent-loop-7d2ffe3d`
- commit: `7da5c2d`
- model: `mock-semantic-filter-v0`
- docs per task: `60`
- tasks: `3`
- elapsed_ms: `12.157`

## Dataset Metrics

| metric | value |
|---|---:|
| mean_best_reward | 1.192035 |
| mean_best_f1 | 1.000000 |
| pass_rate | 1.000000 |
| task_diversity | 1.000000 |
| trajectory_entropy | 2.046663 |
| mean_memory_selectivity | 0.079648 |
| mean_movement_selectivity | 0.079648 |
| agent_vs_human_speedup_estimate | 37014.776381 |

## Episodes

| task | topic | best query | best reward | precision | recall | F1 | movement selectivity |
|---|---|---|---:|---:|---:|---:|---:|
| agent-loop-0 | retry_backoff | retry networking layer | 1.192294 | 1.000000 | 1.000000 | 1.000000 | 0.077063 |
| agent-loop-1 | ir_retrieval | retrieval ranking metrics | 1.192206 | 1.000000 | 1.000000 | 1.000000 | 0.077944 |
| agent-loop-2 | cache_threshold | cache threshold drag without rescoring | 1.191606 | 1.000000 | 1.000000 | 1.000000 | 0.083936 |
