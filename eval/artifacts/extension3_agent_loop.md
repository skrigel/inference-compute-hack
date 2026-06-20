# Extension 3 Agent Loop Experiment

- run_id: `agent-loop-93c73ae6`
- commit: `685e451`
- model: `mock-semantic-filter-v0`
- docs per task: `200`
- tasks: `3`
- elapsed_ms: `13.225`

## Dataset Metrics

| metric | value |
|---|---:|
| mean_best_reward | 1.190463 |
| mean_best_f1 | 1.000000 |
| pass_rate | 1.000000 |
| task_diversity | 1.000000 |
| trajectory_entropy | 2.118688 |
| mean_memory_selectivity | 0.095365 |
| mean_movement_selectivity | 0.095365 |
| agent_vs_human_speedup_estimate | 34025.392162 |

## Episodes

| task | topic | best query | best reward | precision | recall | F1 | movement selectivity |
|---|---|---|---:|---:|---:|---:|---:|
| agent-loop-0 | retry_backoff | retry networking layer | 1.190857 | 1.000000 | 1.000000 | 1.000000 | 0.091429 |
| agent-loop-1 | ir_retrieval | retrieval ranking metrics | 1.190637 | 1.000000 | 1.000000 | 1.000000 | 0.093635 |
| agent-loop-2 | cache_threshold | cache threshold drag without rescoring | 1.189897 | 1.000000 | 1.000000 | 1.000000 | 0.101032 |
