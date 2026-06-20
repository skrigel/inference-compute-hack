# Extension 3 Agent Loop Experiment

- run_id: `agent-loop-64279313`
- commit: `685e451`
- model: `mock-semantic-filter-v0`
- docs per task: `80`
- tasks: `3`
- elapsed_ms: `16.885`

## Dataset Metrics

| metric | value |
|---|---:|
| mean_best_reward | 1.191004 |
| mean_best_f1 | 1.000000 |
| pass_rate | 1.000000 |
| task_diversity | 1.000000 |
| trajectory_entropy | 2.108968 |
| mean_memory_selectivity | 0.089964 |
| mean_movement_selectivity | 0.089964 |
| agent_vs_human_speedup_estimate | 26651.632777 |

## Episodes

| task | topic | best query | best reward | precision | recall | F1 | movement selectivity |
|---|---|---|---:|---:|---:|---:|---:|
| agent-loop-0 | retry_backoff | retry networking layer | 1.191408 | 1.000000 | 1.000000 | 1.000000 | 0.085916 |
| agent-loop-1 | ir_retrieval | retrieval ranking metrics | 1.191148 | 1.000000 | 1.000000 | 1.000000 | 0.088518 |
| agent-loop-2 | cache_threshold | cache threshold drag without rescoring | 1.190454 | 1.000000 | 1.000000 | 1.000000 | 0.095456 |
