# Extension 03: Agentic Iteration Loop Environment

This extension turns the existing query/refine loop into a small RL-style
environment. The goal is not to bolt a full trainer onto the demo yet. The goal
is to expose the right primitives, metrics, and Modal smoke path so we can later
run GRPO/RLAIF experiments on 8 H100s without changing the evaluation surface.

## Thesis

Humans underuse cheap iteration because every failed query costs attention and
time. An agent can spend cheap inference to repeatedly refine the query, verify
evidence, and stop when reward crosses a target. This is the use case where
abundant inference matters: not one better query, but many fast self-corrections.

## Infinite-Compute 3-Axis Mapping

| axis | environment metric | rationale |
|---|---|---|
| memory capacity used | `memory_selectivity` | fraction of stored bytes that are actually positive evidence |
| movement / bandwidth | `movement_selectivity`, `bytes_moved_total` | bytes moved/scored/selected during rollouts |
| answer error | `precision`, `recall`, `F1`, `truth_gain` | whether the final selected evidence supports the task |

Rule of thumb: spend extra compute on query planning and verification, not on
blindly storing or moving more data.

## Applied AI Track Primitives

| primitive | project definition |
|---|---|
| `T` | query-refinement task over a labeled synthetic dynamic corpus |
| `M` | `scorer.score_batch(query, chunks)` returning relevance probabilities |
| `V` | verifier comparing selected chunks to task positive ids |
| `y` | candidate refined query plus selected evidence set |
| `r` | `F1 + recall bonus - byte movement/storage penalties` |

## Metrics

Task-level metrics:

- `task_token_length`
- `branching_factor`
- `reward_variance`
- `max_reward`
- `steps_to_threshold`
- `tool_calls`
- `reasoning_steps`
- `memory_selectivity`
- `movement_selectivity`
- `truth_gain`
- `cost_proxy_model_calls`

Dataset-level metrics:

- `mean_best_reward`
- `mean_best_f1`
- `pass_rate`
- `task_diversity`
- `trajectory_entropy`
- `mean_memory_selectivity`
- `mean_movement_selectivity`
- `agent_vs_human_speedup_estimate`
- `cost_quality_frontier`

These are cheap, pre-training metrics that can be tested for correlation with
post-RL lift once we run cohort-level GRPO/post-training.

## Commands

Local smoke:

```bash
python -m eval.agent_loop --n-docs 1000 --task-count 3 --max-steps 5
```

Modal setup smoke:

```bash
modal run inference/modal_app.py::extension3_agent_loop_smoke \
  --n-docs 1000 \
  --task-count 3 \
  --max-steps 5
```

Artifacts:

| artifact | purpose |
|---|---|
| `eval/artifacts/extension3_agent_loop.json` | local run payload |
| `eval/artifacts/extension3_agent_loop.md` | local human report |
| `eval/artifacts/extension3_agent_loop_modal_smoke.json` | Modal smoke payload |
| `eval/artifacts/extension3_agent_loop_modal_smoke.md` | Modal smoke report |

## Path To 8-H100 Post-Training

1. Generate cohorts by varying one property at a time: task topic, distractor
   density, positive ratio, reward variance, or trajectory entropy.
2. Compute the cheap metrics above for each cohort before training.
3. Run one GRPO/post-training job per cohort with the same model and config.
4. Measure lift on a held-out query-refinement benchmark.
5. Fit metric-to-lift relationships and report R2, RMSE, and rank correlation.
6. Keep the cost-quality Pareto frontier: metric cost versus predictive power.

The current implementation gives us the environment and metrics for steps 1-2
and a Modal smoke path for infrastructure validation. It deliberately avoids
hard-wiring a trainer until the compute budget and selected open RL stack are
available.
