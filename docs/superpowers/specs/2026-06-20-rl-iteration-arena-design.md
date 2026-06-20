# RL Iteration Arena Design

## Goal

Create a polished, live-feeling visualization for the Extension 3 RL/post-training story. The view should make the core premise legible in a demo: an agent can use cheap inference-time iteration to refine queries far faster than a human, and those rollouts produce task-level and dataset-level metrics that support the Applied AI track.

The first implementation will be a **hybrid live replay**. It will use committed artifacts as source-of-truth data, but animate them as if a live run is unfolding. This keeps the demo reliable while still feeling active and technical.

## Source Data

Primary artifact:

- `eval/artifacts/extension3_agent_loop.json`

The UI will read or embed the following fields:

- `run_id`
- `dataset_metrics.agent_vs_human_speedup_estimate`
- `dataset_metrics.estimated_human_ms`
- `dataset_metrics.agent_elapsed_ms`
- `dataset_metrics.mean_best_reward`
- `dataset_metrics.mean_best_f1`
- `dataset_metrics.pass_rate`
- `dataset_metrics.trajectory_entropy`
- `dataset_metrics.mean_memory_selectivity`
- `dataset_metrics.mean_movement_selectivity`
- `dataset_metrics.cost_quality_frontier`
- `episodes[].target_topic`
- `episodes[].initial_query`
- `episodes[].best_query`
- `episodes[].best_reward`
- `episodes[].best_quality`
- `episodes[].steps[]`

Secondary credibility references can be surfaced in compact copy or a secondary panel:

- `eval/artifacts/prime_smoke/training_runs.md`
- `eval/artifacts/prime_readiness/no_credit_readiness_report.json`
- `eval/configs/extension3_prime/metric_to_lift_schema.example.json`

## Visual Direction

Use a math-animation-inspired aesthetic, similar in spirit to 3Blue1Brown:

- dark theorem-board background
- precise geometric reward curves
- blue/cyan traces for agent iteration and reward improvement
- warm brown/orange accents for evidence, verification, and stop decisions
- clean labels, minimal chrome, and smooth motion
- no marketing hero copy or decorative blobs

The visualization should feel like watching a small proof unfold:

1. A human baseline remains slow and mostly static.
2. The agent proposes query rewrites rapidly.
3. Reward, recall, and F1 rise as the rollout progresses.
4. Evidence/selectivity metrics explain why the result is not just selecting everything.
5. The final metrics lock in as the rollout crosses the threshold.

## Product Shape

Add a new frontend surface called **RL Iteration Arena**. Preferred route:

- `/rl-arena`

It can also be linked from the existing navigation or demo page. The route should not replace the current Search/MCP comparison work that teammates are building.

The first viewport should be the live replay itself, not an explanatory landing page.

## Layout

The view is a full-width application screen with three dense zones:

1. **Animation Stage**
   - Large geometric reward curve.
   - Query proposal nodes moving across iteration steps.
   - Human baseline as a slower gray/brown line.
   - Agent reward path as a bright blue curve.
   - Warm evidence markers where a step reaches supported truth.

2. **Metrics Rail**
   - Headline `34,025x` speed estimate, formatted from the artifact rather than hard-coded where practical.
   - Pass rate.
   - Best F1.
   - Mean best reward.
   - Trajectory entropy.
   - Memory and movement selectivity.

3. **Episode Stack**
   - One row per task/topic:
     - retry/backoff
     - retrieval/ranking
     - cache/threshold
   - Each row shows initial query, best query, reward, and F1.
   - Selecting a row replays that episode’s step sequence on the main stage.

Below the fold or in a compact right/lower panel:

- cost-quality frontier mini-scatter from `cost_quality_frontier`
- post-training training-run summary from `training_runs.md`
- short “why this matters” line tying the view to metric-to-lift claims

## Interaction

Default behavior:

- Auto-play the replay on load.
- Replay loops every few seconds.
- The user can pause/resume.
- The user can choose an episode.
- The user can scrub through steps.

Step animation:

- Current query text updates step by step.
- Reward point moves along the curve.
- Metric tiles count up toward current step values.
- Evidence chips light up when precision/recall/F1 cross threshold.
- Final state freezes briefly on the best step, then loops.

The replay should be deterministic. There should be no dependency on a live backend for the first version.

## Data Flow

Implementation options in order of preference:

1. Convert the JSON artifact into a small frontend fixture under `frontend/src/data/rlArenaData.ts`.
2. Or load a static JSON copy from `frontend/public/`.
3. Avoid runtime filesystem reads from `eval/artifacts` because Vite/browser code cannot directly access repository files outside the served app.

The fixture should preserve enough provenance to make the demo honest:

- source artifact path
- run id
- commit if present
- model id
- timestamp if present

## Components

Create focused components:

- `RLArenaPage`
  - route-level composition and replay state
- `RewardStage`
  - SVG/math-animation stage
- `ReplayControls`
  - play/pause, scrubber, episode selector
- `MetricRail`
  - speed estimate and dataset-level metrics
- `EpisodeStack`
  - per-task rows and selection
- `FrontierMiniChart`
  - cost-quality frontier mini chart

Avoid adding a charting library for the first implementation. SVG is enough and keeps the visual style more controlled.

## Styling

Use the existing frontend CSS approach (`App.v2.css`) and add scoped class names for the RL arena. Do not introduce Tailwind or a new component library.

Responsive behavior:

- Desktop: stage left, metrics/episode stack right.
- Tablet/mobile: stage first, metrics second, episode stack third.
- Text must not overlap at narrow widths.
- Fixed-format elements should use stable dimensions or responsive constraints.

Color direction:

- background: near-black/navy
- primary trace: cyan/blue
- secondary trace: muted slate
- evidence/verification: warm brown/orange
- success: restrained green only where needed

## Success Criteria

- The first screen clearly communicates the agent-vs-human iteration premise in under 5 seconds.
- The UI uses real committed artifact values, including `34,025x`, pass rate, best F1, reward, and per-episode query steps.
- The animation loops smoothly without backend calls.
- The route builds with the existing frontend toolchain.
- The visualization is useful for the demo even if the backend and GPU services are offline.

## Non-Goals

- Do not run live Prime/Modal training from the browser.
- Do not add a backend dependency for the first version.
- Do not claim the hybrid replay is a live training run.
- Do not replace the Search/MCP comparison demo.
- Do not fabricate post-training lift values that are not in the committed artifacts.

## Open Follow-Up

After this replay version works, a later refinement can add an optional “Run live episode” button that calls a backend endpoint and falls back to replay if unavailable.

