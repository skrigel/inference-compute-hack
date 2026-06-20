# Optimization Results Ledger

This is the canonical performance ledger for the project. It keeps quality, latency,
throughput, utilization, and RAG comparison results in one place so future
optimizations can be judged against objective baselines instead of memory.

## Rules

- Do not compare unlike workloads. Static, dynamic, single-user, multi-user,
  1-H100, and 6-H100 rows are separate baselines.
- Do not accept a speed optimization unless the quality gate is still valid.
- Every claimed win must point to an artifact path, run id, commit, model, vLLM
  version, prompt variant, GPU count, concurrency, request count, and caveats.
- Treat utilization as a first-class metric: MFU alone is not enough. Capture
  `nvidia-smi` GPU utilization, memory, and power during the measured request window.
- RAG comparisons must say whether the metric is static retrieve-only latency or
  dynamic fresh-file total latency.

### Repetition and Dataset Size Requirements

- **Minimum repetitions:** Every experiment configuration must be run at least 3 times.
  Report mean, standard deviation, and min/max for all primary metrics.
- **Dataset size progression:** Test each optimization at multiple dataset/corpus sizes
  to characterize scaling behavior. Standard sizes: `small` (≤100 docs), `medium`
  (1K docs), `large` (10K docs), `xlarge` (25K+ docs), `xxlarge` (100K+ docs).
  The standard runner preserves the current small sizes and adds larger tiers:
  `7`, `100`, `1_000`, `10_000`, `25_000`, and `100_000` docs. Agents on larger
  machines can add the optional `250_000` tier with `--include-extreme`.
  At minimum, test `small` and one larger tier.
- **Warm-up exclusion:** Exclude the first repetition from aggregate statistics if it
  includes JIT/warmup artifacts. Report whether warmup was excluded.
- **Outlier handling:** If excluding outliers, document the criterion (e.g., >3σ) and
  report both with-outliers and without-outliers statistics.
- **Statistical significance:** When comparing two configurations, report whether the
  difference is statistically significant (p < 0.05) using an appropriate test
  (e.g., t-test, Mann-Whitney U). Do not claim a win without significance.

## Weave Logging Contract

Default project:
`sasha-krigel-massachusetts-institute-of-technology/inference-hack`

Uploader command:

```bash
WANDB_API_KEY=... python -m eval.upload_weave_results
```

Current upload receipt:
`eval/artifacts/phase04_weave_upload_receipt.json`

Logged trace ops:

| op | purpose |
|---|---|
| `eval.phase04.upload_results_trace` | parent trace for a frozen matrix upload |
| `eval.phase04.h100_scenario` | one traced row per scenario/GPU-count |
| `eval.phase04.rag_reference` | one traced row per RAG corpus size |
| `eval.phase04.h100_rag_comparison` | one traced row per H100/RAG comparison |

Logged eval rows:

| row type | count | primary scores |
|---|---:|---|
| H100 scenario | 8 | req/s, p50, p95, GPU util mean, BF16 MFU |
| H100/RAG comparison | 32 | RAG latency over H100 p50, H100 QPS over single-process RAG QPS |

Required Weave attributes:

| attribute | meaning |
|---|---|
| `project_area` | `phase04-performance` |
| `result_type` | artifact class, currently `h100-rag-matrix` |
| `repo_commit` | commit used when uploading the result |
| `matrix_run_id` | frozen matrix run id |
| `model` | model id |
| `vllm_version` | vLLM version |
| `weave_schema_version` | schema marker, currently `phase04.results.v1` |

## Standard Optimization Entry Format

Copy this block for every optimization attempt.

```markdown
### OPT-XXX: Short Name

- status: proposed | running | applied | rejected | reverted
- owner:
- date:
- commit:
- artifacts:
- Weave run/eval:
- hypothesis:
- change:
- expected mechanism:

#### Experiment Configuration
- repetitions: (minimum 3)
- warmup excluded: yes | no
- dataset sizes tested:
  | size tier | doc count | notes |
  |---|---:|---|

#### Quality Gate
- precision: mean ± std (n=)
- recall: mean ± std (n=)
- F1: mean ± std (n=)
- threshold:
- verdict:

#### Workloads Tested
- static single-user:
- static multi-user:
- dynamic single-user:
- dynamic multi-user:

#### Performance Delta (with variance)
| workload | dataset | baseline mean ± std | candidate mean ± std | delta | p-value | verdict |
|---|---|---:|---:|---:|---:|---|

#### Utilization (with variance)
| workload | dataset | GPU util mean ± std | BF16 MFU mean ± std | power mean ± std W | memory max MB |
|---|---|---:|---:|---:|---:|

#### Scaling Analysis
| metric | small→medium | medium→large | large→xlarge | xlarge→xxlarge | scaling behavior |
|---|---:|---:|---:|---:|---|

#### RAG Comparison
- static metric:
- dynamic metric:
- largest advantage:

#### Summary
- caveats:
- regression threshold:
- decision:
- next action:
- rollback:
```

## Agent Experiment Summary

After completing any experiment, agents MUST generate a detailed summary using the
template in **[agent-experiment-summary-format.md](agent-experiment-summary-format.md)**.

### Standard Benchmark Runner

Use `eval.standard_benchmark` for every optimization unless there is a documented
reason it cannot represent the change. The runner creates the experiment folder,
copies matrix runs into `runs/`, computes current-vs-baseline deltas, runs the
standard RAG size ladder, writes aggregate/scaling JSON, and emits a ledger-entry
block that can be pasted into this file.

Fast artifact-only comparison, useful after an agent has already generated a
candidate matrix:

```bash
python -m eval.standard_benchmark \
  --opt-id OPT-XXX \
  --name "short optimization name" \
  --baseline-artifact eval/artifacts/phase04_h100_rag_matrix.json \
  --candidate-artifact eval/artifacts/experiment_results/OPT-XXX/candidate_h100_rag_matrix.json
```

Full Modal/GPU path, useful when the agent should generate the candidate matrix
itself:

```bash
SCORER_MIN_CONTAINERS=0 VLLM_METRICS_VERSION=0.22.1 ENABLE_MFU_METRICS=1 \
GPU_MEMORY_UTILIZATION=0.92 KV_CACHE_DTYPE=auto \
python -m eval.standard_benchmark \
  --opt-id OPT-XXX \
  --name "short optimization name" \
  --run-modal \
  --gpu-counts 1,6 \
  --single-requests 32 \
  --multi-requests 96 \
  --single-concurrency 1 \
  --multi-concurrency 32 \
  --dataset-sizes 7 100 1000 10000 25000 100000
```

Outputs:

| file | purpose |
|---|---|
| `eval/artifacts/experiment_results/OPT-XXX/config.json` | exact inputs and command |
| `eval/artifacts/experiment_results/OPT-XXX/runs/*.json` | per-matrix run records |
| `eval/artifacts/experiment_results/OPT-XXX/aggregated.json` | baseline/candidate stats and deltas |
| `eval/artifacts/experiment_results/OPT-XXX/scaling_analysis.json` | RAG dataset-size scaling |
| `eval/artifacts/experiment_results/OPT-XXX/ledger_entry.md` | paste-ready optimization ledger entry |
| `eval/artifacts/experiment_summaries/OPT-XXX_summary.md` | human-readable experiment summary |

To compare against multiple prior runs, pass `--baseline-artifact` more than once.
To compare repeated candidate runs, pass `--candidate-artifact` more than once.
The runner reports mean/std/min/max and approximate p-values when both sides have
at least two runs.

### Experiment Artifacts Folder Structure

```
eval/artifacts/
├── experiment_summaries/           # Human-readable summaries
│   ├── OPT-001_summary.md
│   ├── OPT-002_summary.md
│   └── ...
├── experiment_results/             # Detailed per-run data
│   ├── OPT-001/
│   │   ├── config.json             # Experiment configuration
│   │   ├── runs/
│   │   │   ├── run_001.json        # Individual run results
│   │   │   ├── run_002.json
│   │   │   └── ...
│   │   ├── aggregated.json         # Aggregated statistics
│   │   ├── scaling_analysis.json   # Dataset scaling data
│   │   └── plots/
│   │       ├── scaling_curve.png
│   │       ├── latency_distribution.png
│   │       └── ...
│   └── OPT-002/
│       └── ...
├── phase04_h100_rag_matrix.json    # Current baseline artifacts
├── phase04_quality_gate.json
└── ...
```

### Agent Instructions: Recording and Appending Results

**Step 1: Create experiment folder**
```bash
mkdir -p eval/artifacts/experiment_results/OPT-XXX/runs
mkdir -p eval/artifacts/experiment_results/OPT-XXX/plots
```

**Step 2: Save configuration**
Save `eval/artifacts/experiment_results/OPT-XXX/config.json`:
```json
{
  "opt_id": "OPT-XXX",
  "hypothesis": "...",
  "commit": "abc1234",
  "timestamp": "2026-06-20T12:00:00Z",
  "independent_variables": {...},
  "controlled_variables": {...},
  "dataset_sizes": ["small", "medium", "large"],
  "repetitions_per_config": 5,
  "warmup_runs": 1
}
```

**Step 3: Save individual run results**
For each run, save `eval/artifacts/experiment_results/OPT-XXX/runs/run_NNN.json`:
```json
{
  "run_id": "run_001",
  "timestamp": "2026-06-20T12:05:00Z",
  "dataset_size": "medium",
  "dataset_doc_count": 1000,
  "config_variant": "baseline",
  "repetition": 1,
  "is_warmup": false,
  "metrics": {
    "throughput_req_s": 301.5,
    "latency_p50_ms": 86.4,
    "latency_p95_ms": 160.6,
    "latency_p99_ms": 185.2,
    "gpu_util_mean": 32.0,
    "gpu_util_max": 64.0,
    "mfu_bf16": 0.082,
    "power_mean_w": 134.7,
    "power_max_w": 153.1,
    "memory_max_mb": 75309,
    "precision": 1.0,
    "recall": 0.889,
    "f1": 0.941
  },
  "errors": [],
  "notes": ""
}
```

**Step 4: Compute and save aggregated results**
After all runs complete, compute statistics and save `eval/artifacts/experiment_results/OPT-XXX/aggregated.json`:
```json
{
  "opt_id": "OPT-XXX",
  "generated": "2026-06-20T14:00:00Z",
  "total_runs": 15,
  "warmup_excluded": 3,
  "effective_runs": 12,
  "by_config": {
    "baseline_medium": {
      "n": 4,
      "throughput_req_s": {"mean": 301.2, "std": 5.3, "min": 294.1, "max": 308.7, "ci_95": [296.1, 306.3]},
      "latency_p50_ms": {"mean": 86.5, "std": 2.1, "min": 83.2, "max": 89.1, "ci_95": [84.4, 88.6]}
    }
  },
  "comparisons": {
    "candidate_vs_baseline_medium": {
      "metric": "throughput_req_s",
      "baseline_mean": 301.2,
      "candidate_mean": 352.8,
      "abs_delta": 51.6,
      "rel_delta_pct": 17.1,
      "test": "t-test",
      "t_statistic": 4.23,
      "p_value": 0.0032,
      "significant": true,
      "effect_size_cohens_d": 1.85
    }
  }
}
```

**Step 5: Generate summary**
Create `eval/artifacts/experiment_summaries/OPT-XXX_summary.md` using the template.

**Step 6: Append to ledger**
Add a new entry to the Optimization Log section of this file following the Standard
Optimization Entry Format. Include:
- Reference to artifacts: `eval/artifacts/experiment_results/OPT-XXX/`
- Key aggregated metrics with variance
- Verdict based on statistical significance

## Current Artifacts

| artifact | run id | note |
|---|---|---|
| `eval/artifacts/phase04_h100_rag_matrix.json` | `phase04-h100-rag-matrix-1781946340` | current 1-vs-6 H100 static/dynamic matrix |
| `eval/artifacts/phase04_h100_rag_matrix.md` | `phase04-h100-rag-matrix-1781946340` | human-readable matrix report |
| `eval/artifacts/phase04_quality_gate.json` | `quality-36f7ff08` | measured modal quality gate |
| `eval/artifacts/phase04_rag_vs_6xh100.json` | `rag-vs-6xh100-1781944574` | earlier 6-H100 vs RAG comparison |
| `eval/artifacts/phase04_weave_upload_receipt.json` | `phase04-h100-rag-matrix-1781946340` | Weave upload receipt |

## Quality Baseline

Source: `eval/artifacts/phase04_quality_gate.json`

| backend | model | artifact commit | threshold | precision | recall | F1 | verdict |
|---|---|---|---:|---:|---:|---:|---|
| modal | `Qwen/Qwen2.5-3B-Instruct-AWQ` | `3de70ca` | 0.016403 | 1.000000 | 0.888889 | 0.941176 | pass |

Quality caveat: this is the small pinned Phase 04 gate over the demo corpus and
scripted predicates. Future performance wins should be rechecked on a larger gold
set before being treated as general.

## Current H100 Matrix Baseline

Source: `eval/artifacts/phase04_h100_rag_matrix.json`

Run config:

| field | value |
|---|---|
| model | `Qwen/Qwen2.5-3B-Instruct-AWQ` |
| vLLM | `0.22.1` |
| prompt variant | `compact` |
| GPU memory utilization | `0.92` |
| max batched tokens | `8192` |
| GPU counts | `1, 6` |

H100 rows:

| scenario | H100s | req/s | p50 ms | p95 max ms | BF16 MFU | GPU util mean/max | power mean/max W | memory max MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_user_dynamic | 1 | 301.459 | 86.463 | 160.617 | 0.082889 | 32.0/64.0 | 134.7/153.1 | 75309.0 |
| multi_user_dynamic | 6 | 1762.192 | 92.042 | 212.250 | 0.080755 | 19.4/63.0 | 138.5/168.0 | 75309.0 |
| multi_user_static | 1 | 208.341 | 158.134 | 205.087 | 0.009498 | 10.0/20.0 | 122.6/128.4 | 75309.0 |
| multi_user_static | 6 | 1351.971 | 132.197 | 244.472 | 0.010272 | 11.4/52.0 | 123.4/134.8 | 75309.0 |
| single_user_dynamic | 1 | 72.296 | 13.542 | 15.570 | 0.019709 | 4.5/9.0 | 122.0/128.2 | 75309.0 |
| single_user_dynamic | 6 | 419.784 | 14.665 | 27.157 | 0.019073 | 15.6/38.0 | 129.0/145.1 | 75309.0 |
| single_user_static | 1 | 78.544 | 12.237 | 14.573 | 0.003581 | 15.5/31.0 | 119.7/124.1 | 75309.0 |
| single_user_static | 6 | 441.056 | 13.947 | 21.098 | 0.003351 | 12.8/34.0 | 123.4/131.9 | 75309.0 |

1-vs-6 scaling:

| scenario | 1 H100 req/s | 6 H100 req/s | throughput scale | 1 H100 p50 ms | 6 H100 p50 ms | p50 ratio |
|---|---:|---:|---:|---:|---:|---:|
| multi_user_dynamic | 301.459 | 1762.192 | 5.846x | 86.463 | 92.042 | 1.065x |
| multi_user_static | 208.341 | 1351.971 | 6.489x | 158.134 | 132.197 | 0.836x |
| single_user_dynamic | 72.296 | 419.784 | 5.806x | 13.542 | 14.665 | 1.083x |
| single_user_static | 78.544 | 441.056 | 5.615x | 12.237 | 13.947 | 1.140x |

Interpretation:

- 6-H100 throughput scales near linearly for the matrix workloads.
- Latency does not improve with 6 replicas in single-user lanes because the work
  is not parallelized within one request; 6 replicas mainly improve aggregate QPS.
- GPU utilization is still low relative to the refinement target. The best mean
  GPU utilization is `32.0%` on 1-H100 dynamic multi-user and `19.4%` on
  6-H100 dynamic multi-user.
- Static lanes confirm prefix/shared-prompt behavior is being measured on both
  1-H100 and 6-H100 paths, but static MFU is very low because repeated compact
  prompts make the measured request window too small to saturate the GPUs.

## RAG Baseline

Source: `eval/artifacts/phase04_h100_rag_matrix.json`

| docs | static retrieve p50 ms | dynamic fresh-file total ms | retrieve QPS |
|---:|---:|---:|---:|
| 7 | 0.188 | 0.650 | 5327.423 |
| 1000 | 6.674 | 40.297 | 149.824 |
| 10000 | 55.398 | 223.745 | 18.051 |
| 25000 | 138.775 | 503.477 | 7.206 |

Largest current latency separation:

| workload | H100s | RAG docs | RAG metric | H100 p50 ms | RAG ms | ratio |
|---|---:|---:|---|---:|---:|---:|
| single_user_dynamic | 1 | 25000 | fresh-file total | 13.542 | 503.477 | 37.178x |

RAG caveat: this baseline uses the repo's `numpy-fallback` hashing-vectorizer path.
It is useful for structural latency and recall comparison, not for claiming victory
over a tuned FAISS/neural embedding production RAG stack.

## Optimization Log

### OPT-001: Compact Prompt + vLLM Metrics + GPU Sampling Matrix

- status: applied
- owner: A / eval-inference
- date: 2026-06-20
- commit: `77c7280`
- artifacts:
  - `eval/artifacts/phase04_h100_rag_matrix.json`
  - `eval/artifacts/phase04_h100_rag_matrix.md`
  - `eval/artifacts/phase04_weave_upload_receipt.json`
- Weave run/eval: uploaded via `python -m eval.upload_weave_results`
- hypothesis: compact max-tokens-1 classifier prompts plus vLLM metrics expose
  prefill throughput, queue state, MFU, and utilization more directly than the
  earlier Modal run.
- change:
  - Added static/dynamic prompt modes.
  - Enabled matrix runs for 1-H100 and 6-H100 scenarios.
  - Captured vLLM queue/KV/MFU metrics and `nvidia-smi` GPU utilization, memory,
    and power samples.
  - Added RAG static retrieve and dynamic fresh-file comparison rows.
- expected mechanism: isolate prefill-heavy scorer behavior and show where data
  parallelism helps throughput versus where single-request latency remains unchanged.
- quality gate:
  - precision: 1.000000
  - recall: 0.888889
  - F1: 0.941176
  - threshold: 0.016403
  - verdict: pass on the small Phase 04 gate
- performance delta:
  | workload | baseline | candidate | delta | verdict |
  |---|---:|---:|---:|---|
  | single_user_static 1-to-6 | 78.544 req/s | 441.056 req/s | 5.615x | pass |
  | multi_user_static 1-to-6 | 208.341 req/s | 1351.971 req/s | 6.489x | pass |
  | single_user_dynamic 1-to-6 | 72.296 req/s | 419.784 req/s | 5.806x | pass |
  | multi_user_dynamic 1-to-6 | 301.459 req/s | 1762.192 req/s | 5.846x | pass |
- utilization:
  | workload | GPU util mean/max | BF16 MFU | power mean/max W | memory max MB |
  |---|---:|---:|---:|---:|
  | multi_user_dynamic, 6 H100 | 19.4/63.0 | 0.080755 | 138.5/168.0 | 75309.0 |
  | multi_user_static, 6 H100 | 11.4/52.0 | 0.010272 | 123.4/134.8 | 75309.0 |
- caveats:
  - Startup, CUDA graph capture, and Triton JIT are not fully separated from the
    end-to-end Modal experience.
  - Short request windows undersample sustained GPU utilization.
  - 6 replicas improve aggregate throughput; they do not make one single-user
    request six times faster.
- regression threshold:
  - Same workload req/s must not drop more than 5% unless quality or latency improves.
  - Same workload p95 must not regress more than 10% without an explicit tradeoff.
  - Quality F1 must stay above 0.7 and should not drop from 0.941176 without review.
- decision: keep as the current benchmark baseline.
- next action: run longer sustained multi-user loads and warm exact measured shapes
  before timing to separate steady-state GPU utilization from startup/JIT effects.
- rollback: disable only the new matrix/uploader path; production scoring path is unchanged.

### OPT-002: Threshold Calibration

- status: applied
- artifacts:
  - `eval/artifacts/phase04_quality_gate.json`
  - `eval/artifacts/phase04_rag_vs_6xh100.json`
- hypothesis: the real scorer separates positives but the default 0.5 threshold
  discards recall.
- change: use the threshold sweep recommendation for the demo operating point.
- measured result: threshold `0.016403`, precision `1.000000`, recall `0.888889`,
  F1 `0.941176`.
- caveat: validate on a larger gold set before treating this threshold as universal.
- decision: keep for Phase 04 demo/eval reporting.

## Bottlenecks To Target Next

| bottleneck | evidence | next optimization |
|---|---|---|
| Low sustained utilization | 6-H100 dynamic multi-user mean GPU util is `19.4%`; static lanes lower | longer request windows, higher concurrency, bigger batches |
| First-use warmup/JIT contamination | vLLM logs showed CUDA graph capture and Triton JIT around startup | prewarm exact prompt shapes and exclude warmup from measured interval |
| Single-request latency not improved by replicas | 6-H100 p50 is similar or higher than 1-H100 in single-user lanes | do not sell data parallelism as single-query latency improvement |
| Prefix reuse not persisted at document level | static prompt lane exercises vLLM prefix caching, not a document prefix tree | implement persistent document-prefix KV/prefix-tree reuse later |
| Scheduling imbalance untested | current 6-H100 path is data-parallel replicas, not token-length bin packing | add token-count bin packing and straggler ratio to matrix |

## Future Optimization Queue

| id | candidate | measurement required before accepting |
|---|---|---|
| OPT-003 | longer sustained load and warm measured shapes | same matrix rows with request window long enough for stable utilization |
| OPT-004 | concurrency/max batched token sweep | req/s, p50, p95, GPU util, MFU, queue depth at each setting |
| OPT-005 | `--gpu-memory-utilization` sweep | KV capacity, memory used, OOM rate, latency, throughput |
| OPT-006 | token-length bin packing across replicas | straggler ratio, p95 latency, throughput |
| OPT-007 | persistent prefix-tree/document KV reuse | prefix hit rate, prefill tokens avoided, TTFT, memory tradeoff |
| OPT-008 | quantization/attention backend sweep | quality gate plus steady-state prefill tok/s and MFU |
