# Phase 04 GPU Cluster Agent Prompt

You are an autonomous coding agent running on the GPU cluster over SSH. Your job is to implement and validate Phase 04: the real vLLM scorer swap, score-quality gate, measured performance freeze, and real-run replay fixtures. Work in the existing repo. Do not invent a parallel architecture.

## Mission

The project is "grep for meaning": replace RAG-style index-first retrieval with direct semantic filtering over raw chunks when inference is cheap enough. The main proof is not a dashboard timer. The proof is:

1. score quality is acceptable before any speed claim,
2. repeated interaction uses fewer model-scored chunks than full re-score / RAG re-retrieve,
3. threshold drag and chip deletion are zero inference,
4. fresh-file ingestion is queryable immediately without derived index writes,
5. real vLLM measurements are frozen with commit, model, corpus, backend, date, and caveats.

Keep Phase 04 focused. Do not build Tier-2 cascade, second domain, editable chip algebra, or unrelated UI polish unless the real scorer path and eval freeze are already green.

## Repository State To Expect

Start from `main` after Phase 03:

```bash
git fetch --prune origin
git switch main
git pull --ff-only
git log --oneline --max-count=5
```

Phase 03 should already include:

- `eval/cut_line.py`
- `scripts/replay_sse.py`
- `scripts/preload_demo.sh`
- `eval/artifacts/cut_line_trace.json`
- `eval/artifacts/area_under_loop.png`
- `eval/weave_ops.py`
- `eval/requirements.txt`
- `backend/main.py` with `/ingest`, `/query`, `/refine`, `/results`, and `DELETE /clause/{id}`
- `inference/config.py` where `SCORER_BACKEND=vllm` currently raises `NotImplementedError`

Run the mock cut-line before touching vLLM:

```bash
PYTHON=.venv/bin/python bash scripts/preload_demo.sh
.venv/bin/python -m unittest discover -s tests
```

If these fail, stop and fix the mock path first.

## Known Phase 3 Carry-Forward Risks

Address these before freezing real numbers:

1. `scripts/replay_sse.py` currently records only one query fixture and one require-refine fixture. It does not record the full demo sequence: query -> click-NOT -> AND refine -> threshold recut -> fresh-file query. For Phase 04, record real vLLM fixtures for the actual demo beats, or make the limitation explicit in the fallback ladder. The replay test must prove the fixtures that matter for the demo.
2. `eval/cut_line.py` writes `eval/artifacts/cut_line_trace.json` with a toy RAG re-index timing. That timing is non-deterministic and dirties the worktree when preflight is run. For Phase 04, separate "stable committed mock fixture" from "fresh measured output", or round/freeze the field intentionally. Preflight should not leave a dirty tree unless it is explicitly producing new freeze artifacts.

Do not ignore these. They are demo reliability issues.

## Hard Constraints

- Preserve the frozen scorer interface in `inference/scorer.py`.
- Preserve the backend API shape. Frontend and eval must keep using `make_scorer()`.
- `SCORER_BACKEND=mock` must continue to pass all existing tests.
- `SCORER_BACKEND=vllm` must be additive and isolated behind `inference/config.py`.
- Every trace row must follow `CONTRACTS.md` section 6.
- Every figure/number must be labeled `mock/projected`, `predicted`, or `measured`.
- Score quality gate comes before speed sweeps. Do not optimize the speed of being wrong.
- Do not put unverified constants or guessed speedups in the final artifacts.

## Cluster Setup

Discover the environment first:

```bash
nvidia-smi
python --version
which python
which uv || true
which pip
git status --short --branch
```

Create or reuse a Python 3.11/3.12 venv if possible. If the cluster only exposes another version, document it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r backend/requirements.txt
python -m pip install -r performance/requirements.txt
python -m pip install -r eval/requirements.txt
```

Install GPU dependencies in a new file if one does not exist:

```bash
# create inference/requirements-gpu.txt if missing
vllm
httpx
numpy
```

Then:

```bash
python -m pip install -r inference/requirements-gpu.txt
python -m pip check
```

If vLLM install requires a specific CUDA/PyTorch wheel, use the cluster's documented module/conda path and write the exact commands into `eval/artifacts/phase04_environment.md`.

## vLLM Serving Target

Target model: start with `Qwen/Qwen2.5-3B-Instruct-AWQ` unless the score gate fails or the cluster lacks support. If it fails quality, record that decision and try the next fallback model, for example `meta-llama/Llama-3.1-8B-Instruct` or another available AWQ/GPTQ instruction model with reliable Yes/No logprobs.

Serve six single-GPU replicas if the cluster has enough GPUs. Keep two GPUs free for experiments or fallback if available. Prefer independent replicas over tensor parallel for the Tier-1 scorer.

Create `inference/serve.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8001}"
N_REPLICAS="${N_REPLICAS:-6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

pids=()
urls=()
for i in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((BASE_PORT + i))
  urls+=("http://${HOST}:${port}/v1")
  CUDA_VISIBLE_DEVICES="$i" python -m vllm.entrypoints.openai.api_server \
    --host "$HOST" \
    --port "$port" \
    --model "$MODEL" \
    --served-model-name tier1-filter \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-mfu-metrics \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    $EXTRA_VLLM_ARGS &
  pids+=("$!")
done

printf 'export VLLM_REPLICAS=%s\n' "$(IFS=,; echo "${urls[*]}")" | tee .vllm_replicas.env
trap 'kill "${pids[@]}"' INT TERM EXIT
wait
```

Adjust flags to match the installed vLLM version. If AWQ quantization needs `--quantization awq_marlin`, add it after testing. Do not claim FP8/AWQ effects unless you measure them.

`GPU_MEMORY_UTILIZATION` is intentionally parameterized. vLLM versions differ in documented/default behavior; many examples use `0.90`, while current docs may report another default. Always record the installed vLLM version and the actual value used in `eval/artifacts/phase04_environment.md`.

Health check:

```bash
source .vllm_replicas.env
for url in ${VLLM_REPLICAS//,/ }; do
  curl -s "$url/models" | head
done
```

Metrics check:

```bash
for url in ${VLLM_REPLICAS//,/ }; do
  metrics_url="${url%/v1}/metrics"
  curl -s "$metrics_url" | grep -E 'vllm:(kv_cache_usage_perc|num_requests_running|num_requests_waiting|.*mfu)' | head -20
done
```

Required vLLM metrics to scrape during Phase 04:

- `vllm:kv_cache_usage_perc`: percentage/fraction of the pre-allocated GPU KV cache currently in active use. Map this into the repo trace field `gpu_cache_usage_perc` for contract compatibility.
- `vllm:num_requests_running`: requests actively executing on a replica.
- `vllm:num_requests_waiting`: queued requests waiting for scheduling. Use `running` + `waiting` to interpret p50/p95 latency under saturation.
- MFU metrics exposed by `--enable-mfu-metrics`: raw GPU compute efficiency. Record the metric names exactly as emitted by the installed vLLM version.

Run a small GPU-memory utilization sweep after the first green serve:

```bash
for util in 0.80 0.85 0.90 0.92 0.95; do
  GPU_MEMORY_UTILIZATION="$util" N_REPLICAS=6 bash inference/serve.sh
  # In another shell: run the quality smoke and one warm refine benchmark, then stop the server.
  # Record: util, max model len, max num seqs, OOM/no-OOM, kv cache usage, running/waiting queues, p50/p95.
done
```

If higher utilization causes OOM, degraded queueing, or unstable p95, choose the highest stable value and document the rejected values. Do not treat larger KV allocation as automatically faster; it is a capacity/headroom experiment.

## Implement `VLLMScorer`

Add `inference/vllm_scorer.py` implementing `ScorerClient`.

Required behavior:

- `warm(corpus_id, chunks)` stores/writes enough local prefix state to make health and trace metadata meaningful. If vLLM prefix caching is implicit, warm by issuing prefill-like score requests over the corpus with a fixed neutral predicate and record warm duration. If true prefill-only is not exposed, document that limitation.
- `score_batch(items, tier=1)` sends OpenAI-compatible completions/chat-completions to vLLM and returns `ScoreResult` in the same order as input.
- Use `max_tokens=1`, `temperature=0`, and request logprobs if supported.
- Score with normalized Yes/No probability:
  `score = p_yes / (p_yes + p_no)`
- Aggregate token variants robustly: `Yes`, ` yes`, `YES`, `No`, ` no`, `NO`.
- If guided choice is supported and improves reliability, use it. If it hurts p50 or is unavailable, drop it and rely on logprob aggregation.
- Round-robin requests across `VLLM_REPLICAS`.
- Support timeouts and clear errors.
- `health()` returns `ready`, `backend`, `model_id`, `replicas`, and any useful prefix/cache fields.

Then update `inference/config.py`:

```python
if backend == "vllm":
    from inference.vllm_scorer import VLLMScorer
    return VLLMScorer.from_env()
```

Add focused tests with mocked HTTP responses. Tests must not require a GPU.

Minimum tests:

- `make_scorer()` returns `VLLMScorer` when `SCORER_BACKEND=vllm`.
- Yes/No logprobs normalize correctly.
- response order matches request order.
- missing Yes/No logprobs gives a clear exception or fallback behavior.
- multiple replicas are round-robined.

## Prompt Format For Scoring

Use one short, stable prompt. Keep chunk prefix and predicate suffix obvious.

Example:

```text
You are a strict semantic filter. Answer only Yes or No.

Chunk:
{chunk_text}

Question:
Does this chunk satisfy the predicate: {predicate}

Answer:
```

The model must produce a single-token Yes/No answer. Do not ask for rationale during scoring.

## Quality Gate

Before speed sweeps, implement or extend `eval.bench` so this works:

```bash
SCORER_BACKEND=vllm VLLM_REPLICAS="$VLLM_REPLICAS" python -m eval.bench --backend vllm --gate-only
```

If the existing CLI does not support those flags yet, add them. Keep `python -m eval.bench --smoke` working.

The gate should use the best available gold labels in the repo. If gold labels are incomplete, create a small explicit Phase 04 gold set for the pinned demo and a few non-demo predicates, then label it honestly as "small gate". Minimum output:

- precision
- recall
- F1
- threshold used
- model id
- corpus id / corpus size
- commit
- scorer backend

Hard stop default: F1 < 0.7. If you override, require `--force` and write the reason to artifacts.

Write:

- `eval/artifacts/phase04_quality_gate.json`
- `eval/artifacts/phase04_quality_gate.md`

## Performance Freeze

After the quality gate passes, measure:

1. cold full query over pinned corpus,
2. warm same query,
3. click-NOT refine,
4. AND refine over current survivors,
5. threshold drag / `/results` cache recut,
6. chip delete,
7. fresh-file ingest and immediate query,
8. RAG baseline index build / retrieve / re-index comparison,
9. vLLM `/metrics` GPU/cache/queue stats, including KV-cache usage, running/waiting request counts, and MFU when exposed.

Required trace fields follow `CONTRACTS.md`:

- `run_id`
- `commit`
- `corpus_id`
- `model_id`
- `scorer_backend`
- `turn`
- `operation`
- `threshold`
- `n_chunks_total`
- `candidate_count`
- `chunks_scored`
- `chunks_served_from_cache`
- `survivor_count`
- `rho`
- `elapsed_ms`
- `model_ms`
- `queue_ms`
- `ttft_ms`
- `cache_hit_rate`
- `gpu_cache_usage_perc`
- `warm_state`
- `latency_kind`
- `quality_slice`

Supplemental Phase 04 metrics may live in `phase04_metrics.json` / `phase04_gpu_memory_sweep.json`
instead of every trace row:

- `vllm_kv_cache_usage_perc`
- `vllm_num_requests_running`
- `vllm_num_requests_waiting`
- `mfu`
- `gpu_memory_utilization`

Write:

- `eval/artifacts/phase04_vllm_trace.jsonl`
- `eval/artifacts/phase04_metrics.json`
- `eval/artifacts/phase04_environment.md`
- `eval/artifacts/phase04_gpu_memory_sweep.json`
- regenerated measured figures under `eval/artifacts/` or `performance/figures/`

The area-under-loop chart must compare:

- measured scoped loop,
- full re-score counterfactual,
- suffix-only / warm counterfactual if available,
- RAG re-retrieve and, separately, RAG re-index on data change.

Mock figures must remain labeled projected. Real figures must include model, commit, corpus size, date, and backend.

## Weave / W&B Logging

The project is:

```python
weave.init("sasha-krigel-massachusetts-institute-of-technology/inference-hack")
```

On the cluster:

```bash
python -m pip install -r eval/requirements.txt
wandb login
python -m eval.bench --smoke --weave
```

Then make the Phase 04 gate/freeze commands log to Weave as well. If auth is missing, the code should fail non-interactively with a clear message. Do not silently skip logging for the freeze run.

## Real Replay Fixtures

Record real vLLM replay fixtures after the scorer is green. The fixtures should support the actual demo path, not only a single generic refine.

At minimum record:

- initial query SSE,
- click-NOT refine SSE,
- AND refine SSE after click-NOT,
- fresh-file query SSE after ingest.

If the current replay server cannot route multiple refine fixtures by request type or sequence, extend it. Add tests proving the replay order and operation semantics match the demo beats.

Write fixtures under `eval/artifacts/` with names that include `phase04_vllm` or metadata in a sidecar:

- `eval/artifacts/phase04_vllm_query.sse`
- `eval/artifacts/phase04_vllm_click_not.sse`
- `eval/artifacts/phase04_vllm_and_refine.sse`
- `eval/artifacts/phase04_vllm_fresh_query.sse`
- `eval/artifacts/phase04_vllm_fixture_meta.json`

Metadata must include commit, model, corpus id, corpus size, timestamp, command, and scorer backend.

## Backend Live Run

Run backend with real scorer:

```bash
source .venv/bin/activate
source .vllm_replicas.env
SCORER_BACKEND=vllm VLLM_REPLICAS="$VLLM_REPLICAS" uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Smoke:

```bash
curl -s http://127.0.0.1:8000/healthz | jq .
curl -N -X POST http://127.0.0.1:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"corpus_id":"demo"}'
curl -N -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"predicate":"every place we retry a network call without backoff","threshold":0.5}'
```

If a frontend is needed on the cluster, prefer serving a prebuilt `frontend/dist`. Do not burn time fighting Node if Node < 20.19. Build on a Node-20 machine and copy the dist if needed.

## Verification Commands

Run before committing:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall eval backend inference data performance baseline
.venv/bin/python -m pip check
.venv/bin/python -m eval.bench --smoke
PYTHON=.venv/bin/python bash scripts/preload_demo.sh
SCORER_BACKEND=vllm VLLM_REPLICAS="$VLLM_REPLICAS" .venv/bin/python -m eval.bench --backend vllm --gate-only
SCORER_BACKEND=vllm VLLM_REPLICAS="$VLLM_REPLICAS" .venv/bin/python -m eval.bench --backend vllm --tag freeze
git diff --check
git status --short --branch
```

If frontend files changed:

```bash
cd frontend
npm test -- --run
npm run build
```

## Commit / Handoff

Commit only after the checks above pass or after documenting a blocker clearly.

Suggested commits:

1. `inference: add vllm scorer`
2. `eval: add phase 4 quality gate`
3. `eval: freeze real vllm metrics`
4. `demo: record real vllm replay fixtures`

Final response must include:

- branch and commit hash,
- model id,
- number of replicas,
- exact `VLLM_REPLICAS`,
- whether score gate passed and F1,
- measured warm refine p50/p95,
- scoped vs full cumulative chunks,
- real fixture paths,
- Weave project/run link if available,
- explicit caveats and anything still mock/projected.

Do not claim Phase 04 is complete unless the real vLLM backend has run, the quality gate has passed or a fallback decision is documented, measured artifacts exist, and verification commands have completed.
