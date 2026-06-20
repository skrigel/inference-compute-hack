# Evaluation Plan

> **Purpose:** Framework to evaluate optimization effectiveness during development and demonstrate MVP quality before judging.

## 1. Goals

1. **Development:** Measure each optimization's impact (scoping, caching, warm KV) independently
2. **Validation:** Confirm quality before optimizing speed ("don't optimize the speed of being wrong")
3. **Demonstration:** Produce slides showing compute utilization, iteration efficiency, and RAG comparison
4. **Tracking:** Record metrics throughout the hack via Weave for reproducibility

---

## 2. Metric Hierarchy

From `METRICS.md` — ordered by what matters most:

| Layer | Primary Metric | Why It Matters |
|-------|----------------|----------------|
| **Work** | `chunks_scored`, `chunks_served_from_cache` | Deterministic compute count — the x-axis |
| **Compute** | FLOPs, achieved TFLOP/s, MFU | Shows H100 utilization |
| **Latency** | p50/p95 query and refine time | The felt interactive experience |
| **Cache** | prefix hit rate, warm-KV footprint | Explains cold/warm/scoped behavior |
| **Quality** | F1, precision/recall, AUC, ECE | Prevents optimizing speed while wrong |
| **RAG Exchange** | index build, query cost, break-even churn | Quantifies recompute-over-store |

### 2.1 Required Counters Per Trace

Every query/refine call logs:

```python
@dataclass
class TurnTrace:
    # Identity
    run_id: str
    commit: str
    corpus_id: str
    model_id: str
    scorer_backend: str  # "mock" | "vllm"

    # Turn info
    turn: int
    operation: str  # "query" | "require" | "exclude" | "include" | "refocus" | "brush"
    threshold: float

    # Work counters (PRIMARY)
    n_chunks_total: int
    candidate_count: int
    chunks_scored: int              # cache miss → model call
    chunks_served_from_cache: int   # cache hit → free
    survivor_count: int
    rho: float                      # survivor_count / candidate_count

    # Timing
    elapsed_ms: float
    model_ms: float                 # from CUDA events
    queue_ms: float                 # vLLM queue time

    # vLLM metrics (scraped from /metrics)
    ttft_ms: float                  # time-to-first-token
    cache_hit_rate: float
    gpu_cache_usage_perc: float

    # Quality (when gold labels exist)
    quality_slice: Optional[QualityMetrics]
```

---

## 3. Optimization Ladder Evaluation

### 3.1 The Ladder (from MVP §11)

Each rung is a config flag. Sweep cumulatively in build-effort order.

**Ladder B — Refinement Latency (the headline):**

| Rung | Config | What It Adds |
|------|--------|--------------|
| B0 | `baseline` | Full corpus re-score every turn, cold |
| B1 | `+warm_kv` | Warm-on-ingest + suffix-only re-prefill |
| B2 | `+scoped` | Candidate-set scoping (`require`/`exclude` over survivors) |
| B3 | `+score_cache` | Persistent (chunk_id, clause_id) → score cache |

**Ladder A — Cold-Pass Throughput:**

| Rung | Config | What It Adds |
|------|--------|--------------|
| A0 | `baseline` | Single request, no batching |
| A1 | `+batching` | Continuous batching |
| A2 | `+replicas` | Data-parallel replicas (×8) |
| A3 | `A3_fp8_compute` | FP8 tensor-core prefill compute |
| C1 | `C1_4bit_weights_kv` | 4-bit weights/KV for memory capacity, not raw scan throughput |

### 3.2 Ablation Protocol

```python
REGIMES = {
    "B0_baseline": ComputeRegime.baseline(),
    "B1_warm":     ComputeRegime.warm(),
    "B2_scoped":   ComputeRegime.scoped(),
    "B3_cached":   ComputeRegime.cached(),
}

def run_ablation(session: ScriptedSession) -> Dict[str, SessionMetrics]:
    results = {}
    for name, regime in REGIMES.items():
        configure(regime)
        results[name] = run_session(session)
    return results
```

### 3.3 Metrics Per Rung

| Metric | B0 | B1 | B2 | B3 | Target |
|--------|----|----|----|----|--------|
| Refine latency p50 | seconds | ~500ms | ~200ms | ~150ms | <300ms |
| Cumulative work (6 turns) | 6N | 6N | ~2.2N | ~2.2N | saturates |
| Work saved vs B0 | 0% | 0% | ~65% | ~65%+ | >50% |

**Capture B0 (naive floor) early** — it's perishable once optimizations are on.

---

## 4. Scripted Sessions (6 sessions)

### 4.1 Session Definitions

| # | Name | Domain | Turns | Tests |
|---|------|--------|-------|-------|
| 1 | Narrow and Recover | papers | 5 | Refocus recovery after over-narrowing |
| 2 | Progressive Exclusion | code | 5 | NOT compounding, precision improvement |
| 3 | Semantic Pivot | mixed | 4 | OR expansion, then narrowing |
| 4 | Threshold Tuning | papers | 4 | Zero-inference threshold drags |
| 5 | Cross-Format Search | mixed | 5 | Papers + code on same concept |
| 6 | Ambiguity Resolution | code | 4 | Clarifying word sense mid-session |

### 4.2 Session Details

**Session 1: Narrow and Recover**
```yaml
name: narrow_and_recover
domain: papers
query: "papers about attention mechanisms"
turns:
  - {op: "require", text: "self-attention specifically", expected_rho: [0.4, 0.6]}
  - {op: "exclude", text: "survey or review papers", expected_rho: [0.7, 0.9]}
  - {op: "require", text: "with ablation studies", expected_rho: [0.3, 0.5]}
  - {op: "refocus", text: "attention in vision models, not NLP", expected_rho: "recovery"}
  - {op: "require", text: "convolutional attention", expected_rho: [0.4, 0.6]}
```

**Session 2: Progressive Exclusion**
```yaml
name: progressive_exclusion
domain: code
query: "error handling code"
turns:
  - {op: "exclude", text: "test files", expected_rho: [0.7, 0.85]}
  - {op: "exclude", text: "logging statements", expected_rho: [0.8, 0.9]}
  - {op: "exclude", text: "retry logic", expected_rho: [0.8, 0.9]}
  - {op: "require", text: "database operations", expected_rho: [0.2, 0.4]}
  - {op: "exclude", text: "ORM-specific", expected_rho: [0.6, 0.8]}
```

**Session 3: Semantic Pivot**
```yaml
name: semantic_pivot
domain: mixed
query: "authentication implementations"
turns:
  - {op: "require", text: "token-based", expected_rho: [0.3, 0.5]}
  - {op: "include", text: "session-based auth", expected_rho: "expands"}
  - {op: "exclude", text: "deprecated methods", expected_rho: [0.85, 0.95]}
  - {op: "require", text: "with refresh mechanism", expected_rho: [0.2, 0.4]}
```

**Session 4: Threshold Tuning**
```yaml
name: threshold_tuning
domain: papers
query: "machine learning optimization"
turns:
  - {op: "brush", value: 0.6, expected_chunks_scored: 0}
  - {op: "brush", value: 0.4, expected_chunks_scored: 0}
  - {op: "require", text: "gradient descent variants", expected_rho: [0.3, 0.5]}
  - {op: "brush", value: 0.7, expected_chunks_scored: 0}
```

**Session 5: Cross-Format Search**
```yaml
name: cross_format
domain: mixed
query: "retry without backoff"
turns:
  - {op: "exclude", text: "unit tests or mocks", expected_rho: [0.6, 0.8]}
  - {op: "require", text: "network or HTTP", expected_rho: [0.3, 0.5]}
  - {op: "require", text: "handles timeout", expected_rho: [0.4, 0.6]}
  - {op: "include", text: "papers discussing retry patterns", expected_rho: "expands"}
  - {op: "exclude", text: "theoretical only", expected_rho: [0.7, 0.85]}
```

**Session 6: Ambiguity Resolution**
```yaml
name: ambiguity_resolution
domain: code
query: "model training code"
turns:
  - {op: "require", text: "neural network", expected_rho: [0.5, 0.7]}
  - {op: "refocus", text: "ML model, not data model", expected_rho: "semantic_shift"}
  - {op: "exclude", text: "inference only", expected_rho: [0.7, 0.85]}
  - {op: "require", text: "with checkpointing", expected_rho: [0.3, 0.5]}
```

### 4.3 Session Metrics

```python
@dataclass
class SessionMetrics:
    session_name: str
    regime: str

    # Per-turn traces
    turns: List[TurnTrace]

    # Aggregate work
    total_chunks_scored: int
    total_chunks_from_cache: int

    # Timing
    total_elapsed_ms: float
    refine_latency_p50: float
    refine_latency_p95: float

    # Scoping effectiveness
    rho_per_turn: List[float]
    cumulative_work_curve: List[int]
    theoretical_asymptote: float        # N / (1 - mean_rho)
    scoping_efficiency: float           # 1 - (actual / full_rescore)

    # Regression checks
    rho_in_expected_range: List[bool]
```

---

## 5. Task Completion Queries (10 tasks)

### 5.1 Task Definitions

| # | Name | Domain | Targets | Distractors | Max Turns | Success |
|---|------|--------|---------|-------------|-----------|---------|
| 1 | Transformer Architecture | papers | 5 | 50 | 6 | 0.8 |
| 2 | Retry Bug Patterns | code | 3 | 30 | 8 | 1.0 |
| 3 | Auth Token Expiry | code | 2 | 40 | 6 | 1.0 |
| 4 | Efficient Attention | papers | 5 | 100 | 10 | 0.6 |
| 5 | Memory Leak Patterns | code | 4 | 35 | 8 | 0.75 |
| 6 | Reinforcement Learning | papers | 6 | 80 | 8 | 0.67 |
| 7 | SQL Injection Risk | code | 3 | 45 | 6 | 1.0 |
| 8 | Diffusion Models | papers | 4 | 60 | 8 | 0.75 |
| 9 | Race Conditions | code | 3 | 40 | 8 | 1.0 |
| 10 | Few-Shot Learning | papers | 5 | 70 | 8 | 0.6 |

### 5.2 Task Details

```python
@dataclass
class CompletionTask:
    name: str
    description: str
    domain: str
    initial_query: str
    targets: List[str]          # chunk IDs that MUST be found
    target_count: int
    distractors: int            # similar but wrong chunks
    max_turns: int
    success_threshold: float    # fraction of targets needed
```

**Task 1: Transformer Architecture**
```yaml
name: transformer_architecture
description: "Find papers introducing/extending transformer architecture"
initial_query: "transformer neural network architecture"
targets:
  - arxiv:1706.03762  # Attention Is All You Need
  - arxiv:1810.04805  # BERT
  - arxiv:2005.14165  # GPT-3
  - arxiv:2010.11929  # ViT
  - arxiv:1901.02860  # Transformer-XL
success_threshold: 0.8
max_turns: 6
```

**Task 2: Retry Bug Patterns**
```yaml
name: retry_bugs
description: "Find retry code without proper backoff (potential bugs)"
initial_query: "retry logic network operations"
targets:
  - file:services/api_client.py:retry_request
  - file:utils/http.py:fetch_with_retry
  - file:workers/sync.py:retry_failed_jobs
success_threshold: 1.0
max_turns: 8
```

**Task 3-10:** (See `eval/tasks/definitions.py` for full specs)

### 5.3 Task Completion Metrics

```python
@dataclass
class TaskCompletionMetrics:
    task_name: str

    # Completion
    targets_found: int
    targets_total: int
    recall: float
    success: bool               # recall >= threshold

    # Effort
    turns_used: int
    turns_to_first_target: int
    turns_to_80pct: Optional[int]

    # Precision
    false_positives_shown: int
    precision_at_completion: float

    # Work
    total_chunks_scored: int
    efficiency: float           # targets_found / chunks_scored

    # Trace
    recall_per_turn: List[float]
```

---

## 6. RAG Baseline Comparison

### 6.1 RAG Implementation

```
baseline/
├── rag.py           # main pipeline: embed → index → query → rerank
├── embedder.py      # sentence-transformers wrapper
├── index.py         # FAISS IVF-PQ index
├── reranker.py      # cross-encoder rerank (optional)
└── metrics.py       # RAG-specific counters
```

**Stack:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS IVF-PQ
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional)

### 6.2 RAG Metrics

```python
@dataclass
class RAGMetrics:
    # One-time costs
    index_build_time_ms: float
    index_build_energy_joules: Optional[float]
    index_size_bytes: int
    embedding_size_bytes: int
    total_storage_bytes: int    # raw + embeddings + index

    # Per-query costs
    embed_time_ms: float
    search_time_ms: float
    rerank_time_ms: float
    total_query_time_ms: float

    # Quality
    recall_at_k: float
    precision_at_k: float
    f1_at_k: float
```

### 6.3 Head-to-Head Comparison

#### 6.3.1 On Scripted Sessions

```python
@dataclass
class SessionComparison:
    session_name: str

    # Per-turn latency
    ours_latency_per_turn: List[float]
    rag_latency_per_turn: List[float]

    # Cumulative work
    ours_cumulative_chunks: List[int]
    rag_cumulative_queries: int         # = num_turns (independent)

    # Key insight: RAG has no semantic continuity
    ours_semantic_continuity: bool      # True
    rag_semantic_continuity: bool       # False

    # Quality at end
    ours_final_f1: float
    rag_final_f1: float
```

#### 6.3.2 On Task Completion

```python
@dataclass
class TaskComparison:
    task_name: str

    # Completion
    ours_turns_to_completion: int
    rag_turns_to_completion: int
    ours_success: bool
    rag_success: bool

    # Quality
    ours_final_recall: float
    rag_final_recall: float

    # Cost
    ours_total_latency_ms: float
    rag_total_latency_ms: float
```

#### 6.3.3 Fresh File Injection

```python
@dataclass
class FreshFileComparison:
    # Ours: query immediately
    ours_latency_ms: float

    # RAG: must re-index first
    rag_index_update_ms: float
    rag_query_ms: float
    rag_total_ms: float

    # The win
    speedup: float              # rag_total / ours_latency
```

### 6.4 Area-Under-Loop Comparison

The money shot figure. Four lines:

| Line | Formula | Source |
|------|---------|--------|
| Ours (scoped) | measured | Direct measurement |
| Ours (full re-score) | k × N | Derived (counterfactual) |
| RAG (re-retrieve) | k × query_cost | Direct measurement |
| RAG (with data change) | + index_update per change | Direct measurement |

---

## 7. Gold Label Strategy

### 7.1 Sources

1. **BrowseComp-Plus subset:** Existing ground-truth for exact-answer queries
2. **arXiv metadata:** Category labels as relevance proxy

### 7.2 Label Schema

```python
@dataclass
class GoldLabel:
    query_id: str
    chunk_id: str
    relevance: int              # 0 = irrelevant, 1 = relevant, 2 = highly relevant
    source: str                 # "browsecomp" | "arxiv_category" | "manual"
```

### 7.3 Coverage Targets

| Source | Queries | Labeled Chunks | Purpose |
|--------|---------|----------------|---------|
| BrowseComp-Plus | 20-30 | ~500 | Exact-answer quality |
| arXiv categories | 10-20 | ~2000 | Topic filtering quality |

---

## 8. Weave Integration

### 8.1 Project

```python
import weave

weave.init("sasha-krigel-massachusetts-institute-of-technology/inference-hack")
```

The code path is now folded into the eval harness:

- `eval/weave_ops.py` owns the default project string and optional `@weave.op()` wrapper.
- `eval/bench.py --smoke --weave` initializes that project and runs the existing smoke eval through
  the traced op.
- `eval/bench.py --smoke` remains dependency-free and must keep working without Weave installed.
- `WEAVE_PROJECT=team/project` or `--weave-project team/project` can override the destination for
  dry runs, but the freeze run uses the project above.

Eval box setup:

```bash
python -m pip install -r eval/requirements.txt
wandb login
python -m eval.bench --smoke --weave
```

Local/no-auth validation:

```bash
python -m eval.bench --smoke
python -m unittest tests.test_eval_weave_ops
```

`--weave` should fail fast with an install/auth message when Weave is unavailable; it should never
silently skip logging on a freeze run.

### 8.2 What Gets Logged

The current traced smoke op logs the full `TurnTrace.to_dict()` payload. As session/task runners land,
they should wrap the same trace rows in nested ops:

```python
from eval.weave_ops import weave_op

@weave_op(name="eval.score_chunk")
def score_chunk(chunk: str, predicate: str) -> float:
    """Every model call is tracked."""
    ...

@weave_op(name="eval.run_session")
def run_session(session: ScriptedSession) -> SessionMetrics:
    """Session-level tracking with nested calls."""
    ...

@weave_op(name="eval.run_task")
def run_task(task: CompletionTask) -> TaskCompletionMetrics:
    """Task-level tracking."""
    ...
```

### 8.3 Tracked Attributes

Per run:
- `commit`: git SHA
- `corpus_id`: which corpus version
- `model_id`: which model (e.g., "llama-3-8b-awq")
- `regime`: optimization config (B0/B1/B2/B3) once ladder sweeps land
- `scorer_backend`: "mock" | "vllm"
- `chunks_scored`, `chunks_served_from_cache`, `cache_hit_rate`, `rho`
- `elapsed_ms`, `model_ms`, `queue_ms`, `ttft_ms`
- `quality_slice`: null for smoke, populated for gate/task runs

### 8.4 Comparison Views

Weave provides:
- Side-by-side run comparison
- Metric aggregation across runs
- Latency distributions
- Score histograms

---

## 9. Measurement Hygiene

From `performance/docs/02_benchmarking_methodology.md`:

### 9.1 Rules

- [ ] **Count inference, don't just time it.** Primary metric is `chunks_scored`.
- [ ] **CUDA sync before timing.** `torch.cuda.synchronize()` or CUDA events.
- [ ] **Lock GPU clocks.** `nvidia-smi -lgc <freq>` for reproducibility.
- [ ] **Control cold vs warm explicitly.** Reset cache for cold measurements.
- [ ] **Discard warmup iterations.** First few batches are meaningless.
- [ ] **Capture naive floor first.** B0 is perishable.
- [ ] **Label every number.** `predicted`, `measured`, or `projected`.

### 9.2 Counterfactual Replay

One instrumented scoped session yields all comparison curves:

```python
def counterfactual_replay(trace: SessionTrace) -> Dict[str, List[int]]:
    """Derive comparison curves from a single measured trace"""
    n = trace.corpus_size
    k = len(trace.turns)

    return {
        "scoped_measured": trace.cumulative_chunks_scored,
        "full_rescore": [i * n for i in range(1, k+1)],
        "suffix_only": [i * n * (SUFFIX_LEN / TOTAL_LEN) for i in range(1, k+1)],
    }
```

---

## 10. Implementation Roadmap

### 10.1 Directory Structure

```
eval/
├── __init__.py
├── config.py               # ComputeRegime, constants
├── trace.py                # TurnTrace, logging
├── weave_ops.py            # Weave-decorated operations
│
├── sessions/
│   ├── __init__.py
│   ├── definitions.py      # 6 scripted sessions (YAML or dataclass)
│   ├── runner.py           # execute session, collect metrics
│   └── metrics.py          # SessionMetrics
│
├── tasks/
│   ├── __init__.py
│   ├── definitions.py      # 10 completion tasks
│   ├── gold_labels.json    # target chunk IDs
│   ├── runner.py           # execute task, collect metrics
│   └── metrics.py          # TaskCompletionMetrics
│
├── ablation.py             # run all sessions across all regimes
├── comparison.py           # head-to-head with RAG
├── replay.py               # counterfactual curve derivation
├── report.py               # generate summary tables + figures
│
└── bench.py                # CLI entrypoint: orchestrates everything

baseline/
├── __init__.py
├── rag.py                  # main RAG pipeline
├── embedder.py             # embedding model
├── index.py                # FAISS index
├── reranker.py             # cross-encoder (optional)
└── metrics.py              # RAGMetrics
```

### 10.2 Priority Order

| Phase | Component | Effort | Deliverable |
|-------|-----------|--------|-------------|
| **P0** | `trace.py` + Weave setup | 2h | Every call logged |
| **P0** | `sessions/definitions.py` | 1h | 6 sessions defined |
| **P0** | `sessions/runner.py` | 2-3h | Sessions executable |
| **P0** | `tasks/definitions.py` + gold labels | 2-3h | 10 tasks + labels |
| **P0** | `tasks/runner.py` | 2-3h | Tasks executable |
| **P1** | `baseline/rag.py` | 2-3h | RAG baseline working |
| **P1** | `ablation.py` | 2h | Optimization ladder sweep |
| **P1** | `comparison.py` | 2h | Head-to-head metrics |
| **P1** | `replay.py` | 1h | Counterfactual curves |
| **P2** | `report.py` | 2-3h | Summary tables + figures |
| **P2** | Fresh file injection test | 1h | The "no reindex" beat |

### 10.3 Timeline Integration

| Hack Phase | Eval Work |
|------------|-----------|
| H0-H3 | Weave setup, trace logging, capture B0 naive floor |
| H3-H8 | Session runner working, first quality check |
| H8-H14 | Task runner, ablation sweep, RAG baseline |
| H14-H19 | Full comparison, figures for slides |
| H19-H22 | Polish, final numbers |

---

## 11. Key Figures for Slides

### Figure 1: Optimization Ladder (Waterfall)

```
Refine latency (ms)
^
|████████████████████████████  B0: 2000ms (naive)
|████████████████              B1: 800ms  (+warm KV)
|████████                      B2: 300ms  (+scoped)
|██████                        B3: 200ms  (+score cache)
+-------------------------------->
```

### Figure 2: Area Under the Loop

```
Cumulative chunks scored
^
|                              Full re-score (linear)
|                            /
|                          /
|                        /
|      ________________/  Ours scoped (saturates)
|    /
|  /
+-------------------------> Turn
```

### Figure 3: Compute Utilization (Roofline)

```
TFLOP/s
^
|______________ Compute ceiling (7.9 PFLOP/s)
|           *  Our scan (compute-bound)
|         /
|       /
|     /
|   * RAG rerank
| /
|* Decode (BW-bound)
+-------------------------> Arithmetic Intensity
```

### Figure 4: RAG vs Ours — Session Latency

```
Total session time (ms)
^
|████████████████████████  RAG: 4500ms
|██████████                Ours: 1800ms
+------------------------->
```

### Figure 5: Fresh File Injection

```
Latency (ms)
^
|████████████████████████████████  RAG: 3500ms (index) + 150ms (query)
|███                               Ours: 200ms (immediate)
+------------------------->
```

---

## 12. Success Criteria

### 12.1 Quality Gate (Before Speed Optimization)

| Metric | Threshold | Checked |
|--------|-----------|---------|
| F1 on BrowseComp subset | ≥ 0.7 | [ ] |
| Precision@20 | ≥ 0.6 | [ ] |
| No catastrophic failures | 0 sessions with F1 < 0.3 | [ ] |

### 12.2 Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Refine latency p50 (warm, scoped) | < 300ms | [ ] |
| Refine latency p95 (warm, scoped) | < 1s | [ ] |
| Time-to-first-result | < 2s | [ ] |
| Scoping efficiency (vs full re-score) | > 50% | [ ] |
| MFU (cold full scan) | > 30% | [ ] |

### 12.3 RAG Comparison Wins

| Comparison | We Should Win | Measured |
|------------|---------------|----------|
| Index build time | 0 vs minutes | [ ] |
| 6-turn session latency | lower | [ ] |
| Fresh file query | 10× faster | [ ] |
| Iteration scaling | saturates vs linear | [ ] |

### 12.4 Task Completion

| Metric | Target |
|--------|--------|
| Average recall across 10 tasks | ≥ 0.75 |
| Tasks with success=True | ≥ 8/10 |
| Average turns to 80% recall | < 5 |

---

## 13. Open Questions

> Resolve these during implementation:

1. **BrowseComp integration:** Do we have access? What's the format?
2. **Corpus composition:** Exact arXiv categories and code repos to include?
3. **Mock scorer fidelity:** How closely does mock match real vLLM scoring?
4. **Weave project setup:** Organization, project name, API keys?
5. **RAG baseline model:** Same embedding model as typical RAG tutorials, or match our model size?

---

## Appendix A: Config Schema

```python
@dataclass
class ComputeRegime:
    """Optimization configuration for ablation"""
    scoped: bool = True
    score_cache: bool = True
    warm_kv: bool = True
    prefix_caching: bool = True

    @classmethod
    def B0_baseline(cls): return cls(False, False, False, False)

    @classmethod
    def B1_warm(cls): return cls(False, False, True, True)

    @classmethod
    def B2_scoped(cls): return cls(True, False, True, True)

    @classmethod
    def B3_cached(cls): return cls(True, True, True, True)
```

## Appendix B: Weave Example

```python
import weave

weave.init("grep-for-meaning-eval")

@weave.op()
def run_evaluation():
    results = {}

    # Ablation across regimes
    for regime_name, regime in REGIMES.items():
        regime_results = []
        for session in SESSIONS:
            metrics = run_session(session, regime)
            regime_results.append(metrics)
        results[regime_name] = regime_results

    # RAG comparison
    rag_results = run_rag_comparison()
    results["rag_comparison"] = rag_results

    return results
```
