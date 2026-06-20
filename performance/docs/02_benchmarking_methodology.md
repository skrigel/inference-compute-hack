# 02 — Benchmarking methodology

How to measure these curves without lying to yourself. The instrumentation
strategy determines which graphs are even possible, so decide it before writing
`eval/bench.py`.

---

## The principle that changes everything: count inference, don't just time it
For four of the five graphs the x-axis is **compute, not seconds.** The primary
instrument is therefore a **counter inside the score cache**, not a stopwatch.
Each turn, log:
- `chunks_scored` — cache misses that actually hit the model (the compute unit)
- `chunks_served_from_cache` — cache hits (free)

That integer is deterministic, noise-free, hardware-independent, and is exactly
what `theory.py` predicts. Time is a *separate* overlay for the felt-latency
story. Conflating the two is the classic mistake here: under continuous batching,
latency is entangled with batch composition, so a clean compute count reproduces
where a wall-clock wobbles.

### Counterfactual replay (build the harness around this)
Instrument **one** real scoped refine session — record candidate-set size and
survivor fraction `ρ` at each turn. From that single trace, compute *all four*
area-under-loop lines analytically:
- scoped = what you measured
- full re-score = `k·N`
- suffix-only = `k·N·(s/L)`
- RAG = per-query cost + injected re-index steps

One instrumented run yields every curve, and because they share the same session
they are a fair head-to-head **by construction**. Use live A/B runs only to
*validate* that the analytic full-rescore number matches an actual full rescore on
a couple of sessions.

---

## Measurement hygiene (skip these and every graph lies)
1. **CUDA is async.** Wrap device timing in `torch.cuda.synchronize()` or CUDA
   events, or you time kernel *launch*, not execution.
2. **Lock the clocks.** `nvidia-smi -lgc <freq>` to pin GPU clocks; otherwise
   thermal boost variance across the 8 cards jitters the roofline points and MFU
   bars run-to-run, and the perishable naive floor (H0) won't be comparable to
   later optimized rungs.
3. **Control cold vs warm explicitly.** vLLM's prefix cache persists across
   requests; a "cold" measurement needs an actual cache reset or a first-touch
   corpus, or warm reuse leaks in and inflates the cold baseline. **Capture the
   naive cold floor first**, before optimizations make it impossible to isolate.
4. **Discard warmup iterations.** CUDA graph capture, vLLM compilation, allocator
   warmup make the first few batches meaningless. Measure steady state.

---

## Per-figure technique

### Roofline (Fig 1)
- **Cheap path:** theoretical FLOPs (`2PT`) ÷ measured device time → achieved
  TFLOP/s vs ceiling. Fine for a slide.
- **Rigorous path:** a **batch-size sweep** — plot achieved points walking up the
  bandwidth diagonal until they flatten against the compute ceiling. Watching the
  climb is far more convincing than three static dots.
- **Prove compute-bound, don't assert it:** sample SM utilization and DRAM-BW
  utilization (DCGM / `nvidia-smi dmon`) during the scan. Compute-bound looks like
  SM saturated, DRAM moderate, board power near the 700 W TDP.
- **True per-kernel FLOP/byte counts:** Nsight Compute (`ncu`) — but it replays
  kernels and is slow; run once on a small representative batch, not the corpus.

### MFU waterfall (Fig 2)
- One config flag per rung, swept cumulatively in build-effort order (§11 ladder).
- **Token-counting subtlety:** MFU on *real* tokens vs *padded* tokens can differ
  a lot. Log both; report which denominator you used (continuous batching mostly
  saves you, but say so).
- Measure single-replica MFU clean, then verify ~linear scaling to 8 under
  data-parallel. If aggregate MFU sags below 8× single-replica, you found a router
  / loading bottleneck — that's a slide.
- **Power is a free sanity check:** if a bar claims 50% MFU but the board draws
  300 W, the measurement is wrong.

### Area under the loop (Fig 3)
- Counterfactual replay (above). Experimental design: define a handful of
  **canonical scripted sessions** (6–10 turns, known ops, real predicates) and
  hold them fixed while varying the mechanism. Same session, mechanism toggled —
  the only way the four curves are comparable.
- Measure `ρ` per turn empirically; overlay measured cumulative work against
  predicted `N(1−ρ^k)/(1−ρ)` and show the residual. Run several sessions for a
  distribution, not one anecdote.

### Compute-vs-churn (Fig 4)
- Three isolated microbenchmarks feeding the closed form: one full scan
  (`C_scan`), one end-to-end RAG query (embed + ANN + rerank = `C_rag_query`), one
  index build (`C_index`). Same unit → break-even `D*` falls out.
- **Strongly consider joules, not seconds, as the cost axis.** Integrate board
  power over time (DCGM samples power at ~ms) for energy per query and per
  re-index. Energy is the most honest "cost," it's the unit that makes
  recompute-over-store land, and judges remember joules-per-query.
- Validate the line at two or three real `D` values rather than trusting the
  formula blind.

### KV capacity (Fig 5)
- **Don't trust the formula alone — measure real allocation.** vLLM PagedAttention
  allocates in blocks with fragmentation overhead, *and* prefix caching dedups the
  shared instruction prefix across chunks, so real footprint may come in *under*
  the naive `bytes/token · prefix_len · N` estimate.
- Log `gpu_cache_usage_perc` and allocated-block count during the warm pass; sweep
  corpus size until it evicts/OOMs; mark that empirical crossover against the
  predicted ~14k. **The gap between predicted and measured is the dedup benefit** —
  itself a nice finding.

---

## Tooling (most of the work is free)
- **vLLM `/metrics`** (Prometheus): throughput, time-to-first-token, queue depth,
  cache usage. Since decode is a single token, **TTFT ≈ per-chunk scan latency** —
  many headline numbers are already emitted; scrape and aggregate.
- **DCGM** (`dcgmi dmon`) across all 8 GPUs: power / util / clocks at a fixed
  sample interval.
- **Nsight Systems** (`nsys`): one timeline capture to spot async gaps and confirm
  GPUs aren't starving on the FastAPI/SSE path.
- **Nsight Compute** (`ncu`): true per-kernel FLOP/byte (slow; small batch only).
- **CUDA events** inside `score.py`: precise device timing. Everything else is
  `time.perf_counter()` + a synchronize.

## One workload-specific gotcha
Measure **end-to-end through the FastAPI/SSE path** for the *felt-latency* graphs,
but **at the model** for the *compute* graphs. The demo sells the former; the rigor
needs the latter; they differ by tens of ms of queueing + serialization you don't
want polluting MFU.

---

## Suggested `eval/bench.py` shape (not yet built)
- `cache_counter` instrumentation (chunks_scored / served_from_cache per turn)
- `counterfactual_replay(session_trace)` → emits all four area-under-loop curves
- measured-vs-predicted overlay hooks against `theory.py`
  (`implied_mfu`, `suffix_speedup`, `RagModel.breakeven_changes`,
  `kv_crossover_chunks`)
- batch-size sweep driver for the roofline
- DCGM power integrator for the joules axis

This is the concrete content of H0–3's "stand up the eval harness."
