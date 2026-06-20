# Refinement Phase: Prefill Throughput & GPU Utilization

> **Goal:** Maximize prefill throughput and GPU utilization after MVP baseline is established.
> **Approach:** Measure first, then config-only + light modifications.
> **Prerequisite:** Quality gate (F1 ≥ 0.7) must pass before speed optimization.

---

## 0. Baseline (Already in MVP — Do Not Duplicate)

These are **already implemented** in the MVP plan. Do not re-implement:

| Optimization | MVP Location | Status |
|--------------|--------------|--------|
| Suffix-only re-prefill | PLAN §5 #2 | ✅ Primary refine mechanism |
| Candidate-set scoping | PLAN §5 #3 | ✅ Primary refine mechanism |
| Score cache `(chunk_id, clause_id) → float` | CONTRACTS §5 | ✅ In backend/cache.py |
| Prefix caching | `enable_prefix_caching=True` | ✅ In vLLM config |
| FP8 compute | PLAN §5 #5 | ✅ Throughput lever |
| 4-bit weights | PLAN §5 #5 | ✅ Capacity lever |
| 6 data-parallel replicas | PLAN §5 #5 | ✅ In serve.sh |

**The refinement phase builds ON TOP of these, not instead of them.**

---

## 1. Measure First (Phase R0)

**You can't optimize what you haven't measured.** Before any optimization:

### R0.1: Capture Baseline Metrics

```bash
# Run eval harness with current MVP config
python eval/bench.py --backend=vllm --sessions=all --output=baseline.json
```

**Metrics to capture:**

| Metric | How to Measure | Target |
|--------|----------------|--------|
| **GPU utilization %** | `nvidia-smi dmon -s u` | > 80% |
| **GPU power %** | `nvidia-smi dmon -s p` | > 70% TDP |
| **Prefill tok/s** | vLLM `/metrics` or manual | 20k+ tok/s/GPU |
| **MFU** | `theory.implied_mfu(tokens, seconds)` | > 40% |
| **Batch size actual** | vLLM logs | Should be >> 1 |
| **KV cache usage %** | vLLM `/metrics` `gpu_cache_usage_perc` | < 90% |
| **Straggler ratio** | `max(gpu_time) / mean(gpu_time)` | < 1.3 |

### R0.2: Identify Your Bottleneck

```
IF gpu_utilization < 50%:
    → Bottleneck is BATCH SIZE (not enough work per call)
    → Go to Section 2 (Batching)

IF gpu_utilization > 80% BUT throughput low:
    → Bottleneck is COMPUTE CONFIG (FP8, attention backend)
    → Go to Section 3 (Compute)

IF straggler_ratio > 1.5:
    → Bottleneck is LOAD IMBALANCE
    → Go to Section 4 (Scheduling)

IF kv_cache_usage > 90%:
    → Bottleneck is MEMORY
    → Go to Section 5 (Memory)
```

---

## 2. Batching Optimizations (Config-Only)

**Problem:** Small batches → GPU underutilized.

### 2.1: Large Batch Dumps

**Instead of streaming one chunk at a time, batch everything:**

```python
# BAD: sequential
for chunk in chunks:
    score = scorer.score(chunk)  # GPU underutilized

# GOOD: batched
scores = scorer.score_batch(chunks)  # GPU saturated
```

**vLLM config:**
```python
engine_args = EngineArgs(
    max_num_batched_tokens=8192,  # Increase from default
    max_num_seqs=256,             # Allow more concurrent sequences
)
```

### 2.2: Ridge Point Batching

**Concept:** Batch size where arithmetic intensity hits compute ceiling.

From Modal benchmarks: **~600 tokens per GPU** is the ridge point for H100 FP8.

```python
# For 6 replicas: target ~3600 tokens per batch across all GPUs
TOKENS_PER_BATCH = 600 * 6  # 3600 tokens

def batch_by_tokens(chunks: List[Chunk], target_tokens: int = 3600):
    """Group chunks to hit ridge point"""
    batches = []
    current = []
    current_tokens = 0

    for chunk in sorted(chunks, key=lambda c: len(c.tokens)):
        chunk_tokens = len(chunk.tokens)
        if current_tokens + chunk_tokens > target_tokens and current:
            batches.append(current)
            current = [chunk]
            current_tokens = chunk_tokens
        else:
            current.append(chunk)
            current_tokens += chunk_tokens

    if current:
        batches.append(current)
    return batches
```

### 2.3: Measure Batch Impact

```python
# Before
baseline_throughput = measure_throughput(batch_size=64)

# After
optimized_throughput = measure_throughput(batch_size=256)

print(f"Improvement: {optimized_throughput / baseline_throughput:.1f}x")
```

**Expected gain:** +50-100% throughput if currently batch-limited.

---

## 3. Compute Optimizations (Config-Only)

**Problem:** Not using H100 tensor cores efficiently.

### 3.1: FlashInfer Attention Backend

**Best for offline/throughput workloads:**

```python
engine_args = EngineArgs(
    attention_backend="flashinfer",  # vs default "flash_attn"
)
```

**When to use:** Batch inference, not latency-sensitive streaming.

### 3.2: FP8 Compute (Already in MVP)

Verify FP8 is actually enabled:

```bash
# Check vLLM logs for:
# "Using FP8 quantization" or similar

# Or check model dtype:
python -c "from vllm import LLM; llm = LLM('model'); print(llm.llm_engine.model_config.dtype)"
```

**Expected:** ~2× prefill throughput vs BF16.

### 3.3: Async Scheduling

```python
engine_args = EngineArgs(
    async_scheduling=True,  # Modest improvement
)
```

**Expected gain:** +5-10% (modest but free).

---

## 4. Scheduling Optimizations (Light Modifications)

**Problem:** Uneven work distribution → some GPUs idle.

### 4.1: SRJF (Shortest Remaining Job First)

**Why it works:** Single-token output means JCT is predictable from input length.

```python
# In backend or scoring client
def schedule_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Process shortest chunks first — minimizes average JCT"""
    return sorted(chunks, key=lambda c: len(c.tokens))
```

**Where to add:** `inference/vllm_client.py` before calling `score_batch`.

**Expected gain:** -20-30% p50 latency, better tail latency.

### 4.2: Load Balancing Across 6 Replicas

**Current MVP approach:** Stable hash pins chunks to replicas (PLAN §5 #4).

**Problem:** Uneven chunk lengths → some replicas finish early.

**Improvement:** Greedy bin-packing by token count.

```python
def balance_across_replicas(chunks: List[Chunk], n_replicas: int = 6):
    """Distribute chunks to minimize max replica work"""
    # Sort longest first (greedy bin-packing)
    sorted_chunks = sorted(chunks, key=lambda c: len(c.tokens), reverse=True)

    replica_loads = [0] * n_replicas
    assignments = [[] for _ in range(n_replicas)]

    for chunk in sorted_chunks:
        # Assign to least-loaded replica
        min_replica = replica_loads.index(min(replica_loads))
        assignments[min_replica].append(chunk)
        replica_loads[min_replica] += len(chunk.tokens)

    # Log balance quality
    variance = max(replica_loads) - min(replica_loads)
    mean_load = sum(replica_loads) / n_replicas
    straggler_ratio = max(replica_loads) / mean_load
    print(f"Load variance: {variance} tokens, straggler ratio: {straggler_ratio:.2f}")

    return assignments
```

**Where to add:** `inference/vllm_client.py` in the routing logic.

**Expected gain:** -10-20% total time when chunks have high length variance.

### 4.3: Measure Straggler Impact

```python
import time

def measure_replica_times(assignments, scorer):
    """Measure per-replica completion times"""
    times = []
    for replica_chunks in assignments:
        start = time.perf_counter()
        scorer.score_batch(replica_chunks, replica=i)
        times.append(time.perf_counter() - start)

    print(f"Replica times: {times}")
    print(f"Straggler ratio: {max(times) / (sum(times)/len(times)):.2f}")
```

---

## 5. Memory Optimizations (Config + Light Mods)

**Problem:** KV cache fills up → can't warm full corpus.

### 5.1: KV Cache Dtype (Config-Only)

From MVP §5 #4: FP16 KV crosses 640 GB at ~14k chunks.

```python
# Option 1: FP8 KV (fits ~28k chunks)
engine_args = EngineArgs(
    kv_cache_dtype="fp8",
)

# Option 2: 4-bit KV (fits ~56k chunks) — if supported
engine_args = EngineArgs(
    kv_cache_dtype="int4",  # Check vLLM version support
)
```

### 5.2: Memory-Aware Batch Sizing

**Problem:** Large batches cause memory spikes during prefill.

```python
def find_safe_batch_size(target_memory_pct: float = 0.85):
    """Binary search for max batch size without OOM"""
    low, high = 64, 512
    safe_size = low

    while low <= high:
        mid = (low + high) // 2
        try:
            # Test batch
            test_chunks = generate_test_chunks(mid)
            scorer.score_batch(test_chunks)

            # Check memory
            mem_pct = get_gpu_memory_pct()
            if mem_pct < target_memory_pct:
                safe_size = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError:  # OOM
            high = mid - 1

    return safe_size
```

### 5.3: Chunked Prefill for Long Inputs

**For chunks that exceed memory budget:**

```python
def chunked_prefill(tokens: List[int], window_size: int = 2048):
    """Process long prefill in fixed-size windows"""
    if len(tokens) <= window_size:
        return prefill(tokens)

    # Process in windows
    kv_cache = None
    for i in range(0, len(tokens), window_size):
        window = tokens[i:i+window_size]
        kv_cache = prefill_window(window, kv_cache)
    return kv_cache
```

**Note:** This is a light modification to `inference/vllm_client.py`.

---

## 6. vLLM Configuration Deep Dive

### 6.1 Key Parameters for Prefill-Heavy Workloads

Based on [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/) and [Optimization Guide](https://docs.vllm.ai/en/stable/configuration/optimization/):

#### Batching & Scheduling (HIGHEST IMPACT)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--max-num-batched-tokens` | auto | **8192-16384** | "For optimal throughput, set >8192 especially for smaller models on large GPUs" |
| `--max-num-seqs` | auto | **256-512** | More concurrent sequences = better GPU utilization |
| `--enable-chunked-prefill` | True (V1) | Keep enabled | Balances prefill/decode; splits large prefills |
| `--performance-mode` | balanced | **throughput** | Optimizes for batch processing over latency |
| `--scheduling-policy` | fcfs | fcfs | Keep default; SRJF done at our layer |

#### Memory & KV Cache (HIGH IMPACT)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--gpu-memory-utilization` | 0.92 | **0.90-0.95** | Higher = more KV cache; lower = more batch headroom |
| `--kv-cache-dtype` | auto | **fp8** | 2× capacity vs FP16; fits ~28k chunks vs ~14k |
| `--enable-prefix-caching` | False | **True** | Critical for instruction prefix reuse |
| `--block-size` | auto | auto | Let vLLM optimize |

#### Compute & Attention (MEDIUM IMPACT)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--dtype` | auto | auto | Let vLLM choose (usually bfloat16) |
| `--quantization` | None | **awq_marlin** | 4-bit weights for capacity |
| `--attention-backend` | auto | auto or flashinfer | Auto-selects best for GPU arch |
| `--linear-backend` | auto | auto | GEMM kernel selection |
| `--optimization-level` | 2 | 2 | Default is good; -O3 same as -O2 currently |

#### Parallelism (ARCHITECTURE CHOICE)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--tensor-parallel-size` | 1 | **1** | Data-parallel replicas, not tensor-parallel |
| `--data-parallel-size` | 1 | 1 (external) | We manage 6 replicas externally |

#### Monitoring (ESSENTIAL)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--enable-mfu-metrics` | False | **True** | Exposes Model FLOPs Utilization |
| `--kv-cache-metrics` | False | **True** | Track KV residency and reuse |
| `--disable-log-stats` | False | False | Keep stats for debugging |

### 6.2 Recommended Configuration

```bash
# serve.sh for each replica
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
    --quantization awq_marlin \
    --max-model-len 4096 \
    \
    # Batching (CRITICAL)
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --performance-mode throughput \
    \
    # Memory
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    \
    # Monitoring
    --enable-mfu-metrics \
    --kv-cache-metrics \
    \
    # Single GPU per replica
    --tensor-parallel-size 1
```

### 6.3 Monitoring via /metrics Endpoint

vLLM exposes Prometheus metrics at `http://localhost:8000/metrics`:

```bash
# Key metrics to watch
curl -s http://localhost:8000/metrics | grep -E "(gpu_cache|mfu|running|waiting)"
```

| Metric | What It Tells You | Target |
|--------|-------------------|--------|
| `vllm:gpu_cache_usage_perc` | KV cache utilization | < 90% (headroom for spikes) |
| `vllm:num_requests_running` | Active requests | Should be high |
| `vllm:num_requests_waiting` | Queue depth | Low = good; high = bottleneck |
| `vllm:model_flops_utilization` | GPU compute efficiency | > 40% (with `--enable-mfu-metrics`) |
| `vllm:time_to_first_token_seconds` | TTFT distribution | p50 < 500ms |
| `vllm:prefill_tokens_total` | Prefill throughput | Monitor trend |

### 6.4 Parameter Sweep Experiments

Run these experiments to find optimal settings:

```bash
# Experiment 1: max_num_batched_tokens sweep
for tokens in 2048 4096 8192 16384; do
    echo "Testing max_num_batched_tokens=$tokens"
    # Start vLLM with this setting
    # Run eval harness
    # Record throughput + latency
done

# Experiment 2: gpu_memory_utilization sweep
for mem in 0.85 0.90 0.92 0.95; do
    echo "Testing gpu_memory_utilization=$mem"
    # Start vLLM with this setting
    # Run eval harness
    # Record throughput + OOM events
done

# Experiment 3: kv_cache_dtype comparison
for dtype in auto fp8; do
    echo "Testing kv_cache_dtype=$dtype"
    # Start vLLM with this setting
    # Run eval harness
    # Record KV capacity + quality
done
```

### 6.5 Configuration Matrix

| Workload | max_num_batched_tokens | gpu_memory_utilization | kv_cache_dtype | Notes |
|----------|------------------------|------------------------|----------------|-------|
| **Cold query (throughput)** | 16384 | 0.90 | fp8 | Max batch, moderate memory |
| **Warm refine (latency)** | 4096 | 0.95 | fp8 | Smaller batch, more KV |
| **Memory constrained** | 8192 | 0.85 | fp8 | Leave headroom |
| **Quality sensitive** | 8192 | 0.92 | auto | FP16 KV for precision |

---

## 7. Implementation Priority

### Phase R0: Measure (2-3 hours) — DO THIS FIRST

| Task | Effort | Output |
|------|--------|--------|
| Run baseline eval harness | 1h | `baseline.json` |
| Capture GPU metrics (utilization, power, memory) | 30m | Dashboard/logs |
| Identify bottleneck (batch/compute/memory/straggler) | 30m | Decision on next phase |

### Phase R1: Config-Only (2-3 hours)

| Task | Effort | Expected Gain | Risk |
|------|--------|---------------|------|
| Tune `max_num_batched_tokens` | 30m | +50-100% if batch-limited | Low |
| Enable `attention_backend="flashinfer"` | 15m | +10-20% | Low |
| Enable `async_scheduling=True` | 5m | +5-10% | Low |
| Set `kv_cache_dtype="fp8"` | 15m | 2× KV capacity | Low |

### Phase R2: Light Modifications (3-4 hours)

| Task | Effort | Expected Gain | Risk |
|------|--------|---------------|------|
| Add SRJF scheduling | 1h | -20-30% p50 latency | Low |
| Add load balancing | 2h | -10-20% total time | Low-Medium |
| Add memory-aware batch sizing | 1h | Prevent OOM | Low |

### Phase R3: Post-Hackathon (Stretch)

| Task | Effort | Expected Gain | Risk |
|------|--------|---------------|------|
| Prefix tree for predicates | 4-6h | Depends on patterns | Medium |
| Document continuation | 3-4h | Large doc support | Medium |
| Last-layer KV (PrefillOnly) | 8-12h | ~32× memory | High |

---

## 8. Measurement Protocol

### Before Each Optimization

```bash
# Capture baseline
python eval/bench.py --sessions=all --output=before_${OPT_NAME}.json
nvidia-smi dmon -s puc -d 1 > gpu_before_${OPT_NAME}.log &
```

### After Each Optimization

```bash
# Capture optimized
python eval/bench.py --sessions=all --output=after_${OPT_NAME}.json
nvidia-smi dmon -s puc -d 1 > gpu_after_${OPT_NAME}.log &

# Compare
python eval/compare.py before_${OPT_NAME}.json after_${OPT_NAME}.json
```

### A/B Comparison Table

| Metric | Baseline | +R1 Config | +R2 Scheduling | Target |
|--------|----------|------------|----------------|--------|
| Throughput (tok/s) | ? | ? | ? | 20k+/GPU |
| MFU | ? | ? | ? | > 40% |
| GPU utilization % | ? | ? | ? | > 80% |
| Refine p50 (ms) | ? | ? | ? | < 300ms |
| Straggler ratio | ? | ? | ? | < 1.3 |
| KV memory (GB) | ? | ? | ? | < 500GB |
| F1 (quality) | ? | ? | ? | ≥ 0.7 |

---

## 9. Decision Tree

```
START
  │
  ▼
┌─────────────────────────────┐
│ R0: Measure baseline        │
│ GPU util, throughput, memory│
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ GPU utilization < 50%?      │──YES──▶ R1: Increase batch size
└─────────────────────────────┘         (max_num_batched_tokens)
  │ NO
  ▼
┌─────────────────────────────┐
│ Straggler ratio > 1.3?      │──YES──▶ R2: Add load balancing
└─────────────────────────────┘
  │ NO
  ▼
┌─────────────────────────────┐
│ KV cache > 90%?             │──YES──▶ R1: kv_cache_dtype=fp8
└─────────────────────────────┘
  │ NO
  ▼
┌─────────────────────────────┐
│ MFU < 40%?                  │──YES──▶ R1: flashinfer + FP8
└─────────────────────────────┘
  │ NO
  ▼
┌─────────────────────────────┐
│ Already optimized!          │
│ Consider R3 stretch goals   │
└─────────────────────────────┘
```

---

## 10. Resources

**MVP Integration:**
- `METRICS.md` — trace schema and metric hierarchy
- `performance/docs/02_benchmarking_methodology.md` — measurement hygiene
- `CONTRACTS.md` §6 — trace schema

**External:**
- [PrefillOnly (SOSP 2025)](https://arxiv.org/pdf/2505.07203) — Prefill-only workload insights
- [Modal vLLM Throughput](https://modal.com/docs/examples/vllm_throughput) — 30k tok/s config
- [vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html)

---

## 11. Quick Reference: What Changes Where

| Optimization | File to Modify | Type |
|--------------|----------------|------|
| Batch size | `inference/serve.sh` or vLLM config | Config |
| FlashInfer | `inference/serve.sh` or vLLM config | Config |
| KV dtype | `inference/serve.sh` or vLLM config | Config |
| Async scheduling | `inference/serve.sh` or vLLM config | Config |
| SRJF scheduling | `inference/vllm_client.py` | Light mod |
| Load balancing | `inference/vllm_client.py` | Light mod |
| Memory-aware batching | `inference/vllm_client.py` | Light mod |