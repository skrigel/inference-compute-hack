"""
Modal deployment for Grep for Meaning MVP.

Deploys 6 data-parallel vLLM replicas on H100 GPUs for single-token Yes/No scoring.
This is the MVP configuration - refinement phase optimizations are separate.

Usage:
    modal deploy inference/modal_app.py          # Deploy to Modal
    modal serve inference/modal_app.py           # Local dev server
    modal run inference/modal_app.py::test       # Health check

Environment:
    MODAL_TOKEN_ID, MODAL_TOKEN_SECRET must be set, or `modal token new` run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import modal

# -----------------------------------------------------------------------------
# Configuration (matches PLAN.md §5 and CONTRACTS.md)
# -----------------------------------------------------------------------------

# Tier-1: small model, single GPU per replica, 6 replicas
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
MODEL_REVISION = "main"
N_REPLICAS = 6  # PLAN §5 #5: "6 fully independent single-GPU replicas"
VLLM_METRICS_VERSION = os.environ.get("VLLM_METRICS_VERSION", "0.22.1")
H100_SXM_BF16_FLOPS_PER_GPU = 989.5e12
H100_SXM_FP8_FLOPS_PER_GPU = 1979.0e12

# Tier-2 (stretch goal, commented out): large model, TP=2
# TIER2_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
# TIER2_TP = 2


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


GPU_MEMORY_UTILIZATION = _env_float("GPU_MEMORY_UTILIZATION", 0.92)
ENABLE_MFU_METRICS = _env_flag("ENABLE_MFU_METRICS", True)
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")
SCORER_MIN_CONTAINERS = _env_int("SCORER_MIN_CONTAINERS", N_REPLICAS)

# vLLM engine configuration per PLAN.md and REFINEMENTS.md
VLLM_ENGINE_KWARGS: dict[str, Any] = {
    "max_model_len": 4096,           # PLAN §5: context window
    "max_num_seqs": 256,             # concurrent sequences
    "max_num_batched_tokens": 8192,  # prefill throughput lever
    "quantization": "awq_marlin",    # AWQ 4-bit weights via Marlin kernels
    "kv_cache_dtype": KV_CACHE_DTYPE,  # fp8 is sweepable but may fail on some vLLM/Torch stacks
    "enable_prefix_caching": True,   # PLAN §5 #2: suffix-only re-prefill
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,  # parameterized for Phase 04 sweep
    "enforce_eager": False,          # allow CUDA graphs
    "disable_log_stats": False,      # we want stats for metrics
}
if ENABLE_MFU_METRICS:
    # vLLM exposes this as `--enable-mfu-metrics` in newer versions. Some older
    # Python APIs do not accept the kwarg, so start_engine retries without it.
    VLLM_ENGINE_KWARGS["enable_mfu_metrics"] = True

# Sampling for single-token Yes/No scoring (PLAN §5 #1)
DEFAULT_SAMPLING_KWARGS: dict[str, Any] = {
    "max_tokens": 1,                 # single-token output
    "logprobs": 20,                  # capture Yes/No variants
    "temperature": 0.0,              # deterministic
}

VLLM_COUNTER_METRICS = {
    "vllm:estimated_flops_per_gpu_total",
    "vllm:estimated_read_bytes_per_gpu_total",
    "vllm:estimated_write_bytes_per_gpu_total",
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens",
    "vllm:generation_tokens_total",
    "vllm:request_success",
    "vllm:request_success_total",
}

VLLM_GAUGE_METRICS = {
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:model_flops_utilization",
}

VLLM_HISTOGRAM_PREFIXES = (
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
)

MINUTES = 60  # seconds

# -----------------------------------------------------------------------------
# Modal infrastructure
# -----------------------------------------------------------------------------

app = modal.App("grep-for-meaning-scorer")

# Container image: CUDA 12.9 + vLLM
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install(
        "vllm==0.8.5",
        "transformers>=4.51.1,<5",
        "huggingface-hub>=0.24.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster model downloads on current huggingface-hub
    })
)

# Newer vLLM server image for Prometheus MFU counters. Kept separate from the
# working in-process scorer image so benchmark changes cannot destabilize the demo path.
vllm_metrics_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install(
        f"vllm=={VLLM_METRICS_VERSION}",
        "transformers>=4.51.1,<5",
        # vLLM 0.22.1 can resolve to FastAPI/Starlette versions that break
        # prometheus_fastapi_instrumentator route inspection on /v1/*.
        "fastapi>=0.115.0,<0.137",
        "prometheus-fastapi-instrumentator>=7.0.0,<8",
        "huggingface-hub>=0.24.0",
        "httpx>=0.27.0",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

# Persistent volumes for model weights (avoid re-downloading)
hf_cache_vol = modal.Volume.from_name("grep-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("grep-vllm-cache", create_if_missing=True)


# -----------------------------------------------------------------------------
# Scorer class: single-token Yes/No logprob scoring
# -----------------------------------------------------------------------------

@app.cls(
    image=vllm_image,
    gpu="H100",  # Single H100 per replica (data-parallel, not tensor-parallel)
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,  # keep warm for refinement loops
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Scale to N_REPLICAS for data parallelism
    min_containers=SCORER_MIN_CONTAINERS,
    max_containers=N_REPLICAS,
)
@modal.concurrent(max_inputs=256, target_inputs=128)  # high concurrency for batching
class Scorer:
    """
    vLLM-backed scorer for single-token Yes/No relevance scoring.

    Implements the ScorerClient contract (CONTRACTS.md §1):
    - score_batch: (chunk_text, predicate) → ScoreResult with P(Yes)/(P(Yes)+P(No))
    - warm: pre-prefill chunk prefixes for KV cache
    - health: replica status and cache info
    """

    model_name: str = MODEL_NAME

    @modal.enter()
    def start_engine(self):
        """Initialize vLLM engine on container start."""
        import vllm

        print(f"[Scorer] Starting vLLM engine with {self.model_name}")
        self._mfu_metrics_active = bool(VLLM_ENGINE_KWARGS.get("enable_mfu_metrics"))
        try:
            self.engine = vllm.LLM(
                model=self.model_name,
                revision=MODEL_REVISION,
                **VLLM_ENGINE_KWARGS,
            )
        except TypeError:
            if "enable_mfu_metrics" not in VLLM_ENGINE_KWARGS:
                raise
            fallback_kwargs = dict(VLLM_ENGINE_KWARGS)
            fallback_kwargs.pop("enable_mfu_metrics", None)
            print("[Scorer] vLLM Python API rejected enable_mfu_metrics; retrying without MFU metrics")
            self._mfu_metrics_active = False
            self.engine = vllm.LLM(
                model=self.model_name,
                revision=MODEL_REVISION,
                **fallback_kwargs,
            )

        # Build Yes/No token ID sets for logprob aggregation (PLAN §5 #1)
        tokenizer = self.engine.get_tokenizer()
        self._yes_ids = self._get_token_ids(tokenizer, ["Yes", "yes", " Yes", " yes", "YES"])
        self._no_ids = self._get_token_ids(tokenizer, ["No", "no", " No", " no", "NO"])

        # Warmup inference
        self.engine.generate(
            ["Is this thing on?"],
            vllm.SamplingParams(**DEFAULT_SAMPLING_KWARGS),
        )
        print(f"[Scorer] Engine ready. Yes tokens: {self._yes_ids}, No tokens: {self._no_ids}")

    def _get_token_ids(self, tokenizer, surface_forms: list[str]) -> set[int]:
        """Get all token IDs that could represent a surface form."""
        ids = set()
        for form in surface_forms:
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return ids

    @modal.method()
    def score_batch(
        self,
        items: list[dict],  # [{"chunk_id": str, "chunk_text": str, "predicate": str}]
        tier: int = 1,
    ) -> list[dict]:
        """
        Score a batch of (chunk, predicate) pairs.

        Returns ScoreResult dicts with:
        - chunk_id: echoed back
        - score: P(Yes)/(P(Yes)+P(No)) in [0,1]
        - p_yes, p_no: raw probabilities
        - tier: which model tier was used
        - from_cache: always False (cache is at backend level)
        - latency_ms: per-item latency estimate
        """
        import time
        import vllm

        if not items:
            return []

        start = time.perf_counter()

        # Build prompts: [instruction + chunk] then [predicate] (PLAN §5 #2)
        prompts = [
            self._build_prompt(item["chunk_text"], item["predicate"])
            for item in items
        ]

        # Run inference
        sampling_params = vllm.SamplingParams(**DEFAULT_SAMPLING_KWARGS)
        outputs = self.engine.generate(prompts, sampling_params)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_item_ms = elapsed_ms / len(items)

        # Extract scores from logprobs
        results = []
        for item, output in zip(items, outputs):
            p_yes, p_no = self._extract_yes_no_probs(output)
            score = p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.5

            results.append({
                "chunk_id": item["chunk_id"],
                "score": score,
                "p_yes": p_yes,
                "p_no": p_no,
                "tier": tier,
                "from_cache": False,
                "latency_ms": per_item_ms,
            })

        return results

    def _build_prompt(self, chunk_text: str, predicate: str) -> str:
        """
        Build the scoring prompt.

        Format: [instruction + chunk] (cached prefix) + [predicate] (changing suffix)
        The predicate is kept short (≤~40 tokens) so suffix-only re-prefill is fast.
        """
        # This matches the prompt template in inference/prompt.py
        return f"""You are a relevance scorer. Given a text chunk and a predicate, respond with ONLY "Yes" if the chunk is relevant to the predicate, or "No" if it is not.

Text chunk:
{chunk_text}

Predicate: {predicate}

Is this chunk relevant? Answer Yes or No:"""

    def _extract_yes_no_probs(self, output) -> tuple[float, float]:
        """
        Extract P(Yes) and P(No) from logprobs.

        Aggregates probability mass over all surface forms (Yes/yes/YES etc).
        Uses 1e-6 smoothing → 0.5 score when neither appears.
        """
        import math

        SMOOTHING = 1e-6
        p_yes = SMOOTHING
        p_no = SMOOTHING

        if output.outputs and output.outputs[0].logprobs:
            first_token_logprobs = output.outputs[0].logprobs[0]
            for token_id, logprob_obj in first_token_logprobs.items():
                prob = math.exp(logprob_obj.logprob)
                if token_id in self._yes_ids:
                    p_yes += prob
                elif token_id in self._no_ids:
                    p_no += prob

        return p_yes, p_no

    @modal.method()
    def warm(self, corpus_id: str, chunks: list[dict]) -> dict:
        """
        Pre-prefill chunk prefixes for KV cache warmth.

        This is a first-query optimization (PLAN §5 #4) - candidate-set scoping
        is the primary refine mechanism regardless of cache state.

        Returns PrefixState dict.
        """
        import vllm

        if not chunks:
            return {
                "corpus_id": corpus_id,
                "n_chunks": 0,
                "warmed": True,
                "model_id": self.model_name,
            }

        # Prefill-only: use the chunk prefix without a predicate
        # This warms the [instruction + chunk] KV that gets reused across predicates
        prompts = [
            self._build_warm_prefix(chunk["text"])
            for chunk in chunks
        ]

        # Run with max_tokens=0 to only do prefill (no generation)
        # Note: vLLM may not support max_tokens=0, so we use 1 and discard
        sampling_params = vllm.SamplingParams(max_tokens=1, temperature=0.0)
        self.engine.generate(prompts, sampling_params)

        return {
            "corpus_id": corpus_id,
            "n_chunks": len(chunks),
            "warmed": True,
            "model_id": self.model_name,
        }

    def _build_warm_prefix(self, chunk_text: str) -> str:
        """Build the cacheable prefix for warming (instruction + chunk only)."""
        return f"""You are a relevance scorer. Given a text chunk and a predicate, respond with ONLY "Yes" if the chunk is relevant to the predicate, or "No" if it is not.

Text chunk:
{chunk_text}

Predicate:"""

    @modal.method()
    def health(self) -> dict:
        """Return health and cache info for the replica."""
        return {
            "ready": True,
            "backend": "modal",
            "model_id": self.model_name,
            "engine_config": {
                "max_model_len": VLLM_ENGINE_KWARGS["max_model_len"],
                "kv_cache_dtype": VLLM_ENGINE_KWARGS["kv_cache_dtype"],
                "prefix_caching": VLLM_ENGINE_KWARGS["enable_prefix_caching"],
                "gpu_memory_utilization": VLLM_ENGINE_KWARGS["gpu_memory_utilization"],
                "enable_mfu_metrics": bool(getattr(self, "_mfu_metrics_active", False)),
            },
        }

    @modal.method()
    def model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name


# -----------------------------------------------------------------------------
# Web endpoint for HTTP access (alternative to direct Modal function calls)
# -----------------------------------------------------------------------------

@app.function(
    image=vllm_image,
    timeout=5 * MINUTES,
)
@modal.asgi_app()
def web_endpoint():
    """
    FastAPI web endpoint for the scorer.

    Endpoints:
    - POST /score: batch scoring
    - POST /warm: pre-prefill chunks
    - GET /health: replica status
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    api = FastAPI(title="Grep for Meaning Scorer")
    scorer = Scorer()

    class ScoreRequest(BaseModel):
        items: list[dict]  # [{"chunk_id", "chunk_text", "predicate"}]
        tier: int = 1

    class WarmRequest(BaseModel):
        corpus_id: str
        chunks: list[dict]  # [{"chunk_id", "text"}]

    @api.post("/score")
    async def score(req: ScoreRequest):
        try:
            results = scorer.score_batch.remote(req.items, tier=req.tier)
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.post("/warm")
    async def warm(req: WarmRequest):
        try:
            state = scorer.warm.remote(req.corpus_id, req.chunks)
            return state
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/health")
    async def health():
        try:
            return scorer.health.remote()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return api


# -----------------------------------------------------------------------------
# OpenAI-compatible vLLM server benchmark for /metrics and MFU
# -----------------------------------------------------------------------------

def _parse_vllm_prometheus_metrics(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, raw_value = line.partition(" ")
        base_name = name.split("{", 1)[0]
        if base_name.endswith("_bucket"):
            continue
        histogram_base = base_name.removesuffix("_sum").removesuffix("_count")
        if (
            base_name in VLLM_COUNTER_METRICS
            or base_name in VLLM_GAUGE_METRICS
            or histogram_base in VLLM_HISTOGRAM_PREFIXES
            or "mfu" in base_name.lower()
            or "flops_utilization" in base_name.lower()
        ):
            try:
                value = float(raw_value)
            except ValueError:
                continue
            _record_prometheus_sample(parsed, base_name, value)
    return parsed


def _record_prometheus_sample(parsed: dict[str, float], base_name: str, value: float) -> None:
    histogram_base = base_name.removesuffix("_sum").removesuffix("_count")
    if base_name in VLLM_COUNTER_METRICS or histogram_base in VLLM_HISTOGRAM_PREFIXES:
        parsed[base_name] = parsed.get(base_name, 0.0) + value
        return
    parsed[base_name] = max(parsed.get(base_name, value), value)


def _metric_delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    if key not in after:
        return None
    return after[key] - before.get(key, 0.0)


def _first_metric_delta(before: dict[str, float], after: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = _metric_delta(before, after, key)
        if value is not None:
            return value
    return None


def _histogram_avg_ms(before: dict[str, float], after: dict[str, float], prefix: str) -> float | None:
    sum_delta = _metric_delta(before, after, f"{prefix}_sum")
    count_delta = _metric_delta(before, after, f"{prefix}_count")
    if sum_delta is None or count_delta is None or count_delta <= 0:
        return None
    return (sum_delta / count_delta) * 1000.0


def _summarize_server_metrics(before: dict[str, float], after: dict[str, float], elapsed_s: float) -> dict[str, float | None]:
    elapsed_s = max(elapsed_s, 1e-9)
    prompt_tokens = _first_metric_delta(before, after, "vllm:prompt_tokens", "vllm:prompt_tokens_total")
    generation_tokens = _first_metric_delta(before, after, "vllm:generation_tokens", "vllm:generation_tokens_total")
    request_success = _first_metric_delta(before, after, "vllm:request_success", "vllm:request_success_total")
    request_count = request_success if request_success is not None and request_success > 0 else _first_metric_delta(
        before,
        after,
        "vllm:e2e_request_latency_seconds_count",
        "vllm:time_to_first_token_seconds_count",
    )
    flops = _metric_delta(before, after, "vllm:estimated_flops_per_gpu_total")
    read_bytes = _metric_delta(before, after, "vllm:estimated_read_bytes_per_gpu_total")
    write_bytes = _metric_delta(before, after, "vllm:estimated_write_bytes_per_gpu_total")

    return {
        "elapsed_s": elapsed_s,
        "request_success_delta": request_success,
        "request_count_delta": request_count,
        "prompt_tokens_delta": prompt_tokens,
        "generation_tokens_delta": generation_tokens,
        "requests_per_s": (request_count / elapsed_s) if request_count is not None else None,
        "prompt_tokens_per_s": (prompt_tokens / elapsed_s) if prompt_tokens is not None else None,
        "generation_tokens_per_s": (generation_tokens / elapsed_s) if generation_tokens is not None else None,
        "estimated_tflops_per_gpu": (flops / elapsed_s / 1e12) if flops is not None else None,
        "derived_mfu_bf16_peak": (flops / elapsed_s / H100_SXM_BF16_FLOPS_PER_GPU) if flops is not None else None,
        "derived_mfu_fp8_peak": (flops / elapsed_s / H100_SXM_FP8_FLOPS_PER_GPU) if flops is not None else None,
        "estimated_memory_gb_per_s": ((read_bytes or 0.0) + (write_bytes or 0.0)) / elapsed_s / 1e9
        if (read_bytes is not None or write_bytes is not None)
        else None,
        "reported_model_flops_utilization": after.get("vllm:model_flops_utilization"),
        "kv_cache_usage_perc": after.get("vllm:kv_cache_usage_perc", after.get("vllm:gpu_cache_usage_perc")),
        "num_requests_running": after.get("vllm:num_requests_running"),
        "num_requests_waiting": after.get("vllm:num_requests_waiting"),
        "server_e2e_latency_avg_ms": _histogram_avg_ms(before, after, "vllm:e2e_request_latency_seconds"),
        "server_ttft_avg_ms": _histogram_avg_ms(before, after, "vllm:time_to_first_token_seconds"),
        "server_queue_avg_ms": _histogram_avg_ms(before, after, "vllm:request_queue_time_seconds"),
        "server_prefill_avg_ms": _histogram_avg_ms(before, after, "vllm:request_prefill_time_seconds"),
        "server_inference_avg_ms": _histogram_avg_ms(before, after, "vllm:request_inference_time_seconds"),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _benchmark_prompt(i: int) -> str:
    return f"""You are a strict semantic filter. Answer only Yes or No.

Chunk:
The service batch {i} describes retry logic, GPU queue saturation, prefix-cache behavior, and latency measurement under load.

Question:
Does this chunk satisfy the predicate: GPU queue saturation and throughput metrics

Answer:"""


@app.function(
    image=vllm_metrics_image,
    gpu="H100",
    timeout=20 * MINUTES,
    scaledown_window=2 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def openai_server_benchmark(
    num_requests: int = 128,
    concurrency: int = 32,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
    max_num_batched_tokens: int = 8192,
    enable_mfu_metrics: bool = True,
    replica_label: str = "replica-0",
) -> dict:
    """Run vLLM's OpenAI-compatible server and collect latency/throughput/MFU."""
    import asyncio
    import subprocess
    import sys
    import threading
    import time

    import httpx

    port = 8000
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    api_url = f"{base_url}/v1"

    help_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"]
    help_proc = subprocess.run(help_cmd, capture_output=True, text=True, timeout=90, check=False)
    help_text = help_proc.stdout + help_proc.stderr
    mfu_flag_supported = "--enable-mfu-metrics" in help_text

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        MODEL_NAME,
        "--served-model-name",
        "tier1-filter",
        "--revision",
        MODEL_REVISION,
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--quantization",
        "awq_marlin",
    ]
    if enable_mfu_metrics and mfu_flag_supported:
        cmd.append("--enable-mfu-metrics")
    if "--disable-log-requests" in help_text:
        cmd.append("--disable-log-requests")

    logs: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _read_logs() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if len(logs) < 240:
                logs.append(line.rstrip())
            print(f"[vllm-server] {line}", end="")

    reader = threading.Thread(target=_read_logs, daemon=True)
    reader.start()

    async def _wait_ready(client: httpx.AsyncClient) -> None:
        deadline = time.time() + 12 * MINUTES
        last_error = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {proc.returncode}: {logs[-40:]}")
            try:
                response = await client.get(f"{api_url}/models")
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = str(exc)
            await asyncio.sleep(2)
        raise TimeoutError(f"vLLM server did not become ready: {last_error}")

    async def _scrape(client: httpx.AsyncClient) -> dict[str, float]:
        response = await client.get(f"{base_url}/metrics")
        response.raise_for_status()
        return _parse_vllm_prometheus_metrics(response.text)

    async def _post_completion(client: httpx.AsyncClient, idx: int) -> dict:
        payload = {
            "model": "tier1-filter",
            "prompt": _benchmark_prompt(idx),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 5,
        }
        started = time.perf_counter()
        response = await client.post(f"{api_url}/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        return {
            "latency_ms": latency_ms,
            "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
            "completion_tokens": float(usage.get("completion_tokens") or 0.0),
            "total_tokens": float(usage.get("total_tokens") or 0.0),
        }

    async def _run() -> dict:
        limits = httpx.Limits(max_connections=max(concurrency * 2, 8), max_keepalive_connections=max(concurrency, 4))
        async with httpx.AsyncClient(timeout=90.0, limits=limits) as client:
            await _wait_ready(client)
            await _post_completion(client, -1)  # warm the served model and metrics path
            before = await _scrape(client)
            semaphore = asyncio.Semaphore(max(1, concurrency))

            async def _bounded(idx: int) -> dict:
                async with semaphore:
                    return await _post_completion(client, idx)

            started = time.perf_counter()
            request_results = await asyncio.gather(*(_bounded(i) for i in range(num_requests)))
            elapsed_s = time.perf_counter() - started
            after = await _scrape(client)

        latencies = [float(item["latency_ms"]) for item in request_results]
        prompt_tokens = sum(float(item["prompt_tokens"]) for item in request_results)
        completion_tokens = sum(float(item["completion_tokens"]) for item in request_results)
        total_tokens = sum(float(item["total_tokens"]) for item in request_results)
        server_summary = _summarize_server_metrics(before, after, elapsed_s)
        return {
            "replica_label": replica_label,
            "server": "vllm OpenAI-compatible API",
            "model": MODEL_NAME,
            "vllm_version": VLLM_METRICS_VERSION,
            "command": cmd,
            "config": {
                "num_requests": num_requests,
                "concurrency": concurrency,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_num_batched_tokens": max_num_batched_tokens,
                "enable_mfu_metrics_requested": enable_mfu_metrics,
                "enable_mfu_metrics_supported": mfu_flag_supported,
                "enable_mfu_metrics_active": enable_mfu_metrics and mfu_flag_supported,
            },
            "client_summary": {
                "elapsed_s": elapsed_s,
                "requests_per_s": num_requests / elapsed_s,
                "prompt_tokens_per_s": prompt_tokens / elapsed_s,
                "completion_tokens_per_s": completion_tokens / elapsed_s,
                "total_tokens_per_s": total_tokens / elapsed_s,
                "latency_ms_p50": _percentile(latencies, 0.50),
                "latency_ms_p95": _percentile(latencies, 0.95),
                "latency_ms_p99": _percentile(latencies, 0.99),
                "latency_ms_max": max(latencies) if latencies else 0.0,
            },
            "server_summary": server_summary,
            "metrics_before": before,
            "metrics_after": after,
            "mfu_metrics": {
                key: value
                for key, value in after.items()
                if "mfu" in key.lower() or "flops" in key.lower()
            },
            "logs_tail": logs[-80:],
        }

    try:
        return asyncio.run(_run())
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)


@app.local_entrypoint()
def benchmark_openai_server(
    num_replicas: int = 1,
    num_requests: int = 128,
    concurrency: int = 32,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
    max_num_batched_tokens: int = 8192,
):
    """Benchmark vLLM's OpenAI server on Modal and write a Phase 04 artifact."""
    import asyncio
    import json
    import time
    from pathlib import Path

    async def _run_all() -> list[dict]:
        tasks = [
            openai_server_benchmark.remote.aio(
                num_requests=num_requests,
                concurrency=concurrency,
                gpu_memory_utilization=gpu_memory_utilization,
                max_num_batched_tokens=max_num_batched_tokens,
                replica_label=f"replica-{idx}",
            )
            for idx in range(num_replicas)
        ]
        return list(await asyncio.gather(*tasks))

    started = time.perf_counter()
    results = asyncio.run(_run_all())
    elapsed_s = time.perf_counter() - started
    client_summaries = [result["client_summary"] for result in results]
    server_summaries = [result["server_summary"] for result in results]
    payload = {
        "run_id": f"modal-openai-server-{int(time.time())}",
        "replicas": num_replicas,
        "elapsed_s": elapsed_s,
        "model": MODEL_NAME,
        "vllm_version": VLLM_METRICS_VERSION,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_batched_tokens": max_num_batched_tokens,
        "aggregate_client": {
            "requests_per_s": sum(item["requests_per_s"] for item in client_summaries),
            "prompt_tokens_per_s": sum(item["prompt_tokens_per_s"] for item in client_summaries),
            "completion_tokens_per_s": sum(item["completion_tokens_per_s"] for item in client_summaries),
            "total_tokens_per_s": sum(item["total_tokens_per_s"] for item in client_summaries),
            "latency_ms_p50_mean": sum(item["latency_ms_p50"] for item in client_summaries) / len(client_summaries),
            "latency_ms_p95_max": max(item["latency_ms_p95"] for item in client_summaries),
            "latency_ms_p99_max": max(item["latency_ms_p99"] for item in client_summaries),
        },
        "aggregate_server": {
            "estimated_tflops_per_gpu_mean": _mean_present(server_summaries, "estimated_tflops_per_gpu"),
            "derived_mfu_bf16_peak_mean": _mean_present(server_summaries, "derived_mfu_bf16_peak"),
            "derived_mfu_fp8_peak_mean": _mean_present(server_summaries, "derived_mfu_fp8_peak"),
            "prompt_tokens_per_s": _sum_present(server_summaries, "prompt_tokens_per_s"),
            "generation_tokens_per_s": _sum_present(server_summaries, "generation_tokens_per_s"),
            "requests_per_s": _sum_present(server_summaries, "requests_per_s"),
            "kv_cache_usage_perc_max": _max_present(server_summaries, "kv_cache_usage_perc"),
            "num_requests_running_max": _max_present(server_summaries, "num_requests_running"),
            "num_requests_waiting_max": _max_present(server_summaries, "num_requests_waiting"),
            "server_e2e_latency_avg_ms_mean": _mean_present(server_summaries, "server_e2e_latency_avg_ms"),
            "server_ttft_avg_ms_mean": _mean_present(server_summaries, "server_ttft_avg_ms"),
            "server_queue_avg_ms_mean": _mean_present(server_summaries, "server_queue_avg_ms"),
            "server_prefill_avg_ms_mean": _mean_present(server_summaries, "server_prefill_avg_ms"),
            "server_inference_avg_ms_mean": _mean_present(server_summaries, "server_inference_avg_ms"),
        },
        "replica_results": results,
    }
    artifact = Path("eval/artifacts/phase04_modal_openai_server_benchmark.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote {artifact}")


def _mean_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _sum_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) if values else None


def _max_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


# -----------------------------------------------------------------------------
# Local entrypoint for testing
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def test():
    """Test the scorer with a simple example."""
    print("Testing Grep for Meaning Scorer...")

    scorer = Scorer()

    # Health check
    health = scorer.health.remote()
    print(f"Health: {health}")

    # Test scoring
    test_items = [
        {
            "chunk_id": "test_001",
            "chunk_text": "The retry logic uses exponential backoff with jitter to avoid thundering herd.",
            "predicate": "retry without backoff",
        },
        {
            "chunk_id": "test_002",
            "chunk_text": "This function implements a simple for loop over the items list.",
            "predicate": "retry without backoff",
        },
    ]

    results = scorer.score_batch.remote(test_items)
    print(f"\nScoring results:")
    for r in results:
        print(f"  {r['chunk_id']}: score={r['score']:.3f} (p_yes={r['p_yes']:.4f}, p_no={r['p_no']:.4f})")

    print("\n✓ Test complete")
