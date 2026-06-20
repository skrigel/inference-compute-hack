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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


GPU_MEMORY_UTILIZATION = _env_float("GPU_MEMORY_UTILIZATION", 0.92)
ENABLE_MFU_METRICS = _env_flag("ENABLE_MFU_METRICS", True)
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")

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
