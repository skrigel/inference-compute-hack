from __future__ import annotations

import asyncio
import itertools
import math
import os
import time
import zlib
from collections.abc import Iterable
from typing import Any

import httpx

from data.schema import Chunk
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient

H100_SXM_BF16_FLOPS_PER_GPU = 989.5e12
H100_SXM_FP8_FLOPS_PER_GPU = 1979.0e12

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


class VLLMScoringError(RuntimeError):
    """Raised when a vLLM response cannot be mapped to a Yes/No score."""


class VLLMScorer(ScorerClient):
    """OpenAI-compatible vLLM scorer for single-token semantic filtering."""

    def __init__(
        self,
        replicas: Iterable[str],
        *,
        model_id: str = "tier1-filter",
        timeout_s: float = 30.0,
        max_concurrency: int = 128,
        priority_reserved: int = 0,
        routing_mode: str = "round_robin",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._replicas = [replica.rstrip("/") for replica in replicas if replica.strip()]
        if not self._replicas:
            raise ValueError("VLLMScorer requires at least one replica URL")
        self._model_id = model_id
        self._replica_counter = itertools.count()
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)
        self._max_concurrency = max(1, max_concurrency)
        self._priority_reserved = max(0, min(priority_reserved, self._max_concurrency - 1))
        bulk_capacity = max(1, self._max_concurrency - self._priority_reserved)
        self._global_semaphore = asyncio.Semaphore(self._max_concurrency)
        self._bulk_semaphore = asyncio.Semaphore(bulk_capacity)
        normalized_routing = routing_mode.strip().lower()
        if normalized_routing not in {"round_robin", "chunk_sticky"}:
            raise ValueError("routing_mode must be 'round_robin' or 'chunk_sticky'")
        self._routing_mode = normalized_routing
        self._warmed_corpora: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> "VLLMScorer":
        raw_replicas = os.environ.get("VLLM_REPLICAS", "http://127.0.0.1:8001/v1")
        return cls(
            raw_replicas.split(","),
            model_id=os.environ.get("VLLM_MODEL_ID", "tier1-filter"),
            timeout_s=float(os.environ.get("VLLM_TIMEOUT_S", "30")),
            max_concurrency=int(os.environ.get("VLLM_MAX_CONCURRENCY", "128")),
            priority_reserved=int(os.environ.get("VLLM_PRIORITY_RESERVED", "16")),
            routing_mode=os.environ.get("VLLM_ROUTING_MODE", "chunk_sticky"),
        )

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        if chunks:
            neutral_predicate = "semantic relevance warmup"
            await self.score_batch(
                [
                    ScoreRequest(chunk.chunk_id, chunk.text, neutral_predicate)
                    for chunk in chunks
                ]
            )
        self._warmed_corpora[corpus_id] = len(chunks)
        return PrefixState(
            corpus_id=corpus_id,
            n_chunks=len(chunks),
            warmed=True,
            model_id=self._model_id,
        )

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        if not items:
            return []
        tasks = [self._score_one(item, tier=tier) for item in items]
        return list(await asyncio.gather(*tasks))

    async def health(self) -> dict:
        replicas = []
        for replica in self._replicas:
            ready = False
            detail: dict[str, Any] = {}
            try:
                response = await self._client.get(f"{replica}/models")
                ready = response.status_code < 500
                detail = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            except Exception as exc:
                detail = {"error": str(exc)}
            replicas.append({"url": replica, "ready": ready, "detail": detail})
        return {
            "ready": any(replica["ready"] for replica in replicas),
            "backend": "vllm",
            "model_id": self._model_id,
            "routing_mode": self._routing_mode,
            "max_concurrency": self._max_concurrency,
            "priority_reserved": self._priority_reserved,
            "replicas": replicas,
            "warmed_corpora": dict(self._warmed_corpora),
        }

    async def collect_metrics(self) -> dict:
        metrics = {}
        for replica in self._replicas:
            metrics[replica] = await self._collect_replica_metrics(replica)
        return metrics

    def model_id(self) -> str:
        return self._model_id

    def _next_replica(self) -> str:
        return self._replicas[next(self._replica_counter) % len(self._replicas)]

    def _route_replica(self, item: ScoreRequest) -> str:
        if self._routing_mode == "chunk_sticky":
            idx = zlib.crc32(item.chunk_id.encode("utf-8")) % len(self._replicas)
            return self._replicas[idx]
        return self._next_replica()

    async def _score_one(self, item: ScoreRequest, *, tier: int) -> ScoreResult:
        replica = self._route_replica(item)
        started = time.perf_counter()
        payload = {
            "model": self._model_id,
            "prompt": build_prompt(item.chunk_text, item.predicate),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 20,
        }
        bulk_acquired = False
        if tier > 0:
            await self._bulk_semaphore.acquire()
            bulk_acquired = True
        try:
            async with self._global_semaphore:
                response = await self._client.post(f"{replica}/completions", json=payload)
        finally:
            if bulk_acquired:
                self._bulk_semaphore.release()
        if response.status_code >= 400:
            raise VLLMScoringError(f"vLLM request failed on {replica}: HTTP {response.status_code} {response.text[:300]}")
        p_yes, p_no = yes_no_probabilities(response.json())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ScoreResult(
            chunk_id=item.chunk_id,
            score=p_yes / (p_yes + p_no),
            p_yes=p_yes,
            p_no=p_no,
            tier=tier,
            from_cache=False,
            latency_ms=elapsed_ms,
        )

    async def _collect_replica_metrics(self, replica: str) -> dict[str, float]:
        metrics_url = f"{replica.removesuffix('/v1')}/metrics"
        response = await self._client.get(metrics_url)
        if response.status_code >= 400:
            return {"error_status": float(response.status_code)}
        return parse_vllm_prometheus_metrics(response.text)


def parse_vllm_prometheus_metrics(text: str) -> dict[str, float]:
    """Extract the vLLM metrics needed for latency, throughput, queues, and MFU."""
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, raw_value = line.partition(" ")
        base_name = name.split("{", 1)[0]
        if base_name.endswith("_bucket"):
            continue
        if (
            base_name in VLLM_COUNTER_METRICS
            or base_name in VLLM_GAUGE_METRICS
            or (
                base_name.endswith(("_sum", "_count"))
                and base_name.removesuffix("_sum").removesuffix("_count") in VLLM_HISTOGRAM_PREFIXES
            )
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


def summarize_vllm_metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    elapsed_s: float,
    *,
    peak_flops_per_gpu: float = H100_SXM_BF16_FLOPS_PER_GPU,
) -> dict[str, float | None]:
    """Summarize counter deltas from two vLLM /metrics scrapes."""
    elapsed_s = max(elapsed_s, 1e-9)
    prompt_tokens = _first_delta(before, after, "vllm:prompt_tokens", "vllm:prompt_tokens_total")
    generation_tokens = _first_delta(before, after, "vllm:generation_tokens", "vllm:generation_tokens_total")
    request_success = _first_delta(before, after, "vllm:request_success", "vllm:request_success_total")
    request_count = request_success if request_success is not None and request_success > 0 else _first_delta(
        before,
        after,
        "vllm:e2e_request_latency_seconds_count",
        "vllm:time_to_first_token_seconds_count",
    )
    flops = _delta(before, after, "vllm:estimated_flops_per_gpu_total")
    direct_mfu = after.get("vllm:model_flops_utilization")

    tflops_per_gpu = (flops / elapsed_s / 1e12) if flops is not None else None
    derived_mfu = (flops / elapsed_s / peak_flops_per_gpu) if flops is not None else None

    return {
        "elapsed_s": elapsed_s,
        "request_success_delta": request_success,
        "request_count_delta": request_count,
        "prompt_tokens_delta": prompt_tokens,
        "generation_tokens_delta": generation_tokens,
        "requests_per_s": (request_count / elapsed_s) if request_count is not None else None,
        "prompt_tokens_per_s": (prompt_tokens / elapsed_s) if prompt_tokens is not None else None,
        "generation_tokens_per_s": (generation_tokens / elapsed_s) if generation_tokens is not None else None,
        "estimated_tflops_per_gpu": tflops_per_gpu,
        "derived_mfu_bf16_peak": derived_mfu,
        "reported_model_flops_utilization": direct_mfu,
        "server_e2e_latency_avg_ms": _histogram_avg_ms(before, after, "vllm:e2e_request_latency_seconds"),
        "server_ttft_avg_ms": _histogram_avg_ms(before, after, "vllm:time_to_first_token_seconds"),
        "server_queue_avg_ms": _histogram_avg_ms(before, after, "vllm:request_queue_time_seconds"),
        "server_prefill_avg_ms": _histogram_avg_ms(before, after, "vllm:request_prefill_time_seconds"),
        "server_inference_avg_ms": _histogram_avg_ms(before, after, "vllm:request_inference_time_seconds"),
    }


def _delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    if key not in after:
        return None
    return after[key] - before.get(key, 0.0)


def _first_delta(before: dict[str, float], after: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = _delta(before, after, key)
        if value is not None:
            return value
    return None


def _histogram_avg_ms(before: dict[str, float], after: dict[str, float], prefix: str) -> float | None:
    sum_delta = _delta(before, after, f"{prefix}_sum")
    count_delta = _delta(before, after, f"{prefix}_count")
    if sum_delta is None or count_delta is None or count_delta <= 0:
        return None
    return (sum_delta / count_delta) * 1000.0


def build_prompt(chunk_text: str, predicate: str) -> str:
    return (
        "You are a strict semantic filter. Answer only Yes or No.\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        "Question:\n"
        f"Does this chunk satisfy the predicate: {predicate}\n\n"
        "Answer:"
    )


def yes_no_probabilities(payload: dict[str, Any]) -> tuple[float, float]:
    top_logprobs = _extract_top_logprobs(payload)
    p_yes = 0.0
    p_no = 0.0
    for token, logprob in top_logprobs.items():
        normalized = _normalize_token(token)
        if normalized == "yes":
            p_yes += math.exp(logprob)
        elif normalized == "no":
            p_no += math.exp(logprob)
    if p_yes == 0.0 or p_no == 0.0:
        raise VLLMScoringError("vLLM response did not include both Yes/No logprobs")
    return p_yes, p_no


def _extract_top_logprobs(payload: dict[str, Any]) -> dict[str, float]:
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMScoringError("vLLM response missing choices") from exc

    completion_logprobs = choice.get("logprobs") or {}
    top_logprobs = completion_logprobs.get("top_logprobs")
    if top_logprobs:
        return dict(top_logprobs[0])

    chat_logprobs = completion_logprobs.get("content")
    if chat_logprobs:
        top = chat_logprobs[0].get("top_logprobs") or []
        return {item["token"]: item["logprob"] for item in top}

    raise VLLMScoringError("vLLM response missing top logprobs")


def _normalize_token(token: str) -> str:
    return token.strip().lstrip("Ġ▁").strip().lower()
