from __future__ import annotations

import asyncio
import itertools
import math
import os
import time
from collections.abc import Iterable
from typing import Any

import httpx

from data.schema import Chunk
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient


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
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._replicas = [replica.rstrip("/") for replica in replicas if replica.strip()]
        if not self._replicas:
            raise ValueError("VLLMScorer requires at least one replica URL")
        self._model_id = model_id
        self._replica_counter = itertools.count()
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._warmed_corpora: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> "VLLMScorer":
        raw_replicas = os.environ.get("VLLM_REPLICAS", "http://127.0.0.1:8001/v1")
        return cls(
            raw_replicas.split(","),
            model_id=os.environ.get("VLLM_MODEL_ID", "tier1-filter"),
            timeout_s=float(os.environ.get("VLLM_TIMEOUT_S", "30")),
            max_concurrency=int(os.environ.get("VLLM_MAX_CONCURRENCY", "128")),
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

    async def _score_one(self, item: ScoreRequest, *, tier: int) -> ScoreResult:
        replica = self._next_replica()
        started = time.perf_counter()
        payload = {
            "model": self._model_id,
            "prompt": build_prompt(item.chunk_text, item.predicate),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 20,
        }
        async with self._semaphore:
            response = await self._client.post(f"{replica}/completions", json=payload)
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
        wanted = {
            "vllm:gpu_cache_usage_perc",
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:model_flops_utilization",
        }
        parsed: dict[str, float] = {}
        for line in response.text.splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, raw_value = line.partition(" ")
            base_name = name.split("{", 1)[0]
            if base_name in wanted or "mfu" in base_name.lower():
                try:
                    parsed[base_name] = float(raw_value)
                except ValueError:
                    continue
        return parsed


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
