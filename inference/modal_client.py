"""
Modal-backed ScorerClient implementation.

This client connects to the Modal-deployed vLLM scorer replicas.
It implements the ScorerClient interface (CONTRACTS.md §1) so the backend
can swap between mock/vllm/modal transparently via SCORER_BACKEND.

Usage:
    export SCORER_BACKEND=modal
    # Backend will use Modal replicas for scoring
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from data.schema import Chunk
from inference.scorer import PrefixState, ScorerClient, ScoreRequest, ScoreResult

if TYPE_CHECKING:
    pass


class ModalScorer(ScorerClient):
    """
    ScorerClient that delegates to Modal-deployed vLLM replicas.

    The Modal app handles data parallelism across 6 H100 GPUs internally.
    This client just calls the Modal function and maps to/from our types.
    """

    def __init__(self):
        # Lazy import to avoid requiring modal when not used
        try:
            import modal
        except ImportError:
            raise ImportError(
                "Modal is not installed. Install with: pip install modal"
            )

        # Get reference to the deployed Scorer class
        # This connects to the running Modal app
        self._scorer_cls = modal.Cls.from_name("grep-for-meaning-scorer", "Scorer")
        self._scorer = self._scorer_cls()
        self._model_id_cache: str | None = None

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        """
        Pre-prefill chunk prefixes for KV cache warmth.

        Delegates to Modal's warm method which runs on the GPU replicas.
        """
        # Convert Chunk objects to dicts for serialization
        chunk_dicts = [
            {"chunk_id": c.chunk_id, "text": c.text}
            for c in chunks
        ]

        # Call Modal function (runs on GPU)
        result = self._scorer.warm.remote(corpus_id, chunk_dicts)

        return PrefixState(
            corpus_id=result["corpus_id"],
            n_chunks=result["n_chunks"],
            warmed=result["warmed"],
            model_id=result["model_id"],
        )

    async def score_batch(
        self, items: list[ScoreRequest], *, tier: int = 1
    ) -> list[ScoreResult]:
        """
        Score a batch of (chunk, predicate) pairs.

        Delegates to Modal which handles batching and parallelism across replicas.
        """
        if not items:
            return []

        # Convert to dicts for serialization
        item_dicts = [
            {
                "chunk_id": item.chunk_id,
                "chunk_text": item.chunk_text,
                "predicate": item.predicate,
            }
            for item in items
        ]

        # Call Modal function (runs on GPU)
        result_dicts = self._scorer.score_batch.remote(item_dicts, tier=tier)

        # Convert back to ScoreResult objects
        return [
            ScoreResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                p_yes=r["p_yes"],
                p_no=r["p_no"],
                tier=r["tier"],
                from_cache=r["from_cache"],
                latency_ms=r["latency_ms"],
            )
            for r in result_dicts
        ]

    async def health(self) -> dict:
        """Return health info from Modal replicas."""
        result = self._scorer.health.remote()
        return {
            "ready": result.get("ready", False),
            "backend": "modal",
            "scorer": "modal",
            "replicas": [result],  # Modal handles replica management internally
            "warmed_corpora": [],  # Would need to track this at backend level
        }

    def model_id(self) -> str:
        """Return the model identifier."""
        if self._model_id_cache is None:
            self._model_id_cache = self._scorer.model_id.remote()
        return self._model_id_cache


class ModalScorerAsync(ScorerClient):
    """
    Async-native version using Modal's async API for better concurrency.

    Use this when you want to fire off multiple score_batch calls concurrently
    and gather results (e.g., warming multiple chunk batches in parallel).
    """

    def __init__(self):
        try:
            import modal
        except ImportError:
            raise ImportError(
                "Modal is not installed. Install with: pip install modal"
            )

        self._scorer_cls = modal.Cls.from_name("grep-for-meaning-scorer", "Scorer")
        self._scorer = self._scorer_cls()
        self._model_id_cache: str | None = None

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        """Pre-prefill chunk prefixes using Modal's async API."""
        chunk_dicts = [
            {"chunk_id": c.chunk_id, "text": c.text}
            for c in chunks
        ]

        # Use .aio for true async
        result = await self._scorer.warm.remote.aio(corpus_id, chunk_dicts)

        return PrefixState(
            corpus_id=result["corpus_id"],
            n_chunks=result["n_chunks"],
            warmed=result["warmed"],
            model_id=result["model_id"],
        )

    async def score_batch(
        self, items: list[ScoreRequest], *, tier: int = 1
    ) -> list[ScoreResult]:
        """Score batch using Modal's async API."""
        if not items:
            return []

        item_dicts = [
            {
                "chunk_id": item.chunk_id,
                "chunk_text": item.chunk_text,
                "predicate": item.predicate,
            }
            for item in items
        ]

        # Use .aio for true async
        result_dicts = await self._scorer.score_batch.remote.aio(item_dicts, tier=tier)

        return [
            ScoreResult(
                chunk_id=r["chunk_id"],
                score=r["score"],
                p_yes=r["p_yes"],
                p_no=r["p_no"],
                tier=r["tier"],
                from_cache=r["from_cache"],
                latency_ms=r["latency_ms"],
            )
            for r in result_dicts
        ]

    async def health(self) -> dict:
        """Return health info from Modal replicas."""
        result = await self._scorer.health.remote.aio()
        return {
            "ready": result.get("ready", False),
            "backend": "modal",
            "scorer": "modal",
            "replicas": [result],
            "warmed_corpora": [],
        }

    def model_id(self) -> str:
        """Return the model identifier."""
        if self._model_id_cache is None:
            # This is sync because model_id is typically called once at init
            self._model_id_cache = self._scorer.model_id.remote()
        return self._model_id_cache
