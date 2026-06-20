from __future__ import annotations

import hashlib
import math
import re
import time

from data.schema import Chunk
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient


TOKEN_RE = re.compile(r"[a-z0-9_]+")


class MockScorer(ScorerClient):
    """Deterministic local scorer with a stable semantic-ish score shape."""

    def __init__(self, model_name: str = "mock-semantic-filter-v0") -> None:
        self._model_name = model_name
        self._warmed_corpora: set[str] = set()

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        self._warmed_corpora.add(corpus_id)
        return PrefixState(
            corpus_id=corpus_id,
            n_chunks=len(chunks),
            warmed=True,
            model_id=self._model_name,
        )

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        started = time.perf_counter()
        results: list[ScoreResult] = []
        for item in items:
            score = self._score(item.chunk_text, item.predicate)
            p_yes = max(score, 1e-6)
            p_no = max(1.0 - score, 1e-6)
            norm = p_yes + p_no
            results.append(
                ScoreResult(
                    chunk_id=item.chunk_id,
                    score=p_yes / norm,
                    p_yes=p_yes / norm,
                    p_no=p_no / norm,
                    tier=tier,
                    from_cache=False,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        return results

    async def health(self) -> dict:
        return {
            "ready": True,
            "backend": "mock",
            "model_id": self._model_name,
            "warmed_corpora": sorted(self._warmed_corpora),
        }

    def model_id(self) -> str:
        return self._model_name

    def _score(self, chunk_text: str, predicate: str) -> float:
        chunk_tokens = set(TOKEN_RE.findall(chunk_text.lower()))
        predicate_tokens = set(TOKEN_RE.findall(predicate.lower()))
        if not predicate_tokens:
            return 0.5

        overlap = len(chunk_tokens & predicate_tokens) / len(predicate_tokens)
        keyword_bonus = self._keyword_bonus(chunk_tokens, predicate_tokens)
        jitter = self._stable_jitter(chunk_text, predicate)
        raw = -1.1 + (3.0 * overlap) + keyword_bonus + jitter
        return 1.0 / (1.0 + math.exp(-raw))

    def _keyword_bonus(self, chunk_tokens: set[str], predicate_tokens: set[str]) -> float:
        bonus = 0.0
        if {"retry", "backoff"} <= predicate_tokens and "retry" in chunk_tokens:
            bonus += 0.65
            if "backoff" not in chunk_tokens:
                bonus += 0.35
        if "networking" in predicate_tokens and {"network", "networking", "http"} & chunk_tokens:
            bonus += 0.45
        if {"retrieval", "ir"} & predicate_tokens and {"retrieval", "ranking", "search"} & chunk_tokens:
            bonus += 0.35
        return bonus

    def _stable_jitter(self, chunk_text: str, predicate: str) -> float:
        digest = hashlib.sha1(f"{chunk_text}|{predicate}".encode()).hexdigest()
        bucket = int(digest[:6], 16) / 0xFFFFFF
        return (bucket - 0.5) * 0.18
