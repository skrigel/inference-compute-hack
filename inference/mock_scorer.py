from __future__ import annotations

import hashlib
import math
import re
import time

from data.schema import Chunk
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient


TOKEN_RE = re.compile(r"[a-z0-9_]+")


class MockScorer(ScorerClient):
    """Deterministic, GPU-free scorer for local dev and the demo fallback.

    Scores via demo-tuned keyword matching (concept-substring + token overlap +
    stable jitter) — it is intentionally NOT a semantic proxy for production
    relevance; real scoring comes from the vLLM single-token scorer (Phase 04).
    The concept bonuses are calibrated so the pinned cut-line corpus produces
    reproducible scripted-demo results, not to flatter the numbers.
    """

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

    # Concept lexicons matched as SUBSTRINGS on the raw lowercased text, so
    # "retries"/"networking" trigger the same concept as "retry"/"network"
    # without brittle exact-token matching. This is what makes the scripted demo
    # predicates ("retry without backoff", "in the networking layer", "IR sense")
    # produce stable, known results — the Phase 03 cut-line requirement.
    _RETRY_PRED = ("retry", "retries", "retrying", "backoff")
    _RETRY_DOC = ("retry", "retries", "retrying")
    _NET_PRED = ("network", "http", "socket", "tcp", "connection")
    _NET_DOC = ("network", "http", "socket", "tcp", "connection")
    _IR_PRED = ("retrieval", "ranking", "search", " ir ", " ir.")
    _IR_DOC = ("retrieval", "ranking", "search", "index")

    @staticmethod
    def _mentions(text: str, needles: tuple[str, ...]) -> bool:
        return any(needle in text for needle in needles)

    def _score(self, chunk_text: str, predicate: str) -> float:
        chunk = chunk_text.lower()
        pred = predicate.lower()
        chunk_tokens = set(TOKEN_RE.findall(chunk))
        predicate_tokens = set(TOKEN_RE.findall(pred))
        if not predicate_tokens:
            return 0.5

        overlap = len(chunk_tokens & predicate_tokens) / len(predicate_tokens)
        keyword_bonus = self._keyword_bonus(chunk, pred)
        jitter = self._stable_jitter(chunk_text, predicate)
        raw = -1.0 + (2.2 * overlap) + keyword_bonus + jitter
        return 1.0 / (1.0 + math.exp(-raw))

    def _keyword_bonus(self, chunk: str, pred: str) -> float:
        bonus = 0.0
        if self._mentions(pred, self._RETRY_PRED) and self._mentions(chunk, self._RETRY_DOC):
            bonus += 0.9
            # retry WITHOUT backoff is the headline bug pattern — boost it.
            if "backoff" in pred and "backoff" not in chunk:
                bonus += 0.4
        if self._mentions(pred, self._NET_PRED) and self._mentions(chunk, self._NET_DOC):
            bonus += 1.2
        if self._mentions(pred, self._IR_PRED) and self._mentions(chunk, self._IR_DOC):
            bonus += 0.8
        return bonus

    def _stable_jitter(self, chunk_text: str, predicate: str) -> float:
        digest = hashlib.sha1(f"{chunk_text}|{predicate}".encode()).hexdigest()
        bucket = int(digest[:6], 16) / 0xFFFFFF
        return (bucket - 0.5) * 0.18
