from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from data.schema import Chunk


@dataclass(frozen=True)
class ScoreRequest:
    chunk_id: str
    chunk_text: str
    predicate: str


@dataclass(frozen=True)
class ScoreResult:
    chunk_id: str
    score: float
    p_yes: float
    p_no: float
    tier: int = 1
    from_cache: bool = False
    latency_ms: float = 0.0


@dataclass(frozen=True)
class PrefixState:
    corpus_id: str
    n_chunks: int
    warmed: bool
    model_id: str


class ScorerClient(ABC):
    @abstractmethod
    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        raise NotImplementedError

    @abstractmethod
    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError
