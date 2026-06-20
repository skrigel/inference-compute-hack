from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from data.schema import Chunk
from inference.scorer import ScoreResult


HIST_BINS = 20


class RefineOp(str, Enum):
    require = "require"
    exclude = "exclude"
    include = "include"
    refocus = "refocus"
    brush = "brush"


class IngestRequest(BaseModel):
    corpus_id: str


class QueryRequest(BaseModel):
    predicate: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class FacetBucket(BaseModel):
    key: str
    relevant: int
    total: int


class HistogramBin(BaseModel):
    lo: float
    hi: float
    count: int


class ChunkWireMeta(BaseModel):
    type: Literal["paper", "code"]
    title: str
    category: str | None
    year: int | None
    path: str | None
    lang: str | None
    repo: str | None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkWireMeta":
        return cls(
            type=chunk.type,
            title=chunk.title,
            category=chunk.meta.category,
            year=chunk.meta.year,
            path=chunk.meta.path,
            lang=chunk.meta.lang,
            repo=chunk.meta.repo,
        )


class ResultEvent(BaseModel):
    type: Literal["result"] = "result"
    chunk_id: str
    score: float
    meta: ChunkWireMeta
    rank: int
    rationale: str | None = None

    @classmethod
    def from_score(cls, score: ScoreResult, chunk: Chunk, rank: int) -> "ResultEvent":
        return cls(
            chunk_id=score.chunk_id,
            score=score.score,
            meta=ChunkWireMeta.from_chunk(chunk),
            rank=rank,
            rationale=None,
        )


class AggregateEvent(BaseModel):
    type: Literal["aggregate"] = "aggregate"
    scanned: int
    matched: int
    histogram: list[HistogramBin]
    facets: dict[str, list[FacetBucket]]
    threshold: float
    eta_ms: int


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    scanned: int
    matched: int
    elapsed_ms: int
    warm: bool
    summary: str


class IngestResponse(BaseModel):
    corpus_id: str
    n_chunks: int
    facets: dict[str, list[FacetBucket]]
    warm_eta_s: float
