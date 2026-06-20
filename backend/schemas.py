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
    documents: list["FreshDocument"] = Field(default_factory=list)


class FreshDocument(BaseModel):
    title: str
    text: str
    type: Literal["paper", "code"] = "code"
    category: str | None = None
    year: int | None = None
    path: str | None = None
    lang: str | None = None
    repo: str | None = None


class QueryRequest(BaseModel):
    predicate: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ClickRequest(BaseModel):
    chunk_id: str
    sign: Literal["+", "-"]


class BrushRequest(BaseModel):
    lo: float = Field(ge=0.0, le=1.0)
    hi: float = Field(ge=0.0, le=1.0)


class RefineRequest(BaseModel):
    utterance: str | None = None
    click: ClickRequest | None = None
    brush: BrushRequest | None = None


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


class Chip(BaseModel):
    clause_id: str
    op: RefineOp
    text: str
    label: str
    removable: bool
    confidence: float


class ChipEvent(BaseModel):
    type: Literal["chip"] = "chip"
    operation: RefineOp
    chip: Chip
    refine_ms: int
    latency_kind: Literal["cold", "warm", "cached"]


class DiffEvent(BaseModel):
    type: Literal["diff"] = "diff"
    added: list[ResultEvent]
    removed: list[str]
    rescored: list[dict[str, float | str]]
    refine_ms: int


class IngestResponse(BaseModel):
    corpus_id: str
    n_chunks: int
    facets: dict[str, list[FacetBucket]]
    warm_eta_s: float


class ResultsResponse(BaseModel):
    """Ranked slice computed from the score cache — zero inference."""

    items: list[ResultEvent]
    threshold: float
    top_k: int | None = None
    total_matched: int


class ClauseDeleteResponse(BaseModel):
    removed: bool
    refine_ms: int
