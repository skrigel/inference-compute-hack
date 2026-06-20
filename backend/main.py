from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from inference.config import make_scorer

from backend.cache import ScoreCache
from backend.schemas import (
    AggregateEvent,
    DoneEvent,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    ResultEvent,
    ResultsResponse,
)
from backend.state import BackendState, facet_summary
from backend.streaming import query_stream

app = FastAPI(title="Inference Compute Hack Backend", version="0.1.0")
state = BackendState()
cache = ScoreCache()
scorer = make_scorer()

# Each query mints a fresh clause id (q1, q2, …) so concurrent queries never
# evict each other's cached column. `itertools.count` is atomic under the GIL,
# avoiding the read-modify-write race a shared counter would have.
_clause_seq = itertools.count(1)


@app.get("/healthz")
async def healthz() -> dict:
    scorer_health = await scorer.health()
    return {
        "ready": bool(scorer_health.get("ready")),
        "scorer": scorer_health.get("backend", "mock"),
        "warmed": state.warmed,
    }


@app.post("/ingest")
async def ingest(request: IngestRequest) -> IngestResponse:
    if request.corpus_id != "demo":
        raise HTTPException(status_code=404, detail="Phase 0/1 only support corpus_id='demo'")

    chunks = state.load_demo()
    if state.current_clause is not None:
        cache.evict_clause(state.current_clause)
    state.current_clause = None
    state.warm_state = await scorer.warm(request.corpus_id, chunks)
    return IngestResponse(
        corpus_id=request.corpus_id,
        n_chunks=len(chunks),
        facets=facet_summary(chunks),
        warm_eta_s=0.0,
    )


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    if not state.chunks:
        state.load_demo()

    clause_id = f"q{next(_clause_seq)}"
    previous = state.current_clause
    state.current_clause = clause_id
    if previous is not None:
        # Evicting the *previous* (distinct) clause is race-free: it can't touch
        # this request's fresh column.
        cache.evict_clause(previous)

    async def frames() -> AsyncIterator[str]:
        async for event in query_stream(
            scorer,
            state.chunks,
            request.predicate,
            clause_id=clause_id,
            threshold=request.threshold,
            cache=cache,
        ):
            yield sse(event)

    return StreamingResponse(frames(), media_type="text/event-stream")


@app.get("/results")
async def results(threshold: float = 0.5, top_k: int | None = None) -> ResultsResponse:
    """Pure cache read — the threshold/top-k slice the client mirrors locally.

    Does NOT call the scorer; it re-cuts cached scores. Server-side proof of the
    "drag = zero inference" claim and eval parity.
    """
    chunks_by_id = state.chunks_by_id()
    clause = state.current_clause
    scored = cache.scores_for_clause(clause) if clause else {}
    ranked = sorted(
        (result for result in scored.values() if result.chunk_id in chunks_by_id),
        key=lambda r: r.score,
        reverse=True,
    )
    matched = [r for r in ranked if r.score >= threshold]
    sliced = matched[:top_k] if top_k is not None else matched
    items = [
        ResultEvent.from_score(result, chunks_by_id[result.chunk_id], rank)
        for rank, result in enumerate(sliced)
    ]
    return ResultsResponse(
        items=items,
        threshold=threshold,
        top_k=top_k,
        total_matched=len(matched),
    )


def sse(event: ResultEvent | AggregateEvent | DoneEvent) -> str:
    return f"data: {json.dumps(model_dump(event), separators=(',', ':'))}\n\n"


def model_dump(event: ResultEvent | AggregateEvent | DoneEvent) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    return event.dict()
