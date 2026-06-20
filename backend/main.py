from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from inference.config import make_scorer
from inference.scorer import ScoreRequest, ScoreResult

from backend.schemas import AggregateEvent, DoneEvent, IngestRequest, IngestResponse, QueryRequest, ResultEvent
from backend.state import BackendState, facet_summary, histogram


app = FastAPI(title="Inference Compute Hack Backend", version="0.0.0")
state = BackendState()
scorer = make_scorer()


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
        raise HTTPException(status_code=404, detail="Phase 0 only supports corpus_id='demo'")

    chunks = state.load_demo()
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

    async def frames() -> AsyncIterator[str]:
        started = time.perf_counter()
        items = [
            ScoreRequest(chunk_id=chunk.chunk_id, chunk_text=chunk.text, predicate=request.predicate)
            for chunk in state.chunks
        ]
        scores = await scorer.score_batch(items)
        state.query_count += 1
        state.last_scores = {score.chunk_id: score for score in scores}
        chunks_by_id = state.chunks_by_id()
        ranked = sorted(scores, key=lambda score: score.score, reverse=True)
        matched_scores = [score for score in ranked if score.score >= request.threshold]

        for rank, score in enumerate(matched_scores[:10]):
            yield sse(ResultEvent.from_score(score, chunks_by_id[score.chunk_id], rank))

        yield sse(
            AggregateEvent(
                scanned=len(scores),
                matched=len(matched_scores),
                histogram=histogram(scores),
                facets=facet_summary(state.chunks, state.last_scores, request.threshold),
                threshold=request.threshold,
                eta_ms=0,
            )
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        yield sse(
            DoneEvent(
                scanned=len(scores),
                matched=len(matched_scores),
                elapsed_ms=elapsed_ms,
                warm=state.warmed,
                summary=f"{len(scores):,} scanned · {len(matched_scores):,} matched",
            )
        )

    return StreamingResponse(frames(), media_type="text/event-stream")


def sse(event: ResultEvent | AggregateEvent | DoneEvent) -> str:
    return f"data: {json.dumps(model_dump(event), separators=(',', ':'))}\n\n"


def model_dump(event: ResultEvent | AggregateEvent | DoneEvent) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    return event.dict()
