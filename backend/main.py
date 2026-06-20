from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import replace

from data.schema import Chunk
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from inference.config import make_scorer
from inference.scorer import ScoreRequest, ScoreResult

from backend.beam import run_beam
from backend.cache import ScoreCache
from backend.classifier import ClassifiedRefine, classify_refine
from backend.clause import ClauseRecord, label_for
from backend.knowledge import fetch_arxiv_documents
from backend.schemas import (
    AggregateEvent,
    ArxivIngestRequest,
    BeamEvent,
    Chip,
    ChipEvent,
    ClauseDeleteResponse,
    DiffEvent,
    DoneEvent,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RefineOp,
    RefineRequest,
    ResultEvent,
    ResultsResponse,
    SelectRequest,
    SelectResponse,
)
from backend.score_store import ScoreStore
from backend.select import auto_threshold, smart_select
from backend.state import BackendState, facet_summary, histogram
from backend.streaming import query_stream

app = FastAPI(title="FlashGrep Backend", version="0.1.0")
state = BackendState()
cache = ScoreCache()
scorer = make_scorer()
# Persistent SQLite score cache — repeated queries fetch stored scores instead of
# re-scanning the corpus on the GPU. Opt-in via SCORE_CACHE=1 so the test suite and
# casual runs keep deterministic re-scan behavior; the demo backend turns it on.
SCORE_CACHE_ON = os.environ.get("SCORE_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}
score_store = ScoreStore() if SCORE_CACHE_ON else None
SCORER_TAG = os.environ.get("SCORER_BACKEND", "mock").lower()

# Each query mints a fresh clause id (q1, q2, …) so concurrent queries never
# evict each other's cached column. `itertools.count` is atomic under the GIL,
# avoiding the read-modify-write race a shared counter would have.
_clause_seq = itertools.count(1)
REFINE_BATCH_SIZE = max(1, int(os.environ.get("REFINE_BATCH_SIZE", "64")))
# Axis 3 (Truth): a beam-selected candidate must retain at least this fraction of
# the parent survivors to be eligible — the "minimum coverage threshold" the
# objective function is maximised subject to.
MIN_BEAM_COVERAGE = float(os.environ.get("BEAM_MIN_COVERAGE", "0.2"))


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
    if request.corpus_id not in ("demo", "browsecomp") and not request.documents:
        raise HTTPException(status_code=404, detail="Supported corpus_id: 'demo', 'browsecomp', or provide documents")

    if request.documents:
        if not state.chunks:
            state.load_demo()
        chunks = state.append_documents(request.documents)
        cache.clear()
    elif request.corpus_id == "browsecomp":
        chunks = state.load_browsecomp(limit=request.limit)
        cache.clear()
    else:
        chunks = state.load_demo()
        cache.clear()

    state.warm_state = await scorer.warm(request.corpus_id, chunks)
    return IngestResponse(
        corpus_id=request.corpus_id,
        n_chunks=len(chunks),
        facets=facet_summary(chunks),
        warm_eta_s=0.0,
    )


@app.post("/source/arxiv")
async def ingest_arxiv_source(request: ArxivIngestRequest) -> IngestResponse:
    documents = fetch_arxiv_documents(request.query, max_results=request.count)
    if not documents:
        raise HTTPException(status_code=404, detail="arXiv returned no papers for this query")
    if not state.chunks:
        state.corpus_id = "arxiv"
    chunks = state.append_documents(documents)
    cache.clear()
    state.warm_state = await scorer.warm("arxiv", chunks)
    return IngestResponse(
        corpus_id="arxiv",
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
    state.threshold = request.threshold
    state.clauses[clause_id] = ClauseRecord(
        clause_id=clause_id,
        op="query",
        text=request.predicate,
        parent_clause_id=None,
        removable=False,
    )
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
            tier=1,
            compute_budget=request.compute_budget,
            store=score_store,
            collection=getattr(state, "corpus_id", "demo"),
            scorer_tag=SCORER_TAG,
        ):
            yield sse(event)

    return StreamingResponse(frames(), media_type="text/event-stream")


@app.post("/refine")
async def refine(request: RefineRequest) -> StreamingResponse:
    if not state.current_clause:
        raise HTTPException(status_code=409, detail="Run /query before /refine")

    events = await refine_events(request)

    async def frames() -> AsyncIterator[str]:
        for event in events:
            yield sse(event)

    return StreamingResponse(frames(), media_type="text/event-stream")


@app.delete("/clause/{clause_id}")
async def delete_clause(clause_id: str) -> ClauseDeleteResponse:
    started = time.perf_counter()
    clause = state.clauses.get(clause_id)
    if clause is None or not clause.removable:
        return ClauseDeleteResponse(removed=False, refine_ms=_elapsed_ms(started))
    if state.current_clause == clause_id or _depends_on(state.current_clause, clause_id):
        state.current_clause = clause.parent_clause_id
    for child_id in _descendants_of(clause_id):
        state.clauses.pop(child_id, None)
    del state.clauses[clause_id]
    return ClauseDeleteResponse(removed=True, refine_ms=_elapsed_ms(started))


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


@app.post("/select")
async def select(request: SelectRequest) -> SelectResponse:
    """Axis 2 (Movement): auto-threshold or smart-select over cached scores.

    Pure cache read — never calls the scorer. ``threshold`` mode auto-sets the
    cutoff to the precision target (Mode A); ``smart`` mode runs a max-coverage
    beam search over output subsets within the movement budget (Mode B).
    """
    started = time.perf_counter()
    clause = state.current_clause
    scored = cache.scores_for_clause(clause) if clause else {}
    chunks_by_id = state.chunks_by_id()
    score_values = [
        result.score for chunk_id, result in scored.items() if chunk_id in chunks_by_id
    ]

    threshold, selected_count = auto_threshold(score_values, request.precision_target)

    if request.mode == "threshold":
        selected_ids = sorted(
            (
                chunk_id
                for chunk_id, result in scored.items()
                if chunk_id in chunks_by_id and result.score >= threshold
            ),
            key=lambda cid: scored[cid].score,
            reverse=True,
        )
        return SelectResponse(
            mode="threshold",
            threshold=threshold,
            selected_ids=selected_ids,
            selected_count=len(selected_ids),
            refine_ms=_elapsed_ms(started),
        )

    selection = smart_select(
        chunks_by_id,
        scored,
        threshold=threshold,
        movement_budget=request.movement_budget,
        beam_width=request.beam_width,
    )
    return SelectResponse(
        mode="smart",
        threshold=threshold,
        selected_ids=selection.selected_ids,
        selected_count=len(selection.selected_ids),
        covered_facets=selection.covered_facets,
        objective=selection.objective,
        greedy_objective=selection.greedy_objective,
        movement_budget=selection.movement_budget,
        beam_width=selection.beam_width,
        candidate_pool=selection.candidate_pool,
        refine_ms=_elapsed_ms(started),
    )


StreamEvent = ResultEvent | AggregateEvent | DoneEvent | ChipEvent | DiffEvent | BeamEvent


def sse(event: StreamEvent) -> str:
    return f"data: {json.dumps(model_dump(event), separators=(',', ':'))}\n\n"


async def refine_events(request: RefineRequest) -> list[BeamEvent | ChipEvent | DiffEvent | AggregateEvent | DoneEvent]:
    parent_clause = state.current_clause
    if parent_clause is None:
        raise HTTPException(status_code=409, detail="Run /query before /refine")

    chunks_by_id = state.chunks_by_id()
    parent_scores = cache.scores_for_clause(parent_clause)
    if not parent_scores:
        raise HTTPException(status_code=409, detail="Current query has no cached scores yet")

    started = time.perf_counter()
    previous_survivors = _survivors(parent_scores)
    beam_event: BeamEvent | None = None
    if request.utterance and request.beam_width > 1:
        beam_event, winner_text = await run_beam(
            scorer,
            request.utterance,
            request.beam_width,
            previous_survivors,
            parent_scores,
            chunks_by_id,
            threshold=state.threshold,
            min_coverage=MIN_BEAM_COVERAGE,
        )
        operation, text, confidence, target_chunk_id = (
            RefineOp.require,
            winner_text,
            beam_event.candidates[beam_event.chosen_index].objective,
            None,
        )
    else:
        operation, text, confidence, target_chunk_id = _refine_intent(request)
    clause_id = f"c{next(_clause_seq)}"
    next_scores, rescored_ids = await _apply_refine(
        operation,
        text,
        target_chunk_id,
        parent_scores,
        chunks_by_id,
        clause_id,
    )
    state.current_clause = clause_id
    state.clauses[clause_id] = ClauseRecord(
        clause_id=clause_id,
        op=operation,
        text=text,
        parent_clause_id=parent_clause,
        target_chunk_id=target_chunk_id,
    )

    next_survivors = _survivors(next_scores)
    refine_ms = _elapsed_ms(started)
    ranked = _ranked_results(next_scores, chunks_by_id)
    added = [event for event in ranked if event.chunk_id in (next_survivors - previous_survivors)]
    removed = sorted(previous_survivors - next_survivors)
    rescored = [{"chunk_id": chunk_id, "score": next_scores[chunk_id].score} for chunk_id in sorted(rescored_ids)]

    chip = Chip(
        clause_id=clause_id,
        op=operation,
        text=text,
        label=label_for(operation),
        removable=True,
        confidence=confidence,
    )
    scored_chunks = [chunks_by_id[chunk_id] for chunk_id in next_scores if chunk_id in chunks_by_id]
    aggregate = AggregateEvent(
        scanned=len(rescored_ids),
        matched=len(next_survivors),
        histogram=histogram(list(next_scores.values())),
        facets=facet_summary(scored_chunks, next_scores, state.threshold),
        threshold=state.threshold,
        eta_ms=0,
    )
    done = DoneEvent(
        scanned=len(rescored_ids),
        matched=len(next_survivors),
        elapsed_ms=refine_ms,
        warm=True,
        summary=f"{len(rescored_ids):,} rescored · {len(next_survivors):,} matched",
    )
    chip_event = ChipEvent(operation=operation, chip=chip, refine_ms=refine_ms, latency_kind="warm")
    diff_event = DiffEvent(added=added, removed=removed, rescored=rescored, refine_ms=refine_ms)
    if beam_event is not None:
        return [beam_event, chip_event, diff_event, aggregate, done]
    return [chip_event, diff_event, aggregate, done]


def _refine_intent(request: RefineRequest) -> tuple[RefineOp, str, float, str | None]:
    if request.click is not None:
        operation = RefineOp.exclude if request.click.sign == "-" else RefineOp.require
        text = f"{'drop' if request.click.sign == '-' else 'keep'} {request.click.chunk_id}"
        return operation, text, 1.0, request.click.chunk_id
    if request.brush is not None:
        text = f"{request.brush.lo:.2f} to {request.brush.hi:.2f}"
        return RefineOp.brush, text, 1.0, None
    if request.utterance:
        classified: ClassifiedRefine = classify_refine(request.utterance)
        return classified.operation, request.utterance, classified.confidence, None
    raise HTTPException(status_code=422, detail="Refine request needs utterance, click, or brush")


async def _apply_refine(
    operation: RefineOp,
    text: str,
    target_chunk_id: str | None,
    parent_scores: dict[str, ScoreResult],
    chunks_by_id: dict[str, Chunk],
    clause_id: str,
) -> tuple[dict[str, ScoreResult], set[str]]:
    next_scores = dict(parent_scores)
    if operation == RefineOp.brush:
        _store_scores(clause_id, next_scores)
        return next_scores, set()

    if target_chunk_id is not None:
        if target_chunk_id not in next_scores:
            raise HTTPException(status_code=404, detail="Unknown chunk_id for click refine")
        parent = next_scores[target_chunk_id]
        score = 0.0 if operation == RefineOp.exclude else 1.0
        next_scores[target_chunk_id] = replace(parent, score=score, p_yes=score, p_no=1.0 - score)
        _store_scores(clause_id, next_scores)
        return next_scores, {target_chunk_id}

    if operation == RefineOp.refocus:
        candidate_ids = set(chunks_by_id)
    elif operation == RefineOp.include:
        candidate_ids = set(parent_scores) - _survivors(parent_scores)
    else:
        candidate_ids = _survivors(parent_scores)

    missing = cache.missing(clause_id, candidate_ids)
    if missing:
        requests = [
            ScoreRequest(chunk_id=chunk_id, chunk_text=chunks_by_id[chunk_id].text, predicate=text)
            for chunk_id in sorted(missing)
            if chunk_id in chunks_by_id
        ]
        for start in range(0, len(requests), REFINE_BATCH_SIZE):
            batch = requests[start : start + REFINE_BATCH_SIZE]
            for result in await scorer.score_batch(batch, tier=0):
                cache.put(result.chunk_id, clause_id, result)

    for chunk_id in candidate_ids:
        clause_score = cache.peek(chunk_id, clause_id)
        parent = parent_scores.get(chunk_id)
        if clause_score is None:
            continue
        if operation == RefineOp.require and parent is not None:
            score = parent.score * clause_score.score
        elif operation == RefineOp.exclude and parent is not None:
            score = parent.score * (1.0 - clause_score.score)
        elif operation == RefineOp.include:
            score = max(parent.score if parent else 0.0, clause_score.score)
        else:
            score = clause_score.score
        next_scores[chunk_id] = replace(clause_score, score=score, p_yes=score, p_no=1.0 - score)

    _store_scores(clause_id, next_scores)
    return next_scores, set(candidate_ids)


def _survivors(scores: dict[str, ScoreResult]) -> set[str]:
    return {chunk_id for chunk_id, score in scores.items() if score.score >= state.threshold}


def _ranked_results(scores: dict[str, ScoreResult], chunks_by_id: dict[str, Chunk]) -> list[ResultEvent]:
    ranked = sorted(scores.values(), key=lambda result: result.score, reverse=True)
    return [
        ResultEvent.from_score(result, chunks_by_id[result.chunk_id], rank)
        for rank, result in enumerate(ranked)
        if result.chunk_id in chunks_by_id
    ]


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000.0))


def _store_scores(clause_id: str, scores: dict[str, ScoreResult]) -> None:
    for result in scores.values():
        cache.put(result.chunk_id, clause_id, result)


def _depends_on(clause_id: str | None, ancestor_id: str) -> bool:
    while clause_id is not None:
        if clause_id == ancestor_id:
            return True
        clause = state.clauses.get(clause_id)
        clause_id = clause.parent_clause_id if clause else None
    return False


def _descendants_of(clause_id: str) -> set[str]:
    descendants = set()
    for candidate_id in list(state.clauses):
        if candidate_id != clause_id and _depends_on(candidate_id, clause_id):
            descendants.add(candidate_id)
    return descendants


def model_dump(event: StreamEvent) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    return event.dict()
