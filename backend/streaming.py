from __future__ import annotations

import math
import os
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.score_store import ScoreStore

from data.schema import Chunk
from inference.scorer import ScoreRequest, ScoreResult, ScorerClient

from backend.cache import ScoreCache
from backend.schemas import AggregateEvent, DoneEvent, ResultEvent
from backend.state import facet_summary, histogram

# Default batch size for cold scans. A knob, not the KPI: small enough that the
# dashboard visibly fills as it scans, large enough to keep vLLM batches efficient.
BATCH_SIZE = int(os.environ.get("QUERY_BATCH_SIZE", "64"))

StreamEvent = ResultEvent | AggregateEvent | DoneEvent


def _chunked(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def query_stream(
    scorer: ScorerClient,
    chunks: list[Chunk],
    predicate: str,
    *,
    clause_id: str = "base",
    threshold: float = 0.5,
    cache: ScoreCache,
    batch_size: int = BATCH_SIZE,
    tier: int = 1,
    compute_budget: float = 1.0,
    store: "ScoreStore | None" = None,
    collection: str = "",
    scorer_tag: str = "",
) -> AsyncIterator[StreamEvent]:
    """Score the corpus in batches and stream results best-first with running aggregates.

    Every scored chunk is emitted as a ``result`` event (best-first *within* the
    batch — the reorder window), so the client's score cache ends up complete and
    its threshold re-cut is a pure client-side computation with zero inference.
    A running ``aggregate`` is emitted after each batch (histogram + facets +
    matched + ETA), then a terminal ``done``.

    Axis 1 (Memory): ``compute_budget`` (0 < b <= 1) scores only the first
    ``ceil(b * N)`` chunks — the corpus in scope grows linearly with compute.
    """
    started = time.perf_counter()
    corpus_total = len(chunks)
    budget = min(1.0, max(0.0, compute_budget))
    in_scope = (
        corpus_total
        if budget >= 1.0
        else min(corpus_total, max(1, math.ceil(budget * corpus_total)))
    )
    scoped_chunks = chunks[:in_scope]
    by_id = {chunk.chunk_id: chunk for chunk in scoped_chunks}
    total = len(scoped_chunks)
    scanned = 0
    rank = 0
    ema_batch_ms: float | None = None
    seen_scores: dict[str, ScoreResult] = {}

    for batch in _chunked(scoped_chunks, max(1, batch_size)):
        batch_start = time.perf_counter()
        candidate_ids = {chunk.chunk_id for chunk in batch}
        missing = cache.missing(clause_id, candidate_ids)
        if missing and store is not None:
            # Read-through: pull persisted scores for this (collection, predicate,
            # model) so only true misses go to the scorer (fetch instead of rescan).
            fetched = store.get_scores(collection, predicate, scorer_tag, missing)
            for chunk_id, result in fetched.items():
                cache.put(chunk_id, clause_id, result)
            missing = {cid for cid in missing if cid not in fetched}
        if missing:
            requests = [
                ScoreRequest(chunk_id=chunk.chunk_id, chunk_text=chunk.text, predicate=predicate)
                for chunk in batch
                if chunk.chunk_id in missing
            ]
            scored = await scorer.score_batch(requests, tier=tier)
            for result in scored:
                cache.put(result.chunk_id, clause_id, result)
            if store is not None:
                store.put_scores(collection, predicate, scorer_tag, scored)

        # Pull every chunk without changing hit counters; missing() already
        # recorded whether each candidate was a cache hit or scorer miss.
        batch_results: list[ScoreResult] = []
        for chunk in batch:
            cached = cache.peek(chunk.chunk_id, clause_id)
            if cached is not None:
                batch_results.append(cached)
                seen_scores[chunk.chunk_id] = cached

        scanned += len(batch)
        batch_ms = (time.perf_counter() - batch_start) * 1000.0
        ema_batch_ms = batch_ms if ema_batch_ms is None else (0.5 * ema_batch_ms + 0.5 * batch_ms)

        # Emit results best-first within the batch (the reorder window).
        # NOTE (scale caveat): we emit one result event per scored chunk so the
        # client's score cache ends up complete (enabling zero-inference threshold
        # recut). That is O(corpus) SSE frames — fine at 10–20k, but at 100k+ this
        # must switch to streaming a capped top-N for the feed while the histogram
        # and facets come from server-side full aggregation. Tracked for Phase 3.
        for result in sorted(batch_results, key=lambda r: r.score, reverse=True):
            yield ResultEvent.from_score(result, by_id[result.chunk_id], rank)
            rank += 1

        scored_chunks = [by_id[cid] for cid in seen_scores]
        matched = sum(1 for r in seen_scores.values() if r.score >= threshold)
        batches_remaining = max(0, (total - scanned + batch_size - 1) // batch_size)
        eta_ms = int(batches_remaining * (ema_batch_ms or 0.0))

        yield AggregateEvent(
            scanned=scanned,
            matched=matched,
            histogram=histogram(list(seen_scores.values())),
            facets=facet_summary(scored_chunks, seen_scores, threshold),
            threshold=threshold,
            eta_ms=eta_ms,
            corpus_total=corpus_total,
            corpus_scored=total,
            compute_budget=budget,
        )

    if total == 0:
        # Empty corpus: still emit one aggregate so the stream always carries
        # result*/aggregate/done in the documented order.
        yield AggregateEvent(
            scanned=0,
            matched=0,
            histogram=histogram([]),
            facets=facet_summary([], {}, threshold),
            threshold=threshold,
            eta_ms=0,
            corpus_total=corpus_total,
            corpus_scored=0,
            compute_budget=budget,
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000.0)
    matched = sum(1 for r in seen_scores.values() if r.score >= threshold)
    yield DoneEvent(
        scanned=total,
        matched=matched,
        elapsed_ms=elapsed_ms,
        warm=False,
        summary=f"{total:,} scanned · {matched:,} matched",
        corpus_total=corpus_total,
        corpus_scored=total,
        compute_budget=budget,
    )
