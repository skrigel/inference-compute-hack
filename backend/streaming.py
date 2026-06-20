from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterable, Iterator

from data.schema import Chunk
from inference.scorer import ScoreRequest, ScoreResult, ScorerClient

from backend.cache import ScoreCache
from backend.schemas import AggregateEvent, DoneEvent, ResultEvent
from backend.state import facet_summary, histogram

# Default batch size for cold scans. A knob, not the KPI: small enough that the
# dashboard visibly fills as it scans, large enough to keep vLLM batches efficient.
BATCH_SIZE = 64

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
) -> AsyncIterator[StreamEvent]:
    """Score the corpus in batches and stream results best-first with running aggregates.

    Every scored chunk is emitted as a ``result`` event (best-first *within* the
    batch — the reorder window), so the client's score cache ends up complete and
    its threshold re-cut is a pure client-side computation with zero inference.
    A running ``aggregate`` is emitted after each batch (histogram + facets +
    matched + ETA), then a terminal ``done``.
    """
    started = time.perf_counter()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    total = len(chunks)
    scanned = 0
    rank = 0
    ema_batch_ms: float | None = None
    seen_scores: dict[str, ScoreResult] = {}

    for batch in _chunked(chunks, max(1, batch_size)):
        batch_start = time.perf_counter()
        candidate_ids = {chunk.chunk_id for chunk in batch}
        missing = cache.missing(clause_id, candidate_ids)
        if missing:
            requests = [
                ScoreRequest(chunk_id=chunk.chunk_id, chunk_text=chunk.text, predicate=predicate)
                for chunk in batch
                if chunk.chunk_id in missing
            ]
            for result in await scorer.score_batch(requests):
                cache.put(result.chunk_id, clause_id, result)

        # Pull every chunk in the batch from cache (hits + fresh scores).
        batch_results: list[ScoreResult] = []
        for chunk in batch:
            cached = cache.get(chunk.chunk_id, clause_id)
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
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000.0)
    matched = sum(1 for r in seen_scores.values() if r.score >= threshold)
    yield DoneEvent(
        scanned=total,
        matched=matched,
        elapsed_ms=elapsed_ms,
        warm=False,
        summary=f"{total:,} scanned · {matched:,} matched",
    )
