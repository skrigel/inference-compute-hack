# Phase 01 - Ingest, Query, And Dashboard

**Window:** H3-H8  
**Milestones:** M1 backend query stream, M2 frontend consumes stream  
**Theme:** make the core live experience work on mock while preserving the
performance counters needed for the final story.

## Goals

- Implement `/ingest` and `/query` as one multiplexed SSE stream.
- Render live results, histogram, facets, counters, and threshold handle.
- Prove threshold drag is a client-side recut over cached scores with zero network call.
- Keep mock timings faithful enough for the demo fallback.

## Owner Work

| Owner | Work |
|---|---|
| A | RAG baseline skeleton and first index-build/retrieve timings |
| B | `/ingest`, `/query`, batching, aggregate generation, score-cache writes |
| C | `streamPost()`, result feed, histogram, facets, counters, latency readout |
| D | deterministic corpus build and facet metadata |

## Performance Metrics To Capture

- Query stream: `scanned`, `matched`, `elapsed_ms`, `eta_ms`, batch size, reorder-window depth.
- Histogram recut: number of model calls during drag must be exactly zero.
- RAG baseline: `index_build_ms`, `query_embed_ms`, `ann_ms`, `rerank_ms`.
- Area-under-loop inputs: initial corpus size `N` and first survivor count at default threshold.

## Exit Gate

- Backend can ingest the seeded corpus and stream `result`, `aggregate`, and `done`.
- Frontend consumes the live stream and remains usable with `VITE_DATA_MODE=mock`.
- Unit or harness check proves threshold drag does not call the scorer or backend.
- RAG baseline emits a timing record even if it uses the local fallback implementation.

## Fallback

If SSE integration is unstable, keep the frontend fed through the mock adapter and
record a canned SSE fixture from the backend as soon as M1 is stable.
