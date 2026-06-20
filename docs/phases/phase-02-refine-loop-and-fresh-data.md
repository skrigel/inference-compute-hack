# Phase 02 - Refine Loop And Fresh Data

**Window:** H8-H14  
**Milestone:** M3 refine loop end to end  
**Theme:** build the interaction that RAG cannot follow: incremental semantic
refinement and fresh data without re-indexing.

## Goals

- Implement `/refine` with chip-first SSE, scoped re-score, diff, aggregate, and done.
- Implement click-to-drop and click-to-keep operations with stable `sign` payloads.
- Implement chip removal as zero-inference recompute from cache.
- Add drag-in fresh-file ingest and automatic re-query.
- Record canned SSE from the working mock path.

## Owner Work

| Owner | Work |
|---|---|
| A | Counterfactual replay function for scoped vs full vs suffix vs RAG curves |
| B | clause engine, cache missing-set logic, `/refine`, `DELETE /clause/{clause_id}` |
| C | refine box, chip rail, optimistic chip display, diff application, fresh-file drop |
| D | curated demo beats with known good and known wrong matches |

## Performance Metrics To Capture

- Per refine turn: `candidate_count`, `chunks_scored`, `cache_hits`, `survivor_count`, `rho`, `refine_ms`.
- Chip removal: elapsed time and model calls, expected model calls = 0.
- Fresh ingest: time from file drop to queryable chunks, plus RAG re-index comparison.
- Cumulative compute: scoped curve from measured trace, counterfactual full and suffix curves from the same trace.

## Exit Gate

- Scripted query -> click-NOT -> AND refine -> threshold drag -> fresh-file drop works on mock.
- `/refine` first event is always `chip`.
- Canned SSE fixture exists for every demo beat.
- Area-under-loop graph can be generated from one recorded session trace.

## Fallback

If the classifier is unreliable, force rules-first operations for demo phrases and
surface low-confidence chips as removable. If scoped recompute misbehaves, use
`REFINE_MODE=full` for correctness and keep the measured scoped trace as a Phase 04 fix.
