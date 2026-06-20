# Phase 03 - Cut-Line Stabilization

**Window:** H14 hard cut line  
**Milestone:** core loop green  
**Theme:** stop expanding scope and stabilize the irreducible demo.

## Required Loop

The project is demoable if this loop works:

1. Ingest a corpus.
2. Run a plain-language semantic query.
3. Stream ranked results and aggregates.
4. Apply one click-NOT refine.
5. Apply one AND refine.
6. Drag threshold with zero inference.
7. Drop a fresh file and query it immediately.

## Stop/Continue Rules

- If the loop is green, continue to Phase 04 and the real vLLM swap.
- If any loop step is red, stop adding features and fix only that step.
- If the live path is shaky but canned SSE is good, polish the canned path and keep the live path as a best-effort optional beat.
- Do not start Tier-2 cascade, second domain, or editable chip algebra unless the loop is green.

## Performance Metrics To Preserve

- One clean trace of the loop with `chunks_scored` and cache hits.
- One screenshot or figure for the area-under-loop money shot.
- One explicit threshold-drag zero-inference proof.
- One fresh-file comparison showing recompute now vs RAG re-index later.

## Exit Gate

- The fallback ladder has been tested: real backend, mock backend, canned SSE replay.
- Demo data is pinned and all scripted predicates have known expected results.
- `DEMO.md` has the exact beats and the operator knows the next click.
