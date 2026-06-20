# Phase 05 - Demo Lock And Runbook

**Window:** H19-H22  
**Milestone:** M5 demo locked  
**Theme:** turn the working build into a reliable 90-second story.

## Goals

- Finalize [`../../DEMO.md`](../../DEMO.md) with exact spoken beats and fallback switches.
- Write `scripts/preload_demo.sh` to boot, ingest, warm, health-check, and report go/no-go.
- Freeze eval slide figures and captions.
- Remove scope that does not serve the 90-second loop.

## Demo Beats

1. Stream best-first results for a semantic predicate.
2. Drop a wrong match with click-NOT.
3. Add an AND refine in natural language.
4. Drag threshold and show zero-inference recut.
5. Drop a fresh file and query it instantly.
6. Optional close: performance slide with area-under-loop and roofline.

## Performance Metrics To Surface

- Refine p50/p95 and whether the displayed turn was cold, warm, or cached.
- Cumulative compute curve: scoped vs full vs RAG.
- Break-even churn `D*` for fresh data.
- Quality gate status: F1/AUC before speed claims.

## Exit Gate

- Demo is under 90 seconds without rushing.
- Every live beat has a canned fallback.
- Operator runbook says exactly which env vars and commands to use.
- No unfinished stretch work appears in the primary path.

## Fallback

If the live H100 path is unreliable, lead with replay from a real vLLM fixture and
state that the fixture was recorded from the same scorer path. Keep the mock path
for local UI recovery only.
