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

## Implemented In This Phase

- **Loop harness + clean trace:** `eval/cut_line.py` drives the seven required steps
  against the in-process backend with a model-call counter, asserts the loop is GREEN,
  and emits `eval/artifacts/cut_line_trace.json`. Guarded by `tests/test_phase3_cut_line.py`.
- **Pinned demo data:** the cut-line surfaced a real gap — the live backend's headline
  query yielded only one survivor. Fixed by calibrating `inference/mock_scorer.py` to
  concept-substring scoring and curating `backend/state.py::demo_chunks` to 7 chunks so
  the scripted beats have known results (query → 5 matched; click-NOT → 4; AND → 3).
  The Phase 2 test was made corpus-agnostic (asserts the scoping invariant, not magic numbers).
- **Three proofs:** zero-inference threshold drag (`chunks_scored == 0`), fresh-file vs
  RAG re-index (`fresh_vs_rag`), and the area-under-loop money shot
  (`eval/artifacts/area_under_loop.png`, scoped 11 vs full 21 chunks over the 3-turn refine loop).
- **Fallback ladder:** `scripts/replay_sse.py` records (`cut_line_query.sse`,
  `cut_line_refine.sse`) and serves them byte-for-byte; `tests/test_phase3_replay.py`
  proves the frontend can fail over to it. `scripts/preload_demo.sh` is the GO/NO-GO preflight.
- **Demo:** `DEMO.md` now has the exact beats with measured scanned/matched counts, the
  pinned corpus table, the operator runbook, and the tested fallback ladder.

## Known Carry-Forward (Phase 04)

- Real latencies are mock-tiny here; the measured warm/refine p50 comes from the box.
- Frontend toolchain requires **Node ≥ 20.19** (Vite 8 / vitest 4) — build `dist/` on a
  Node-20 machine ahead of the demo; the venue box may not `npm install`.
