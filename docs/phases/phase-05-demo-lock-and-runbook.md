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

## Implemented In This Phase

- **Locked `DEMO.md`:** word-for-word spoken beats with a per-beat timing budget summing to
  ~85s (≤90s), the pinned-corpus table with measured beat counts, copy-paste operator commands
  per fallback tier, and the honest demo posture.
- **Frozen eval slide `eval/SLIDE.md`:** every figure labeled by provenance (measured-mock /
  measured-6×H100 / projected / pending), with the explicit **MFU caveat** (the 0.06 derived MFU
  is an under-saturated micro-benchmark, not the architecture's MFU) and the **pending** quality
  gate (F1 ≥ 0.7 enforced in code, not yet run on the real model).
- **Complete canned fallback per beat:** `scripts/replay_sse.py` now records and serves a
  fresh-file fixture (`cut_line_fresh.sse`) and arms it after a file drop, so beats 1/2-3/5 all
  have byte-identical canned twins (beat 4 is a client-side recut). Guarded by
  `tests/test_phase5_demo_lock.py`.
- **Preflight:** `scripts/preload_demo.sh` drives the loop, records all fixtures, verifies every
  demo-lock artifact, and prints GO/NO-GO with beat coverage.

## Known Carry-Forward (blocks "Phase 04 complete", not the demo)

- Real-vLLM quality-gate **freeze is pending** the Modal spend-limit reset
  (`eval/artifacts/phase04_modal_blocker.md`). The demo leads with the live mock backend; the
  6×H100 throughput/latency is shown as *measured-6×H100* with the MFU caveat, and no unverified
  real-model number enters the deck (SCHEDULE H19 gate).
- The canned replay twins are recorded from the mock backend; re-record from a real vLLM run once
  the box is unblocked, then state "recorded from the same scorer path."
- Frontend `dist/` must be built ahead of time on Node ≥ 20.19 (Vite 8).
