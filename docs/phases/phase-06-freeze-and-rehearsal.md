# Phase 06 - Freeze And Rehearsal

**Window:** H22-H24  
**Milestone:** repo and demo freeze  
**Theme:** rehearse failure, not just success.

## Goals

- Run three dress rehearsals:
  - live path;
  - injected failure forcing replay fallback;
  - timed <90 second run.
- Freeze the repo and demo data.
- Verify all docs point to the same claims and metric values.

## Freeze Checklist

- `CONTRACTS.md` matches implementation and has no unapproved schema drift.
- `PLAN.md`, `SCHEDULE.md`, phase docs, `METRICS.md`, `DEMO.md`, and `RISKS.md` agree on milestones.
- `performance/theory.py` regenerates figures in `performance/figures/`.
- Figures used in slides are labeled as measured, projected, or predicted.
- Fallback fixtures are present and bootable.
- The operator can recover from failed backend, failed scorer, and failed network in under 30 seconds.

## Performance Metrics To Preserve

- Commit hash, model id, corpus size, chunk count, token counts.
- Frozen metrics JSON and generated figures.
- Any manual constants changed in `performance/theory.py`.
- Known caveats: score quality, warm-cache capacity, MFU denominator, replay provenance.

## Exit Gate

- Three rehearsals pass.
- No new feature work remains in progress.
- The final slide numbers are reproducible from saved artifacts.
- The team can explain why the metrics are performance metrics, not generic telemetry.

## Implemented In This Phase

- **Rehearsal harness `eval/rehearsal.py`:** runs the three dress rehearsals (R1 live loop GREEN,
  R2 injected-failure → canned replay serves every beat, R3 spoken budget 85s ≤ 90s) and writes a
  reproducible `eval/artifacts/freeze_manifest.json`. Guarded by `tests/test_phase6_freeze.py`.
- **`FREEZE.md`:** the freeze checklist (all items ✓), the labeled frozen-metric table read from
  saved artifacts, the caveats, and the one-command path to upgrade the freeze once unblocked.
- **Doc reconciliation:** the newly-pushed real-vLLM small gate (`phase04_quality_gate.json`,
  F1 0.94 @ calibrated ≈0.016) was reconciled into `eval/SLIDE.md` and `DEMO.md` — the gate is now
  shown as *measured* (not pending), with the **threshold-calibration caveat** (real backend needs
  ≈0.016; default 0.5 collapses recall to ~0.11) and the **small-gate-vs-full-freeze** distinction.
- **Blocker reconciliation:** the small gate ran on Modal; the full ladder/scaling freeze is still
  blocked by the spend limit. `FREEZE.md` states both so no one claims "Phase 04 complete."

## Honest Freeze Status

FROZEN for the demo on the live mock backend. The real-vLLM **small gate is measured** (F1 0.94);
the **full freeze is pending** the Modal spend-limit reset. Nothing unverified enters the deck
(SCHEDULE H19 gate); the MFU 0.06 is quarantined as an under-saturated micro-benchmark.
