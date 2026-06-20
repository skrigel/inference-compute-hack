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
