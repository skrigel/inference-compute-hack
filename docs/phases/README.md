# Phase Plan Index

These files split the build into operating phases. Each phase is written so an
owner can pick it up without re-reading the full plan, but the frozen interfaces
still come from [`../../CONTRACTS.md`](../../CONTRACTS.md).

| Phase | Window | Goal | Exit gate |
|---|---:|---|---|
| [Phase 00](phase-00-contracts-and-performance-baseline.md) | H0-H3 | Freeze contracts, mock scorer, eval skeleton, performance baseline | M0 signed, mock path runnable, cold floor captured |
| [Phase 01](phase-01-ingest-query-and-dashboard.md) | H3-H8 | Ingest/query SSE and dashboard on mock | M1/M2 green, threshold recut is zero inference |
| [Phase 02](phase-02-refine-loop-and-fresh-data.md) | H8-H14 | Refine loop, chip removal, fresh-file ingest | M3 green, canned SSE recorded |
| [Phase 03](phase-03-cut-line-stabilization.md) | H14 | Hard cut line | Core loop works end to end |
| [Phase 04](phase-04-real-vllm-performance-freeze.md) | H14-H19 | Real vLLM swap and freeze measured numbers | M4 green, eval slide numbers frozen |
| [Phase 05](phase-05-demo-lock-and-runbook.md) | H19-H22 | Demo script, preload, operator runbook | M5 locked |
| [Phase 06](phase-06-freeze-and-rehearsal.md) | H22-H24 | Dress rehearsals and repo freeze | Three rehearsals pass |

Performance is tracked in every phase. The key rule is to count inference work
first (`chunks_scored`, cache hits, survivor fractions), then overlay time and
energy once the system path is stable.
