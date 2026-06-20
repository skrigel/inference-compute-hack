# Schedule

The build has one critical path: contracts -> mock scorer -> backend query stream
-> frontend dashboard -> refine loop -> hard cut line -> real scorer swap.

| Phase | Window | Milestone | Primary output |
|---|---:|---|---|
| [Phase 00](docs/phases/phase-00-contracts-and-performance-baseline.md) | H0-H3 | M0 | Frozen contracts, mock scorer, perf counters, cold floor |
| [Phase 01](docs/phases/phase-01-ingest-query-and-dashboard.md) | H3-H8 | M1/M2 | Ingest/query SSE and live dashboard on mock |
| [Phase 02](docs/phases/phase-02-refine-loop-and-fresh-data.md) | H8-H14 | M3 | Refine loop, chip removal, fresh-file ingest |
| [Phase 03](docs/phases/phase-03-cut-line-stabilization.md) | H14 | Cut line | Irreducible loop stabilized |
| [Phase 04](docs/phases/phase-04-real-vllm-performance-freeze.md) | H14-H19 | M4 | Real vLLM numbers and frozen eval artifacts |
| [Phase 05](docs/phases/phase-05-demo-lock-and-runbook.md) | H19-H22 | M5 | Demo script, preload runbook, fallback ladder |
| [Phase 06](docs/phases/phase-06-freeze-and-rehearsal.md) | H22-H24 | Freeze | Three rehearsals, repo/data/metric freeze |

## Hard Gates

- **H1:** contract changes require delivery-owner sign-off.
- **H8:** dashboard must be demoable on mock even if backend integration slips.
- **H14:** stop expanding scope unless ingest -> query -> refine -> threshold -> fresh-file is green.
- **H19:** freeze the eval numbers; no unverified numbers enter the deck.
- **H22:** demo path is locked; only rehearsed fixes allowed.

## Parallel Tracks

- A: inference, performance theory, eval, RAG baseline.
- B: backend contracts, state, clause engine, SSE.
- C: frontend dashboard, refine loop, latency readout, replay fallback.
- D: corpus, labels, predicates, demo polish.
