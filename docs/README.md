# Project Docs

This directory breaks the build into phase-level operating docs. The top-level
`PLAN.md` remains the narrative plan; these files are the execution map.

## Core docs

- [`../PLAN.md`](../PLAN.md) - complete 24-hour build plan and architecture.
- [`../CONTRACTS.md`](../CONTRACTS.md) - frozen cross-owner interfaces.
- [`../METRICS.md`](../METRICS.md) - performance-oriented metric plan.
- [`../SCHEDULE.md`](../SCHEDULE.md) - phase index and milestone table.
- [`../RISKS.md`](../RISKS.md) - risk register and mitigations.
- [`../DEMO.md`](../DEMO.md) - demo beats, runbook, and fallback ladder.
- [`optimization-results-ledger.md`](optimization-results-ledger.md) - consolidated
  performance baseline, Weave logging contract, and optimization changelog format.
- [`extension-03-agent-loop-rl.md`](extension-03-agent-loop-rl.md) - RL-style
  query-refinement environment, Prime readiness gates, cohorts, metrics,
  no-credit smoke path, and 8-H100 checkpoint policy.

## Phase docs

- [`phases/phase-00-contracts-and-performance-baseline.md`](phases/phase-00-contracts-and-performance-baseline.md)
- [`phases/phase-01-ingest-query-and-dashboard.md`](phases/phase-01-ingest-query-and-dashboard.md)
- [`phases/phase-02-refine-loop-and-fresh-data.md`](phases/phase-02-refine-loop-and-fresh-data.md)
- [`phases/phase-03-cut-line-stabilization.md`](phases/phase-03-cut-line-stabilization.md)
- [`phases/phase-04-real-vllm-performance-freeze.md`](phases/phase-04-real-vllm-performance-freeze.md)
- [`phases/phase-05-demo-lock-and-runbook.md`](phases/phase-05-demo-lock-and-runbook.md)
- [`phases/phase-06-freeze-and-rehearsal.md`](phases/phase-06-freeze-and-rehearsal.md)

## Performance layer

The imported performance package lives in [`../performance/`](../performance/).
It adds closed-form performance models, benchmarking methodology, and figures for
the metrics story. Use it to make eval read as predicted-then-measured rather
than only measured-after-the-fact.
