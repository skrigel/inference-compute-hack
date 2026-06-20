# Demo Runbook

Target length: 90 seconds. The demo sells interaction speed, not a static search
result. Every beat should either change the candidate set, re-cut cached scores,
or introduce fresh data.

## Primary Path

1. **Open on a seeded query.** Predicate: "every place we retry a network call without backoff".
   Results stream best-first while the histogram and facets update.
2. **Click-NOT a wrong match.** A removable Exclude chip appears first, then the
   result disappears and aggregates reflow.
3. **AND refine.** Type "only in the networking layer". A Require chip appears,
   candidate-set scoping re-scores survivors, and the latency readout shows the turn.
4. **Drag threshold.** Histogram and feed recut instantly. Narration: no new model
   call, just cached score recut.
5. **Drop fresh file.** New chunks become queryable immediately. Narration: RAG
   would re-index; this path recomputes over raw text.
6. **Close on performance.** Show area-under-loop and roofline. The claim is
   predicted-then-measured performance, not only a dashboard timer.

## Optional Closers

- Quality close: threshold drag is moving along a measured ROC curve.
- Compute close: scoped refine approaches `N/(1-rho)` cumulative work.
- Fresh-data close: compute-vs-churn break-even explains when recompute beats RAG.

## Fallback Ladder

1. Real vLLM backend.
2. Mock backend with the same contracts.
3. Canned SSE replay recorded from a real vLLM run.
4. Canned SSE replay recorded from mock.
5. Manual staged loop with preloaded corpus and explicit operation buttons.

## Operator Checks

- `SCORER_BACKEND` is set to the intended backend.
- `VITE_DATA_MODE` is `live` for backend demo or `mock` for standalone fallback.
- Seed corpus is ingested and warmed or the replay server is ready.
- `METRICS.md` numbers match the slide captions.
- Figures are labeled `measured`, `predicted`, or `projected`.

## Do Not Demo

- A single static query over a frozen corpus.
- A speed number that has not passed the quality gate.
- 4-bit weight quantization as a raw scan-throughput multiplier.
- Stretch features before the H14 loop is stable.
