# Demo Runbook

Target length: 90 seconds. The demo sells **interaction speed**, not a static
search result. Every beat changes the candidate set, re-cuts cached scores, or
introduces fresh data — never a single frozen lookup.

> The numbers below are **measured** from the pinned cut-line corpus on the mock
> scorer (`eval/artifacts/cut_line_trace.json`, regenerate with
> `python -m eval.cut_line --figure`). Real latencies come from the Phase 04 box
> run; the *compute counts* (scored / matched) are deterministic and pinned here.

## Pinned demo corpus (7 chunks, `backend/state.py::demo_chunks`)

| # | chunk | retry? | networking layer? | role in the demo |
|---|---|---|---|---|
| 1 | `urllib3/connectionpool.py` | yes (no backoff) | yes | strong match; survives query + AND |
| 2 | `requests/adapters.py` | yes (no backoff) | yes | strong match; survives query + AND |
| 3 | `aiohttp/client.py` | yes (no backoff) | yes | strong match; survives query + AND |
| 4 | `app/db_session.py` | yes (has backoff) | **no** | query survivor; **drops on the AND** |
| 5 | `jobs/worker.py` | yes (no backoff) | **no** | query survivor; the **click-NOT** target |
| 6 | `Neural Retrieval for Code Search` (paper) | no | no | non-match (histogram spread); the IR-sense closer |
| 7 | `demo/ui.ts` | no | no | non-match (histogram spread) |

## Primary Path (measured: scanned / matched per beat)

1. **Seeded query — "every place we retry a network call without backoff".**
   Streams best-first; histogram + paper/code facet fill. **7 scanned · 5 matched ·
   0 index built.** (chunks 1–5 match; the 3 networking-layer retries + the 2
   non-network retries.)
2. **Click-NOT a wrong match.** Click *"not like this"* on the lowest-ranked
   survivor — a **non-network retry** (`jobs/worker.py`). A removable **Exclude**
   chip appears first, the card drops, aggregates reflow. **0 inference · 4 matched.**
3. **AND refine — "only in the networking layer".** A **Require** chip appears;
   candidate-set scoping re-scores **only the 4 current survivors (not the corpus)**;
   the non-network retry (`app/db_session.py`) drops. **4 scored · 3 matched.**
4. **Drag the threshold.** Histogram + feed re-cut instantly from the cached
   scores. **0 inference** (proven: `GET /results` and the `brush` op both score 0).
   Narration: *"zero new inference — scores were computed once."*
5. **Drop a fresh file** (`fresh_incident.py` — a network retry with no backoff).
   Queryable on the very next query. **0 derived bytes written**; RAG would have to
   re-embed + rebuild its index first. **8 scanned · fresh chunk in the matches.**
6. **Close on performance.** Show `eval/artifacts/area_under_loop.png`: over the
   3-turn refine loop (query → click-NOT → AND) our **scoped** cumulative compute
   is **11 chunks** vs **21** for full re-score — it flattens while full climbs
   `k·N`. (The fresh-data / no-re-index win is beat 5 + the fresh-vs-RAG proof.)
   The claim is predicted-then-measured, not a dashboard timer.

## Optional Closers

- **IR-sense recovery:** refocus *"retrieval in the IR sense, not RAG"* recovers
  the `Neural Retrieval` paper the first pass dropped (most differentiated beat).
- **Compute close:** scoped refine approaches `N/(1-ρ)` cumulative work.
- **Fresh-data close:** compute-vs-churn break-even explains when recompute beats RAG.

## Operator Runbook (who clicks what next)

| Beat | Driver action | Expected on screen |
|---|---|---|
| 1 | App opens on the seed query (auto-run) | 5 matched, histogram fills, paper/code facet |
| 2 | Click **drop** on the lowest survivor card | Exclude chip; 4 matched; instant reflow |
| 3 | Type `only in the networking layer` → Enter | Require chip; 3 matched; db/job rows gone |
| 4 | Drag the on-histogram threshold handle | bars recolor; matched count moves; latency tag = `cached` |
| 5 | Drag `fresh_incident.py` onto the surface | corpus +1; re-query; fresh row appears |
| 6 | Switch to the performance figure | area-under-loop: scoped flat, full/RAG climb |

## Fallback Ladder (tested)

1. **Real vLLM backend** (Phase 04; `SCORER_BACKEND=vllm`).
2. **Mock backend** — same contracts, `SCORER_BACKEND=mock` (this is what the cut-line trace runs on).
3. **Canned SSE replay** — `python -m scripts.replay_sse serve --port 8090`, then point
   `VITE_API_BASE` at it. Serves `eval/artifacts/cut_line_query.sse` + `cut_line_refine.sse`
   byte-for-byte; the frontend can't tell it from the live backend (proven by
   `tests/test_phase3_replay.py`).
4. **Pure-frontend mock** — `VITE_DATA_MODE=mock` (no backend at all).
5. **Manual staged loop** — preloaded corpus + explicit op buttons.

## Preflight & Operator Checks

- Run `bash scripts/preload_demo.sh` → must print **GO ✓** (drives the full loop + arms replay).
- `SCORER_BACKEND` is the intended backend; `VITE_DATA_MODE`/`VITE_API_BASE` point where intended.
- **Frontend build needs Node ≥ 20.19** (Vite 8 / vitest 4). Build `dist/` ahead of time on a
  Node-20 machine and serve the static build; do not assume the venue box can `npm install`.
- `METRICS.md` / slide numbers match the trace; every figure labeled `measured`, `predicted`, or `projected`.

## Do Not Demo

- A single static query over a frozen corpus.
- A speed number that has not passed the quality gate.
- 4-bit weight quantization as a raw scan-throughput multiplier (it's a capacity lever; FP8 is throughput).
- Stretch features (Tier-2 cascade, second domain, editable chip algebra) — the H14 loop is the line.
