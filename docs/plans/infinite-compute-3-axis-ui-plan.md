# Implementation Plan: Wiring the Infinite-Compute 3-Axis Framework into the UI

Status: **proposed** (plan only — implementation lands in a follow-up PR)
Owner: agent
Date: 2026-06-20
Source: [../theory/infinite-compute-3-axis-framework.md](../theory/infinite-compute-3-axis-framework.md)

## Goal

Make the three architecture decisions from the Infinite-Compute 3-Axis Framework
(Memory, Movement, Truth) **explicit, wired end-to-end, and visibly exhibited in
the UI**. Each axis becomes a real compute dial backed by real computation — not a
mock — surfaced in the existing search dashboard.

Guiding principle from the framework: extra compute should buy more corpus scored
(Memory), a smarter move decision (Movement), and a wider search over predicates
(Truth), all drawing from one global compute budget.

## Current architecture (baseline)

- Backend (FastAPI): `backend/main.py` exposes `/ingest`, `/query`, `/refine`,
  `/clause/{id}`, `/results`, `/healthz`.
  - `/query` streams SSE `result` → `aggregate` → `done` via
    `backend/streaming.py::query_stream`.
  - `/refine` emits `chip` → `diff` → `aggregate` → `done`.
  - `/results` is a pure cache recut (zero inference) — the load-bearing
    "drag = no inference" proof.
  - Scoring: `inference/scorer.py::ScorerClient.score_batch(items, tier)`;
    per-clause `backend/cache.py::ScoreCache`.
- Frontend (React + react-router, default `VITE_DATA_MODE=mock`):
  - `pages/SearchPage.tsx` → `QuerySection`, `ThresholdSection` (histogram brush),
    `FilterSection` (chips + fresh files), `TabbedSection` (results/analytics/facets).
  - Central state: `hooks/useDashboard.ts` over a client-side
    `lib/scoreCache.ts`.
  - Data path: `lib/api.ts` → `lib/liveAdapter.ts` (SSE) or `lib/mockAdapter.ts`
    (24 seeded items). Wire types in `lib/types.ts`.

Because the demo defaults to **mock**, every axis must be implemented in BOTH the
mock adapter and the live backend so the UI exhibits it without a running server.

## Axis 1 — Memory: explicit compute budget → corpus in scope

**Decision to accommodate.** A `compute_budget` parameter that directly controls
the corpus size scored per query (linear scaling made an explicit knob).

- Backend
  - `QueryRequest.compute_budget: float` in `(0, 1]` (default `1.0`).
  - `query_stream(..., compute_budget)` scores only the first
    `ceil(budget × N)` chunks; report `corpus_total`, `corpus_scored`,
    `compute_budget` on `aggregate`/`done`.
- Frontend
  - `QueryRequest.compute_budget` in `lib/types.ts`; thread through both adapters
    (mock slices its item list; live sends the field).
  - `useDashboard`: `computeBudget` state; re-issue `/query` on change.
  - UI: **Compute Budget** slider (10% → 100%) with a "scored X of N chunks"
    readout. Raising it scores more of the corpus.
- Tests: streaming scopes correctly at budgets `{0.25, 0.5, 1.0}`; mock adapter
  honors budget; budget=1.0 unchanged (regression guard).

## Axis 2 — Movement: auto-threshold + smart selection

**Decisions to accommodate.** (A) Auto-threshold to a precision target completes
Mode A. (B) Facet decomposition + cached facet scores + beam search over output
subsets within a movement budget K (max-coverage objective) is Mode B. Both are
**pure arithmetic over cached scores — zero inference.**

- Backend: new `POST /select` (pure cache read), input
  `{ mode: "threshold" | "smart", precision_target, movement_budget K, beam_width B }`.
  - Mode A: pick the cutoff whose selected set has mean P(Yes) ≥ target; return
    `threshold` + selected ids.
  - Mode B: greedy → beam max-coverage selection of ≤ K chunks over the existing
    facet keys; return selected ids, covered facets, objective value, and the
    `(1 − 1/e)` greedy floor.
- Frontend
  - `lib/computeLab.ts`: pure TS mirror (`autoThreshold`, `maxCoverageSelect`) over
    `scoreCache` entries; used directly in mock mode and as an optimistic preview
    in live mode.
  - `useDashboard`: `selection` state (ids + coverage + objective), `autoThreshold()`,
    `smartSelect(K, B)` actions.
  - UI: **Movement** controls — "Auto-threshold (precision target)" slider+apply,
    and "Smart select" toggle with movement budget K and beam width B; selected
    chunks highlighted in the results list with a coverage/objective readout.
- Tests: `computeLab` auto-threshold hits the precision target; greedy ≥ floor and
  beam ≥ greedy; `/select` parity with `computeLab`.

## Axis 3 — Truth: beam width over predicate combinations

**Decision to accommodate.** `beam_width` is the dial between Mode 1/2 (one clause
per turn, human/agent driven) and Mode 3 (N clause combinations tried in parallel,
objective-function selected). Plus an MCP server exposing the endpoints as tools.

- Backend
  - `RefineRequest.beam_width: int` (default `1`). When `> 1`, generate a small
    candidate clause vocabulary (utterance + facet-derived narrowings), score each
    candidate, and select by objective (mean P(Yes) of selected at a min-coverage
    floor). Emit a new `beam` SSE event listing explored candidates + the winner,
    then the usual `chip`/`diff`/`aggregate`/`done` for the winner.
  - MCP server (`backend/mcp_server.py`): expose `query`, `refine`, `results` as
    MCP tools wrapping the existing handlers. Optional dependency, guarded import;
    ships as a runnable module (Mode 2 enablement).
- Frontend
  - `BeamEvent` type; `useDashboard` captures explored candidates.
  - UI: **Beam Width** control on the refine bar; when `> 1`, show the explored
    candidate predicates with their objective scores and highlight the chosen one.
  - An "Agent (MCP)" affordance documents the same endpoints are callable as tools.
- Tests: `beam_width=1` reproduces current single-refine behavior; `beam_width=N`
  selects the max-objective candidate; objective is monotonic in beam width.

## Shared "compute budget" framing

A single **Compute** panel hosts the three dials and shows the global
`compute_budget` split across axes (score more corpus / smarter move / wider
predicate search), matching the framework's "one global budget" rule of thumb.

## Sequencing

1. Backend Axis 1 (`compute_budget`) + schema/stream/tests.
2. Backend Axis 2 (`/select`) + tests.
3. Backend Axis 3 (`beam_width` refine + `beam` event) + tests; MCP module.
4. Frontend types + `computeLab` pure module + tests.
5. Adapters (mock + live) threaded with new params/events.
6. `useDashboard` state + actions.
7. `ComputePanel` UI + selected-set rendering in `SearchPage`.
8. Frontend tests; `vitest` + `vite build`; full backend `pytest`.

## Guardrails

- `/results` and threshold-drag must remain zero-inference (existing tests guard).
- `compute_budget = 1`, `beam_width = 1`, and threshold mode must reproduce current
  behavior exactly (regression guards).
- Mock and live adapters must stay behavior-compatible so the demo exhibits every
  axis offline.
