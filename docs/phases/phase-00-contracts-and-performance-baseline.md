# Phase 00 - Contracts And Performance Baseline

**Window:** H0-H3  
**Milestone:** M0 contracts signed  
**Theme:** make parallel work possible and capture the perishable performance
floor before optimizations hide it.

## Goals

- Reconcile and freeze [`../../CONTRACTS.md`](../../CONTRACTS.md).
- Stand up `ScorerClient`, `MockScorer`, and `make_scorer()` behind the final interface.
- Start the frontend shell on mock data and the backend shell on the same schema.
- Add eval harness scaffolding with performance counters, not just timers.
- On the H100 box, verify vLLM can return Yes/No logprobs and confirm the prefix-cache assumption.

## Owner Work

| Owner | Work |
|---|---|
| A | `inference/scorer.py`, `inference/mock_scorer.py`, `inference/config.py`, `performance/theory.py` hookup, initial `eval/bench.py` skeleton |
| B | `backend/schemas.py`, API stubs, in-memory session state shape |
| C | `frontend/src/lib/types.ts`, mock adapter shell, dashboard frame |
| D | `data/schema.py`, seed predicates, first corpus manifest |

## Performance Metrics To Capture

- `chunks_scored` and `chunks_served_from_cache` counters in the score-cache path.
- Cold full-scan wall time before warm-cache leakage.
- Token counts: real tokens, padded tokens, prefix length, suffix length.
- H100 constants listed in [`../../performance/docs/04_constants_to_verify.md`](../../performance/docs/04_constants_to_verify.md).
- Roofline inputs: achieved TFLOP/s, SM utilization, DRAM bandwidth utilization.

## Exit Gate

- `CONTRACTS.md` has one signed schema for scorer, chunk, SSE, refine ops, and env vars.
- Mock scorer imports from both backend and eval without duplicate local schemas.
- `python performance/theory.py` regenerates figures.
- Eval harness can record a run trace with at least `turn`, `chunks_scored`, `cache_hits`, `elapsed_ms`, and `threshold`.
- H100 smoke test either proves the prefix-cache path or records that candidate-set scoping is the primary performance lever.

## Fallback

If vLLM setup slips, do not block the mock path. Freeze the interface, continue on
`SCORER_BACKEND=mock`, and record the H100 verification as the first Phase 04 task.
