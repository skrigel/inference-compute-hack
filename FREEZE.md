# FREEZE — Phase 06 (H22–H24)

**Status: FROZEN for the demo** on the live mock backend; the **full real-vLLM freeze is pending**
the Modal spend-limit reset (the *small* real gate is measured — see below).

Machine-readable snapshot: `eval/artifacts/freeze_manifest.json` (regenerate with
`python -m eval.rehearsal`). The numbers below are read from saved artifacts and are reproducible.

## Provenance
- **commit:** `3fd0419` (+ this freeze commit)
- **demo scorer (primary path):** `mock-semantic-filter-v0` (deterministic, GPU-free)
- **real scorer (measured gate):** `Qwen/Qwen2.5-3B-Instruct-AWQ` on Modal vLLM (6×H100)
- **corpus:** `demo`, **7 chunks** (pinned, `backend/state.py::demo_chunks`)

## Frozen metrics (each labeled)

| metric | value | label | source artifact |
|---|---|---|---|
| Area under refine loop (scoped) | **11** chunks | measured-mock | `cut_line_trace.json` |
| Area under refine loop (full re-score) | **21** chunks | measured-mock | `cut_line_trace.json` |
| Quality gate — F1 | **0.94** (P 1.0 / R 0.89) | **measured** (real Modal vLLM, *small* 7-chunk gate) | `phase04_quality_gate.json` |
| Quality gate — operating threshold | **≈ 0.016** (default 0.5 → recall 0.11) | measured | `phase04_quality_gate.json` |
| 6×H100 scoring throughput | **1,578 req/s · 91k tok/s** | measured-6×H100 | `phase04_rag_vs_6xh100.json` |
| 6×H100 latency | p50 **46 ms** / p95 **182 ms** | measured-6×H100 | `phase04_rag_vs_6xh100.json` |
| 6×H100 derived MFU | **0.06** | measured-6×H100 (**under-saturated micro-benchmark**) | `phase04_rag_vs_6xh100.json` |
| Prefill MFU target | 40–55% | **projected** (`performance/theory.py`) | — pending high-batch run |

## Freeze checklist (phase-06)

- [x] `CONTRACTS.md` matches implementation; no unapproved schema drift (frozen types import; mock-parity clarified in Phase 03 changelog).
- [x] `PLAN.md` / `SCHEDULE.md` / phase docs / `METRICS.md` / `DEMO.md` / `RISKS.md` agree on milestones and on the F1 ≥ 0.7 gate.
- [x] `performance/theory.py` regenerates `performance/figures/` (`python performance/theory.py`).
- [x] Slide figures labeled measured / measured-mock / measured-6×H100 / projected / pending (`eval/SLIDE.md`).
- [x] Fallback fixtures present and bootable (`cut_line_query/refine/fresh.sse`; `tests/test_phase5_demo_lock.py`, `tests/test_phase6_freeze.py`).
- [x] Operator can recover from failed backend / scorer / network in seconds: flip `VITE_API_BASE` to the replay server (CORS-enabled, instant).

## Three rehearsals (`python -m eval.rehearsal`)

- **R1 live path** — cut-line loop GREEN on the mock backend (ingest → query → click-NOT → AND → threshold → fresh-file).
- **R2 injected failure → replay** — with the live path "down", the canned replay serves beats 1 / 2-3 / 5 (fresh toggle armed).
- **R3 timed ≤ 90s** — spoken-beat budget **85s** (`eval/rehearsal.py::BEAT_BUDGET_S`, mirrors `DEMO.md`).

## Caveats (state them on stage / in Q&A)
1. **Quality gate is the SMALL gate** (7 chunks, scripted predicates), measured on real Modal vLLM (F1 0.94 @ calibrated ≈0.016). **Default 0.5 collapses recall (~0.11)** — the real backend must use the calibrated threshold. Full gold-set + ladder freeze is **pending** the Modal unblock.
2. **Derived MFU 0.06 is an under-saturated micro-benchmark**, not the architecture's MFU; the 40–55% prefill target is *projected* and needs a high-batch run.
3. **Warm-cache KV crossover ~14k chunks (FP16)** is projected; candidate-set scoping carries the refine path past it.
4. **Canned replay twins are recorded from the mock backend**; re-record from a real vLLM run once unblocked, then state "recorded from the same scorer path."
5. **Frontend `dist/` must be built ahead of time on Node ≥ 20.19** (Vite 8); the venue box may not `npm install`.

## Reconciliation: the Modal blocker vs. the measured gate
`phase04_modal_blocker.md` says real validation is blocked by the spend limit; `phase04_quality_gate.json`
is nonetheless a **measured real-vLLM gate**. These are consistent: the *small* 7-chunk gate completed on
Modal before the limit was hit; the *full* ladder/scaling sweep and large gold-set freeze remain blocked.
Do not claim "Phase 04 complete" until the full gate/freeze regenerates from a real run.

## To upgrade this freeze (single unblock)
Reset the Modal spend limit (or use the on-prem box), then:
```bash
SCORER_BACKEND=modal python -m eval.bench --backend modal --gate-only --weave
SCORER_BACKEND=modal python -m eval.bench --backend modal --tag freeze --weave
python -m scripts.replay_sse record    # re-record canned twins from the real scorer path
python -m eval.rehearsal               # regenerate freeze_manifest.json
```
