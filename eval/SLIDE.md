# Eval Slide — frozen figures & captions (Phase 05)

> **Honesty discipline (SCHEDULE H19 gate): no unverified number enters the deck.**
> Every figure is labeled with its provenance. The **small quality gate ran on real Modal
> vLLM** (measured, F1 0.94 — Figure 5), but the **full gold-set + ladder freeze is still
> blocked** (Modal spend limit — `eval/artifacts/phase04_modal_blocker.md`). Lead the talk
> with the **interaction** (live on the mock backend); cite the real numbers with provenance,
> and never present the small gate as the full freeze.

## Label legend
- **measured-mock** — deterministic mock scorer, on the pinned cut-line corpus. Real and reproducible, but not a real model.
- **measured-6×H100** — really measured on 6×H100 (Modal vLLM OpenAI server), but a single-query micro-benchmark, **not** the gold eval set.
- **projected** — derived from `performance/theory.py`, not measured.
- **pending** — requires the blocked real-backend run.

---

## Figure 1 — Area under the refine loop (the money shot)
`eval/artifacts/area_under_loop.png` · **measured-mock**

> *Caption:* "Over a 3-turn refine loop (query → click-NOT → AND), candidate-set
> scoping holds cumulative compute at **11 chunks** while full re-score climbs to
> **21** (`k·N`). Scoping flattens; re-scanning compounds."

Source: `eval/artifacts/cut_line_trace.json` (`area_under_loop`). Regenerate: `python -m eval.cut_line --figure`.

## Figure 2 — Fresh data: recompute now vs RAG re-index
**measured-mock** (ours) · structural model (RAG)

> *Caption:* "Drop a file → queryable on the next pass, **0 derived bytes written**.
> RAG must re-embed + rebuild its index before the document is retrievable."

Source: `cut_line_trace.json::fresh_vs_rag`. The toy re-index magnitude is illustrative; the
point is structural (ours has no `D` term).

## Figure 3 — 6×H100 scoring throughput / latency
`eval/artifacts/phase04_rag_vs_6xh100.json` · **measured-6×H100**

> *Caption:* "Single-token semantic scoring on 6×H100 (Modal vLLM, `max_tokens=1`):
> **1,578 req/s · 91k tok/s**, p50 **46 ms** / p95 **182 ms**."

**⚠ MFU caveat (say this; do NOT headline the 6%):** the run's *derived* MFU is **0.06**
(6% of BF16 peak) — but it was an **under-saturated** micro-benchmark: concurrency 16, 64
requests, ~57-token prompts, server queue depth **0**. That measures latency at low load, not
the compute-bound ceiling. The **40–55% prefill MFU** target is *projected* and requires a
high-batch / high-concurrency run with realistic ~350–700-token chunks — **pending**. Present
6% only as "we have not yet saturated the GPUs," never as the architecture's MFU.

## Figure 4 — RAG vs ours crossover (corpus size)
`eval/artifacts/phase04_rag_vs_6xh100.md` · **measured-mock** RAG vs **measured-6×H100** scoring

> *Caption:* "RAG retrieve is cheaper on tiny corpora; our full scan crosses RAG at
> ~10k docs (retrieve) and the fresh-file path wins from ~5k docs — and we never re-index."

**Caveats (state them):** the RAG baseline is the repo's pure-Python hashing-vectorizer
fallback (not tuned FAISS / neural embeddings); retrieve-only (no generation, no rerank, no
semantic continuity across refine turns). Honest framing: we win on **iteration + fresh data**,
not on a single small-corpus lookup.

## Figure 5 — Quality gate (real, small) + threshold calibration
`eval/artifacts/phase04_quality_gate.json` · **measured-6×H100 (small 7-chunk gate)**

> *Caption:* "On real Modal vLLM (`Qwen2.5-3B-Instruct-AWQ`) over the demo corpus:
> **precision 1.0 · recall 0.89 · F1 0.94** — clears the F1 ≥ 0.7 gate (`MIN_F1 = 0.7`,
> `eval/bench.py`). At the **default 0.5** threshold recall collapses to **0.11**: the model
> separates positives at a *low absolute score*, so the operating point uses the **calibrated
> threshold ≈ 0.016**."

**Two honest limits:** (1) this is the *small* gate (7 chunks, scripted predicates) — the
**full gold-set + ladder freeze is pending** the Modal unblock; (2) the calibrated threshold is
backend-specific — the **mock** backend is calibrated for 0.5, the **real vLLM** backend needs
≈0.016. Don't blend them, and don't present the small gate as the full freeze.

---

## What to say if asked "is this live / real?"
"The interaction is live on a deterministic local scorer. We stood up the real vLLM scorer on
6×H100, measured throughput/latency (Figure 3), and ran the quality gate on it — F1 0.94 at a
calibrated threshold (Figure 5). The *full* gold-set + ladder freeze is pending a Modal
spend-limit reset, so I show measured-mock and measured-6×H100 separately and don't blend them."
(Honesty converts the blocker into credibility.)
