# Grep for Meaning — Build Plan

> A detailed, comprehensive plan for the 24-hour, 4-person, 8×H100 hackathon build.
> Derived from the FINAL MVP spec, six parallel subsystem designs, and three adversarial design
> reviews (feasibility, demo-narrative, contract-integrity). Read this top-to-bottom once; then
> live in [`CONTRACTS.md`](CONTRACTS.md), [`SCHEDULE.md`](SCHEDULE.md),
> [`METRICS.md`](METRICS.md), [`docs/phases/`](docs/phases/), and [`DEMO.md`](DEMO.md).

---

## 1. What we are building (one screen)

An **interactive semantic filter** that runs natural-language queries directly over raw text — no
index, no embeddings, no schema. Type a predicate, watch matches stream best-first into a live
dashboard, refine in plain language, converge in a handful of sub-second turns.

> **grep, but the predicate is meaning, and you steer it live.**

A single web surface with three live regions:

* **Query bar** — type a predicate in plain English; it scans the raw corpus directly.
* **Live dashboard (the hero visual)** — a score **histogram** with a draggable **on-histogram
  threshold** handle, plus **facet bars** (paper vs. code, category, year). It reflows on every
  refinement. Tabs: **Relevance** (default) and **Performance**. *(The middle **Footprint** tab is
  deferred — see [§10](#10-scope-explicitly-cut--deferred).)*
* **Results feed + refine box** — ranked cards stream best-first with one-line rationales; refine in
  plain language and each change lands as a visible, removable **chip**.

The felt experience is the pitch: *I can iterate as fast as I can think.* Refinement and
re-thresholding cost **zero new inference** — every chunk's relevance score is computed once and
cached per clause.

---

## 2. The bet (for judges)

Every retrieval stack today — RAG, GraphRAG, hybrid search, rerankers, vector DBs — exists because
inference is expensive, so engineers spend their lives curating context. **As inference gets cheap,
that whole layer collapses into one primitive:** a semantic filter that reads a raw chunk and emits a
single Yes/No token whose logprob is a continuous relevance score.

We built the live, interactive surface for that world, and made it fast on 8×H100 with single-token
scoring, FP8 prefill compute, 4-bit weight/KV capacity, data-parallel replicas, warm-cache prefix
reuse, and candidate-set-scoped refinement.

**Where latency is the un-fakeable hero** (and RAG structurally can't follow):
1. **Iterative refinement** — every refine turn is a fresh sub-second pass for us (cached), a full
   re-retrieve for RAG. Latency compounds on their side.
2. **Fresh data mid-session** — drag in a new file and query it instantly; RAG must re-index first.

We **never** demo a single frozen lookup over a static corpus — that is the one comparison we lose.

**The forward thesis (spoken pitch only, not a build change):** push latency → 0 and the
compute/storage tradeoff inverts. When inference is ~free, storage/memory is the binding constraint
and the optimal move is **recompute over store** — persist nothing derived, regenerate on demand. Our
build *does* cache for latency today; the trajectory is that it can drop the cache as inference
cheapens. The one-liner that pre-empts the contradiction: *"Today it caches to be fast; the payoff is
it can drop the cache as inference gets cheap — here's the footprint math."*

---

## 3. The most important thing: freeze the contracts in hour zero

All three design reviews independently found the **same** dominant risk: the six subsystem drafts each
wrote a *different* version of the "frozen" interface — **four** incompatible `ScorerClient`
signatures, **three** chunk schemas, **two** histogram shapes, **three** chip-op vocabularies, **four**
env-var names, and a `clause_id` concept that didn't exist in half the plans. "Build against the mock
in parallel" is a lie if everyone mocks a different schema; the break only surfaces at first
integration (~H5) and burns a half-day.

**Action (H0–1, before anyone writes feature code):** the delivery owner runs a 60–90 minute
contract huddle and produces ONE [`CONTRACTS.md`](CONTRACTS.md), signed off by A/B/C/D. The reconciled
decisions (already captured in that file):

* **One `ScorerClient`** in `inference/scorer.py`: `score_batch(items: list[ScoreRequest], *, tier=1)
  -> list[ScoreResult]` + `warm(corpus_id, chunks) -> PrefixState`. Returns **rich `ScoreResult`**, not
  bare floats. The scorer is **clause-agnostic**.
* **One `Chunk`** in `data/schema.py`. Modality = `chunk.type` (`paper|code`); provenance =
  `chunk.meta.source`; identity = `chunk_id` (one hash function, text in the hash).
* **One SSE schema**: `result` / `aggregate` / `diff` / `done`, multiplexed over a single stream by
  `.type`. Histogram = `{lo,hi,count}` × **20 bins everywhere**. Facets = `FacetBucket[]
  {key,relevant,total}` (backend computes *both* relevant and total).
* **One refine transport**: `POST /refine` returns a single `text/event-stream`; first event is the
  chip, then diff/aggregate/done. No `/refine/stream/{turn_id}`.
* **One chip-op enum**: `require|exclude|include|refocus|brush`; classifier and `classify.ts` emit
  identical tokens (contract-tested). Click key is `sign`.
* **One `clause_id` owner** (backend mints; scorer is clause-agnostic). **SSE `result.score` is the
  final combined relevance** so the client-side threshold recut provably matches the server.
* **One env-var name each**: `SCORER_BACKEND`, `VLLM_REPLICAS`, `VITE_DATA_MODE`, `VITE_API_BASE`.
* **One MockScorer** (`inference/mock_scorer.py`), imported by backend and eval.

This single hour is the highest-leverage time in the whole 24.

---

## 4. Architecture & repo layout

```mermaid
flowchart LR
  subgraph Frontend["frontend/  (React+Vite+TypeScript) — Owner C"]
    UI[Dashboard: Histogram+threshold · FacetBars · ResultFeed · RefineBox · ChipRail · LatencyReadout]
  end
  subgraph Backend["backend/  (FastAPI async) — Owner B"]
    API[/ingest /query /refine /results/]
    CL[clause.py · candidate-set scoping]
    CA[cache.py · (chunk_id,clause_id)→score]
    AG[aggregate.py · histogram + facets]
    CF[classifier.py · NL→op]
  end
  subgraph Inference["inference/ — Owner A"]
    SC[ScorerClient]
    MOCK[MockScorer · GPU-free]
    VLLM[VLLMScorer · 6× data-parallel AWQ replicas on 8×H100]
  end
  Data[(data/ corpus.jsonl + labels — Owner D)]
  Eval[eval/ bench + baseline/ RAG — Owner A/D]

  UI <-->|single multiplexed SSE| API
  API --> CL --> CA
  API --> AG
  API --> CF
  API -->|make_scorer SCORER_BACKEND| SC
  SC --- MOCK
  SC --- VLLM
  Data --> API
  Data --> Eval
  Eval --> SC
```

**Stack:** vLLM target (`enable_prefix_caching=True, max_tokens=1, logprobs=20`, continuous batching) ·
FastAPI + async, single multiplexed SSE stream · React 19 + Vite 8 + TypeScript + plain CSS, capped
best-first feed, refine chips, and fresh-file ingest · raw chunks + score cache **in memory, no DB** ·
standard embeddings+FAISS RAG in `baseline/`, **eval only**.

```
grep-for-meaning/
  inference/     # vLLM serving, scoring client, prompts                 [OWNER A]
    scorer.py        # FROZEN: ScorerClient, ScoreRequest, ScoreResult, PrefixState + score math
    prompt.py        # FROZEN: exact prefix/suffix template (cache-load-bearing)
    mock_scorer.py   # THE deterministic GPU-free scorer (imported by backend + eval)
    vllm_client.py   # real async client to N replicas
    warm.py          # pre-prefill prefixes at ingest + KV-footprint estimate
    config.py        # SCORER_BACKEND swap point, make_scorer()
    serve.sh         # 6 single-GPU AWQ replicas (+ reserved Tier-2)
  backend/       # FastAPI                                               [OWNER B]
    main.py          # /ingest /query /refine /results — single multiplexed SSE
    schemas.py       # pydantic wire models (import inference + data types)
    state.py         # in-memory SessionState (chunks, clause tree, caches)
    chunker.py       # raw → chunks (imports data/schema.chunk_id_of)
    clause.py        # require/exclude/include/refocus + candidate-set scoping + combine
    cache.py         # (chunk_id, clause_id) → score
    aggregate.py     # running histogram (20 bins) + FacetBucket[] counts
    classifier.py    # NL utterance → wire op (rules-first, LLM fallback)
    streaming.py     # batching, best-first buffer, ETA, backpressure
  frontend/      # React + Vite                                          [OWNER C]
    src/lib/types.ts # TS mirror of CONTRACTS.md
    src/lib/{sse,api,mockAdapter,liveAdapter,scoreCache,classify}.ts
    src/store/useAppStore.ts
    src/components/{Dashboard,Counters,Histogram,FacetBars,ResultFeed,ResultCard,
                    RefineBox,ChipRail,LatencyReadout,Header, tabs/*}.tsx
  baseline/      # RAG pipeline — EVAL ONLY                              [OWNER A/D]
    rag.py           # embed + FAISS top-k; logs index-build & retrieve cost (+ numpy fallback)
  eval/          # ground truth + latency + scaling study                [OWNER A/D]
    bench.py         # validate_score gate → ladder sweeps → artifacts
    metrics.py · sessions.py · gold.py · plots.py · config.py
  performance/   # theoretical compute layer + figures                   [OWNER A/D]
    theory.py        # closed-form roofline, MFU, scoped-loop, churn, KV models
    docs/            # performance thesis, methodology, constants to verify
    figures/         # generated chart artifacts for slides
  data/          # mixed papers+code (10–20k) + labels                  [OWNER D]
    schema.py        # FROZEN Chunk/ChunkMeta/chunk_id_of
    build.py · fetch_arxiv.py · fetch_code.py · chunker.py · make_labels.py
    fetch_browsecomp.py · synthetic.py · predicates.yaml · questionnaire.yaml
  scripts/
    preload_demo.sh  # boot + ingest + warm + health-check → GO/NO-GO
    replay_sse.py    # record/replay real SSE for the universal demo fallback
  CONTRACTS.md · PLAN.md · README.md · METRICS.md · DEMO.md · SCHEDULE.md · RISKS.md
  docs/phases/phase-00...phase-06.md
```

**Team split.** A = inference + warm-cache + eval. B = backend (clause + aggregates + cache +
classifier). C = frontend (dashboard + refine loop + latency readout). D = data/corpus + labels +
demo polish (floating).

---

## 5. The technical spine (latency-first, with the review corrections folded in)

Ordered by leverage. The originals are from the spec; the **⚠ corrections** are what the reviews +
the [`performance/`](performance/) analysis caught.

**Organizing insight (from `performance/`):** the workload is almost pure **prefill** with a one-token
decode, so it lives on the **compute roofline** (high arithmetic intensity, tensor-core-saturating) —
*not* the memory-bandwidth wall that generation/decode serving sits on. This is *why* data-parallel
replicas + FP8 + large batches are the right levers, and *why* recompute-over-store is viable (we are
compute-bound with bandwidth headroom). `theory.py` predicts the cold scan at ≈36 s @ 40 % MFU — inside
the spec's 30–60 s window — so the math closes with the spec's own target.

1. **Score = single-token logprob.** Per chunk: `[prompt] → constrained Yes/No token`;
   `score = P(Yes)/(P(Yes)+P(No))`, continuous 0–1. Threshold re-cuts cached scores with **no
   re-inference**. Build the Yes/No token-id sets at startup by tokenizing every surface form
   (`Yes/yes/␣Yes/▁Yes`, same for No) and sum prob mass over `logprobs=20`; 1e-6 smoothing → 0.5 when
   neither appears.

2. **Prompt order is load-bearing.** `[instruction + chunk]` (stable, cached prefix) then `[predicate]`
   (short, changing suffix, ≤ ~40 tokens). Refining re-prefills only the suffix per chunk and reuses
   each chunk's cached KV.
   **⚠ Verify the assumption in the FIRST box session:** confirm vLLM's block-hash actually keys on the
   full token prefix *including* chat-template system tokens (assert two different predicates share an
   identical tokenized prefix). If it misses, suffix-only re-prefill silently goes cold and the refine
   path falls to mechanism #4 (which is fine for correctness; the pitch headline shifts to the
   scoped-vs-full ratio). This is a 30-minute check — do it at hour ~1 on the box, not at hour 20.

3. **Candidate-set scoping is the PRIMARY refine mechanism.**
   **⚠ Promoted from "robust fallback" to primary** (all three reviews). AND/`require` and `exclude`
   score the new clause over **current survivors only**; `include` over the complement; `refocus`
   re-scores over the parent set. It needs only the cheap `(chunk_id, clause_id) → float` cache, is
   independent of KV residency, and has none of the warm-cache pinning contradictions. Make it the
   refine path from H8.

4. **Warm-on-ingest is a measured FIRST-QUERY bonus, not a load-bearing dependency.**
   **⚠ Demoted** (reviews). Pre-prefill every chunk's prefix at load so the first query is warm.
   *Resolve the pin-vs-round-robin contradiction explicitly:* each chunk is pinned to a replica by
   stable hash for **both** warm and scoring, so warm KV is actually reused; a full scan still spreads
   ~evenly across 6 replicas (statistical balance) for throughput. A small skewed candidate set loses
   perfect balance, but it's small so latency is fine. Measure the KV crossover in Phase 0; past it,
   #3 carries the refine path.
   **⚠ KV-fit correction (`performance/theory.py`, verified locally):** the spec's "10–20k warm in
   640 GB" is only half-true. FP16 warm-KV (128 KiB/token × ~350 prefix tokens) crosses the 640 GB
   node budget at **~14k chunks** — so at the top of the 10–20k range **quantized KV is required, not
   optional**. fp8 KV (vLLM `--kv-cache-dtype fp8`) fits 20k at ~459 GB (crossover ~28k); 4-bit KV
   pushes it to ~56k. Own this on our own slide (figure `5_kv_capacity`) instead of letting a judge
   catch the bare 640 GB claim. The predicted ~14k crossover is then *confirmed* against the real OOM
   point in the scaling sweep — and the gap (PagedAttention + shared-prefix dedup) is itself a finding.

5. **FP8 compute + 4-bit capacity + DATA-parallel replicas (not tensor-parallel).** FP8 is the prefill
   throughput lever because the scan is compute-bound. AWQ/4-bit weights are the capacity lever: they
   reduce memory traffic and make the filter/warm cache easier to fit, but they are not a 4× raw scan
   speedup in this regime. Tier-1 (~3–8B) fits on one 80GB H100, so run **6 fully independent
   single-GPU replicas** (no NCCL chatter) ≈ 6× throughput. Tensor-parallel is reserved **only** for
   the 32B Tier-2 (TP=2 across 2 GPUs).

6. **Stretch — cost-based cascade.** Tiny Tier-1 scores everything; only the uncertain band
   (`0.4–0.6`) escalates to Tier-2 → big-model quality at small-model cost, doubling as a decision
   heatmap. **Off by default, behind a flag**; cut hard if the box arrives late.

**Models.** Tier-1 = `Qwen/Qwen2.5-3B-Instruct-AWQ` (prebuilt 4-bit, strong small-model instruction
following). Fallback if mis-calibrated: `meta-llama/Llama-3.1-8B-Instruct-AWQ` (one-line swap in
`config.py`). Tier-2 (stretch) = `Qwen/Qwen2.5-32B-Instruct-AWQ` (TP=2). Don't burn H0–3 on model
selection — stand up whatever serves `logprobs=20` fastest, validate the score, then swap.

**⚠ Chunk size vs context window (review fix).** `max-model-len = 4096`. Keep prefixes ≤ ~512 tokens:
papers chunk at abstract level (~2000 chars), **code chunks capped at ~2500 chars (~600 tokens), NOT
40k**. Oversized code files are split or dropped, never truncated silently. This reconciles the
"~500–700 token prefix", the warm-KV budget (now corrected: quantized KV above ~14k chunks, #4), and
the "context window" claims that contradicted each other across the drafts.

---

## 6. Subsystem plans (condensed; full file lists in each owner's section of `CONTRACTS.md`)

### Owner A — `inference/` + `eval/` + `baseline/`
* Freeze `scorer.py` + `prompt.py` (prefix/suffix split) at H1; ship `mock_scorer.py` + `make_scorer()`
  so everyone unblocks.
* `vllm_client.py`: async round-robin to 6 replicas, `guided_choice=['Yes','No']` (drop if it hurts
  p50 — surface-form math works without it), surface-form logprob aggregation.
* `warm.py`: prefill-only passes, `estimate_kv_bytes()`, `WarmReport.crossover_flag`.
* `serve.sh`: 6× `CUDA_VISIBLE_DEVICES=$i vllm … --quantization awq_marlin --enable-prefix-caching
  --max-model-len 4096 --max-num-seqs 256`; Tier-2 commented/stretch.
* `eval/` + `performance/`: `validate_score` gate (hard STOP if F1 < 0.7 unless `--force`), ladder
  sweeps via config flags, cache counters, counterfactual replay, measured-vs-predicted overlays
  against `performance/theory.py`, the iteration-cost anchor chart, artifacts (`results.jsonl`,
  `metrics.json`, charts).
* `baseline/rag.py`: sentence-transformers + FAISS on the box; **numpy hashing-vectorizer + matmul
  fallback** so it imports on the Mac. Logs index-build + retrieve cost for "RAG: minutes, ours: 0".

### Owner B — `backend/`
* Freeze `schemas.py` (import inference + data types) at H1.
* `state.py` + `cache.py` + `/ingest` (chunk count + facets, kick warm) — H3–8.
* `main.py` `/query` single multiplexed SSE + `streaming.py` (batch-size knob, 64 default for visible
  progress, best-first reorder window, EMA ETA; add queue/coalescing backpressure when scaling beyond
  demo corpus size) — H3–8.
* `aggregate.py`: 20-bin histogram + `FacetBucket[]` with **both** relevant and total — H3–8.
* `clause.py`: **build the four ops the demo needs** — `require`/`exclude` over survivors, chip
  removal recompute, one `refocus`. **Skip `include`-over-complement and rewrite-parent algebra unless
  time remains after H14.** Ship `REFINE_MODE=full` as the correctness fallback.
* `classifier.py`: rules-first, safest default (`require`, low confidence) on ambiguity — chips are the
  safety net. Contract-test against `classify.ts`.
* `/refine` (classify → apply_op → score only `cache.missing` → chip-first SSE → diff + `refine_ms`),
  `DELETE /clause` (zero inference), fresh-file ingest path — H8–14.

### Owner C — `frontend/`
* Scaffold Vite+React+TS; port the demo palette/fonts into plain CSS tokens; freeze `lib/types.ts`
  against `CONTRACTS.md`; stub mock+live adapters with identical signatures — H0–3.
* `mockAdapter.ts`: fixture-backed contract stream + fake single-stream SSE timing
  (cold ~820 ms, warm ~180 ms, cached ~6 ms) so the UI is fully demoable with **no backend** — H0–3.
* `scoreCache.ts` + hook-local state + `streamPost()` (fetch + ReadableStream; **not** EventSource,
  because /query and /refine are POSTs); `liveAdapter` auto-falls-back to mock on network error — H3–8.
* Histogram with **client-side, zero-inference threshold recut** (unit test spies the adapter and fails
  if a network call fires on drag), FacetBars, Counters, capped ResultFeed —
  H3–8. **Demo-able by H8.**
* RefineBox + ChipRail (optimistic removable chips), keep/drop → `/refine` click, LatencyReadout
  (cold/warm/cached tag + sparkline), Header drag-in fresh-file → `/ingest` → auto re-run — H8–14.
* **Own the click-to-pixel budget**: keep *perceived* refine latency (click → chip → diff → histogram
  reflow) under ~300 ms via optimistic chips, independent of backend `refine_ms`.
* **Footprint tab = disabled stub** (no fields added to the aggregate contract).

### Owner D — `data/` + labels + demo polish (floating)
* Freeze `schema.py` at H1; ship `fetch_browsecomp.py` slice (+ arXiv-topic-gold no-token fallback)
  so the score gate is one command — H0–3.
* `fetch_arxiv.py` (Cornell snapshot, abstracts-only, no PDF parsing) + `fetch_code.py` (pinned SHAs of
  retry/HTTP/networking repos: `requests`, `urllib3`, `httpx`, `aiohttp`, `tenacity`, gRPC/k8s subsets —
  chosen so the headline predicates land) → `build.py` writes deterministic `corpus.jsonl` + `facets.json`
  + `manifest.json` (~18k chunks) — H0–8.
* `predicates.yaml` (10 demo/eval predicates incl. retry-without-backoff, networking-layer,
  IR-sense-of-retrieval) + `questionnaire.yaml` exact-answer gold — H8–14.
* `synthetic.py` planted-ground-truth fallback so the demo runs fully offline — H8–14.
* Curate the demo corpus subset so every scripted beat has a known-good match (and a known wrong match
  with a rationale for the click-NOT beat).

---

## 7. The 24-hour schedule

Integration milestones **M0–M5** are "connect X to Y" checkpoints. Everything is built **mock-first**;
the real-vLLM swap is additive and never on the demo critical path.

| Window | Goal | Milestone |
|---|---|---|
| **H0–3** | **Freeze `CONTRACTS.md`** (M0). Scaffold repo + `make` boot target. One `MockScorer` behind the frozen interface. `score.py`/`prompt.py` frozen. Stand up Qwen-3B-AWQ on the box + verify `logprobs=20` **and** the prefix-cache-hit assumption (30-min check). In parallel: RAG baseline + eval harness skeleton + BrowseComp slice ready. Frontend shell demoable on mock. | **M0**: contracts signed |
| **H3–8** | Backend `/ingest` + `/query` single multiplexed SSE against mock (**M1**). Frontend `streamPost` consuming it (**M2**): histogram + draggable threshold (client-side zero-inference recut) + facet bars + capped best-first feed + ETA. RAG index-build/retrieve timed. **Score-validation F1 gate on the box the moment vLLM serves logprobs** — GO/NO-GO; if F1<0.7 swap to Llama-3.1-8B-AWQ. | **M1, M2** |
| **H8–14** | Refine loop end-to-end (**M3**): NL→chip→scoped re-score→diff, click-NOT, chip removal, `refine_ms` in LatencyReadout. Fresh-file drag-in → query instantly (background warm on drop). RAG side-by-side in eval. **Record canned SSE fixtures from the H8 build.** | **M3** |
| **— H14: HARD CUT LINE —** | **Go/no-go:** `ingest → query → refine-in-place → threshold drag` works end-to-end (mock-backed if needed). If anything is shaky, **stop adding and polish exactly this loop.** Re-record canned fixtures for all beats. You are never left with nothing on stage. | **cut-line green** |
| **H14–19** | Real `VLLMScorer` swap (**M4**), measure actual warm refine p50 (confirm 100–300 ms) + first-query warm-vs-cold + scoped-vs-full ratios; freeze eval-slide numbers. **Record the canned fixtures from a REAL vLLM run** so the fallback streams genuine latencies. Stretch: scale sweep / Tier-2 cascade — only if cut-line is solid. | **M4** |
| **H19–22** | Finalize `DEMO.md` (spoken lines, framing, per-beat fallback, operator runbook). `preload_demo.sh` preflight. Scope gate: second domain / editable chips **only if M4 green and cut-line rock-solid**; else pure polish. | **M5: demo locked** |
| **H22–24** | Three dress rehearsals (once live, once with an injected failure forcing replay fallback, once timed to <90 s). Preload datasets. Freeze the repo. | freeze |

**Critical path:** `M0 contracts → MockScorer → backend SSE on mock (M1) → frontend consuming (M2) →
refine loop (M3) → H14 cut-line`. The real-vLLM swap (M4) and the score-validation gate run **in
parallel on the box** and produce the eval numbers but must **not** block the demo.

**Single most likely failure per window:** H0–3 = contract drift not actually resolved; H3–8 =
SSE/field-name mismatch at first integration; H8–14 = clause-engine off-by-one or classifier
mis-route; H14–19 = vLLM/AWQ/prefix-cache version fight eats hours; H19–22 = scope creep into stretch;
H22–24 = demo overruns 90 s.

---

## 8. The demo (5 beats + optional closers)

Lock the **live** run to **five beats**; the irreducible story must land by ~60 s. Word-sense recovery
and the eval slide are **optional closers** (the `refocus`/rewrite op is the hardest to classify, so
keep it off the critical path). Every beat has a canned twin via `scripts/replay_sse.py`.

> **Open hook:** *"grep, but the predicate is meaning — and you steer it live."*

1. **Stream best-first.** Point at the pre-warmed mixed corpus. Type *"every place we retry a network
   call without backoff."* Matches stream best-first; histogram + paper-vs-code facet fill; the ETA
   progress bar shows deterministic completion (single-token output). *"No index was built."*
2. **Click-NOT.** A wrong match appears with its rationale → click *"not like this"* → **Exclude** chip
   → the set re-steers in under a second, histogram reflows. *"We cached the corpus — your clarification
   only re-read a few tokens per chunk."*
3. **AND refine.** Type *"only the ones in the networking layer"* → **Require** chip → the set tightens,
   facet bars shift. *This repeated sub-second refinement is the star.*
4. **Threshold drag.** Drag the on-histogram threshold, precision ↔ recall, set reflows instantly.
   *"Zero new inference — scores were computed once."*
5. **Fresh data.** Drag in a fresh file, query it immediately. *"RAG needs minutes to index this. We
   needed nothing."* ← strongest beat: the un-indexable insight made physical.

> **Optional closers (drop if over time / risky live):**
> 6. **Word-sense recovery** — *"I meant retrieval in the IR sense, not RAG."* (canned fixture is the
>    primary path; live only if rehearsed-clean). ← most differentiated beat.
> 7. **Performance close (eval slide)** — lead with the **one** area-under-loop chart (scoped
>    saturates at ~2.2N while RAG climbs and step-jumps on every re-index); keep the roofline / MFU
>    waterfall as backup-if-asked, not a second spoken chart. This is the single number-heavy moment.
>    See [`performance/`](performance/) and [`DEMO.md`](DEMO.md).
>
> **Landing line:** *"RAG exists because inference is expensive. As inference gets cheap, retrieval
> collapses into a single semantic filter — and the engineer's full-time job of tuning recall/precision
> becomes a few seconds of natural-language refinement. We built the interactive surface for that
> world."*

**Pre-empt "is this real?"** with one rehearsed opening sentence: *"running live on our 8×H100 box; if
the venue network drops I'll switch to a recorded run from that same box."* Honesty up front turns the
biggest risk into a credibility asset.

---

## 9. Eval — the slide that makes them believe it

**Validate the score FIRST.** *Don't optimize the speed of being wrong.* `eval/bench.py` hard-STOPs the
speed sweeps if gold F1 < 0.7 (unless `--force`). The **real** GO/NO-GO runs on the box against the
real model — a green gate on the mock proves nothing (the mock is constructed to land ~0.8). Add a
**histogram-shape check**: require a visibly bimodal distribution on the actual scripted predicates, or
the threshold-drag beat is a slider over noise. Gold = arXiv topic gold (automatic, no token — the
safe fallback if BrowseComp-Plus is auth-walled) + a hand-authored codebase questionnaire +
BrowseComp-Plus slice.

**The money shot is iteration, not a single lookup.** The one believability chart is the **cumulative
iteration-cost curve**: our scoped refine-loop compute vs RAG's per-turn re-retrieve (+ re-index on
changed data) over a realistic 6–10-turn session, with the area between the curves shaded. Everything
else (F1, throughput, scaling) is a backup row, not a spoken beat.

**Performance metrics are predicted-then-measured.** The imported [`performance/`](performance/)
layer supplies the roofline, FLOP/MFU accounting, scoped-loop model, compute-vs-churn model, and KV
capacity model. `eval/bench.py` should log deterministic work counters first (`chunks_scored`,
`chunks_served_from_cache`, `rho` — the per-turn trace schema is frozen in [`CONTRACTS.md`](CONTRACTS.md)
§6) and overlay latency/energy afterward. The winning move is **counterfactual replay**: instrument
*one* real scoped session, then compute all four area-under-loop curves (scoped/full/suffix/RAG)
analytically from that single trace — a fair head-to-head by construction. **Metric additions** (each
derived, not assumed; full hierarchy in [`METRICS.md`](METRICS.md)): MFU (achieved ÷ theoretical peak,
target 40–55 % prefill), arithmetic intensity (scan ≫ ridge ⇒ compute-bound), suffix-only speedup vs
the predicted **12–24×** band, cumulative compute @ turn k, break-even churn **D\*** (recompute beats
RAG above it; for streaming/logs D→∞ so we always win), and **energy/query in joules** (the most honest
cost unit). Count inference *before* timing it — wall-clock is an overlay, never the x-axis for the
compute curves.

**Quality is a curve, not a vibe.** Report ROC/PR + AUC and an ECE/reliability diagram vs gold; set the
threshold-handle default from a target-precision operating point — the on-histogram drag becomes
*"sliding along the ROC curve."* Miscalibrated logprobs (high ECE) are the **measured** justification
for the Tier-2 cascade on the uncertain band, not a guess.

**Verify the `performance/theory.py` constants on the box before any number reaches a slide**
([`performance/docs/04_constants_to_verify.md`](performance/docs/04_constants_to_verify.md)): GQA
`n_kv_heads` (it sizes the whole KV figure — easy to get wrong), real peak TFLOP/s by precision
(microbench, don't trust datasheet), HBM bw/size, and real prefix/suffix token counts. Every figure
regenerates from them. And **capture the perishable naive cold floor first**, before warm-state
optimizations make it impossible to isolate.

**Optimization ladders** (config flags so the harness sweeps the curve, apples-to-apples on the same
session):
* **Ladder B — refine latency (headline):** B0 full-corpus re-score cold (seconds) → B1 warm+suffix-only
  → B2 candidate-set scoping → B3 persistent score cache → **~100–300 ms/turn at 10–20k**.
* **Ladder A — throughput:** batching → data-parallel ×6 → FP8 compute.
* **Ladder A2 — capacity:** 4-bit weights/KV → larger warm cache and more comfortable replica sizing.
* **Ladder C — scale:** 10k→20k→100k; mark the KV crossover where warm hands off to scoping. If the box
  is tight, report 10k+20k measured and 100k projected-from-cost-model, **clearly labeled**.

Run numbers twice: an H0–3 sizing run, and an H22 frozen run. `metrics.json` carries
`backend: mock|vllm`; any chart from mock is stamped **PROJECTED (mock)** — never present projected
latencies as measured.

---

## 10. Scope: explicitly cut / deferred

**Deferred per the user / off the critical path:**
* **Footprint tab** (middle dashboard tab) — disabled stub, **no** fields added to the aggregate
  contract. The recompute-over-store thesis stays a *spoken* pitch layer.
* **Version-2 live log stream** — not built. (Strongest thesis illustration, but near-duplicate noise +
  24h streaming risk; a candidate H19–22 ceiling-raiser only, scripted/canned if ever shown.)

**Cut for 24h (good post-event, none load-bearing on stage):**
* Tier-2 cascade + decision heatmap (stretch; cut hard if box late).
* `include`-over-complement and rewrite-re-score-parent clause algebra (build only `require`, `exclude`,
  threshold, one `refocus`; ship `REFINE_MODE=full` as correctness fallback).
* 100k scale sweep as a live beat (eval-slide row only).
* Editable/flippable chips (Approach C) — removable chips (Approach A) are the safety net.
* Section-level paper chunking & function-level code chunking (abstract+file level only; get 10× from
  more code/papers, not finer chunking).
* Live arXiv API top-up (snapshot-only is the safe path; synthetic planted corpus is the offline floor).
* Two-physical-SSE-stream variant (commit to single multiplexed stream).
* Autonomous agents, multimodal/video, distillation, cross-query caching, PDF parsing, exotic corpora.
* RAG as a **live** on-stage race (eval-table + pitch line only — a live race invites a failure we can't
  control and a fairness argument we can't win in 90 s).

---

## 11. Risk register & degradation ladder

| Risk | Trigger | Mitigation / fallback | Owner | By |
|---|---|---|---|---|
| **Contract drift** (4 scorer sigs, 3 chunk schemas…) | first integration breaks on field names | reconcile ONE `CONTRACTS.md` in hour zero; delete duplicate defs; contract test | Delivery | H1 |
| **Score poorly calibrated / F1<0.7** | gate fails on box | hard-STOP speed work; swap to Llama-3.1-8B-AWQ; relative ranking + tuned threshold; Tier-2 cleans band | A/D | H3–8 |
| **Histogram is mush** (no bimodal separation) | threshold-drag beat meaningless | histogram-shape check in the gate on real scorer for scripted predicates; tune default threshold | A/C | H3–8 |
| **H100 box unreachable / vLLM won't come up** | no GPU at demo | `SCORER_BACKEND=mock` runs the whole demo; timebox bring-up to ~90 min; drop `guided_choice`; fall back model | A | M4 |
| **Prefix-cache doesn't hit** | suffix-only re-prefill goes cold | candidate-set scoping (#3) carries refine; pitch headline shifts to scoped-vs-full ratio | A | H0–3 |
| **Classifier mis-routes** | wrong chip mid-demo | removable chips + explicit op buttons; scripted utterances verified in rehearsal; canned fixture primary for `refocus` beat | B/C | H8–14 |
| **Fresh-file warm too slow** | drag-in hangs | background warm **on drop** (not on query); pre-staged 2nd corpus as Level-4 fallback | A/B/C | H8–14 |
| **Clause-engine off-by-one** | scoped set wrong | `REFINE_MODE=full` correctness fallback; assert scoped == full in tests | B | H14 |
| **Live SSE/backend hangs** | stall on stage | `replay_sse.py` byte-identical canned stream; flip `VITE_API_BASE` | Delivery | H8+ |
| **Python 3.14 dep drift** | local install breaks | standardize local dev on pinned **3.11/3.12** venv; numpy fallback for faiss; never install vllm locally | All | H0 |
| **Demo overruns 90 s** | loses the room | per-beat time budget; beats 1–5 by ~60 s; 6–8 droppable; timed rehearsal ×3 | Delivery | H22 |
| **Scope creep into stretch** | polish window eaten | scope gate at H19–22 only opens if M4 green + cut-line solid | Delivery | H19 |
| **Warm-KV exceeds HBM at ~14k** (FP16) | warm path OOMs above the demo corpus | quantized KV (`--kv-cache-dtype fp8`) required above ~14k; mark crossover (fig 5); #3 scoping carries past it | A | H14–19 |
| **Quantization overclaim** ("4-bit → 4× scan speed") | judge catches it in the compute-bound regime | split levers in slides: 4-bit = *capacity*, FP8 = *throughput*; never claim a 4-bit scan speedup | A/Delivery | H4 |
| **Cold floor contaminated** by warm-cache leakage | MFU / ladder numbers invalid | capture the naive cold floor FIRST; explicit cache reset or fresh corpus id | A | H0–3 |
| **MFU measured on padded tokens** | reported MFU misleads | log real-token vs padded-token MFU; report which denominator; power-draw sanity check (50 % MFU ⇏ 300 W) | A | H14–19 |
| **Replay provenance unclear** | canned fallback "looks fake" | label every fixture by backend/model/corpus/commit/run-time; record the demo fixtures from a **real** vLLM run | Delivery | H19 |

> Full live register (these + triggers/phases) is maintained in [`RISKS.md`](RISKS.md).

**Degradation ladder — "never nothing on stage":**
* **L0** real vLLM on 8×H100, live SSE, live classifier, live fresh-ingest — full demo.
* **L1** `SCORER_BACKEND=mock` — identical demo, simulated scorer, every interaction still real.
* **L2** backend/SSE flakes → `VITE_API_BASE` → `replay_sse.py` (recorded from a **real** vLLM run).
* **L3** classifier mis-routes → explicit Require/Exclude/Include buttons + removable chips.
* **L4** fresh-ingest broken → pre-staged un-warmed 2nd corpus, narrated as "something we never indexed".
* **L5** worst case → the H14 irreducible loop (`ingest → query → refine → threshold`) on mock, narrated.
* Eval slide is always static (pre-computed JSON) — the numbers beat never depends on anything live.

---

## 12. Open questions to settle in the H0–1 huddle

1. Final chunk-id hash inputs (text included? — yes, per `CONTRACTS.md`) and the one home for the
   function (`data/schema.py`, imported by both chunkers).
2. Histogram bins: **20** confirmed everywhere (server + client recut)?
3. Refine transport: single stream, chip-first event confirmed (no `/refine/stream/{turn_id}`)?
4. Do new `require` clauses default soft (graded, recoverable) — yes — and how is "hard" marked
   (keep/drop click sets the hard bit)?
5. `rationale` source for the click-NOT beat: Tier-2 string vs category snip vs canned — default
   "always-optional; canned for the one scripted beat".
6. Cornell arXiv snapshot revision to pin; code-repo allowlist + SHAs + licenses (prefer MIT/Apache/BSD).
7. Will the box be reachable from the venue network? Default stage posture = local-mock; live-vLLM is
   "and it runs for real on 8 H100s." Decide by M4.
8. Measure the `performance/theory.py` constants on the real node before freezing any slide number —
   especially the GQA `n_kv_heads` (sizes the KV figure) and real peak TFLOP/s by precision. Which
   KV-cache dtype do we ship for the demo corpus (fp8 vs 4-bit), given the ~14k FP16 crossover?

---

## 13. Definition of done (per checkpoint)

* **M0 (H1):** `CONTRACTS.md` signed; `inference/scorer.py`, `data/schema.py`, `frontend/lib/types.ts`,
  `backend/schemas.py` all reference the same shapes; one `MockScorer`; `make` boots backend+mock.
* **M1 (H~5):** `curl POST /query` against mock streams valid `result`+`aggregate`+`done` frames that
  validate against `schemas.py`.
* **M2 (H~8):** frontend on mock SSE: query → histogram + facets + threshold drag (zero network on
  drag, proven by test) + capped best-first feed. **Demo-able.**
* **M3 (H~14):** refine loop live on mock: NL→chip→scoped re-score→diff, click-NOT, chip removal,
  fresh-file ingest. **Cut-line green.**
* **M4 (H~19):** `SCORER_BACKEND=vllm` on the box: F1 gate passed, warm refine p50 measured, canned
  fixtures recorded from the real run, eval-slide numbers frozen.
* **M5 (H~22):** `DEMO.md` locked, `preload_demo.sh` prints GO, rehearsed under 90 s.
