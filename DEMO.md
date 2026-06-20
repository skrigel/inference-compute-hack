# Demo Runbook — LOCKED (Phase 05 / M5)

Target: **≤ 90 seconds**, no rushing. The demo sells **interaction speed**, not a static
search result. Every beat changes the candidate set, re-cuts cached scores, or introduces
fresh data — never a single frozen lookup.

## Demo posture (read before the run)
- **Primary path = live mock backend** (`SCORER_BACKEND=mock`): fully interactive, deterministic,
  faithful. This is what we drive on stage.
- **Real vLLM on 6×H100 is partial:** throughput/latency measured (`eval/SLIDE.md` Fig 3) and the
  **small quality gate passed on real Modal vLLM — F1 0.94** (Fig 5, `measured`). The **full
  gold-set + ladder freeze is still pending** a Modal spend-limit reset
  (`eval/artifacts/phase04_modal_blocker.md`). Frame it as "and it also ran on 6×H100 at F1 0.94";
  never present the small gate as the full freeze.
- **⚠ Threshold is backend-specific.** The demo default **0.5 is correct for the mock backend**
  (the primary path). On the **real vLLM backend, set the calibrated threshold ≈ 0.016** (per the
  gate) or matches collapse to ~11% recall — the real model separates positives at a low absolute
  score. Do not run the real backend at 0.5 on stage.
- **Numbers on screen are measured-mock** on the pinned 7-chunk corpus (`cut_line_trace.json`);
  the eval slide labels every figure (`eval/SLIDE.md`).

---

## Spoken script (≤90s; per-beat budget in brackets)

**Hook [7s]** — *"This is grep, but the predicate is meaning — and you steer it live. No index,
no embeddings; it reads the raw text and emits one Yes/No token per chunk."*

1. **Stream best-first [15s].** *Action:* app opens on the seed query
   **"every place we retry a network call without backoff."**
   *Say:* *"It streams matches best-first as it scans — **7 scanned, 5 matched**, paper-vs-code
   facet filling. No index was built."*
2. **Click-NOT [12s].** *Action:* click **drop** on the lowest survivor — a non-network retry
   (`jobs/worker.py`). *Say:* *"That one's a job retry, not a network call — not like this."*
   An **Exclude** chip appears, the card drops, aggregates reflow → **4 matched, zero inference**:
   *"we cached the corpus; your clarification only re-read a few tokens."*
3. **AND refine [13s].** *Action:* type **`only in the networking layer`** → Enter. A **Require**
   chip appears; the DB retry drops → **3 matched**. *Say:* *"This refine scored **only the 4
   survivors, not the corpus** — that's candidate-set scoping, and it's the star."*
4. **Threshold drag [10s].** *Action:* drag the on-histogram threshold. Bars recolor, matched
   count moves, latency tag flips to **cached**. *Say:* *"Zero new inference — scores were
   computed once; I'm just re-cutting them."*
5. **Fresh file [15s].** *Action:* drag `fresh_incident.py` onto the surface; it re-queries.
   *Say:* *"Queryable on the next pass — **zero derived bytes written**. RAG would have to
   re-embed and rebuild its index before it could find this. We needed nothing."*
6. **Close on performance [13s].** *Action:* show `eval/artifacts/area_under_loop.png`.
   *Say:* *"Over the refine loop our scoped compute flattens at **11 chunks** while full
   re-score climbs to **21**. RAG exists because inference is expensive; as it gets cheap,
   retrieval collapses into this one live filter."* **[≈85s total]**

> **Optional closers (only if ahead):** IR-sense recovery (`"retrieval in the IR sense, not RAG"`
> recovers the dropped paper) · compute-vs-churn break-even · the 6×H100 throughput number.

---

## Pinned demo corpus (7 chunks, `backend/state.py::demo_chunks`)

| # | chunk | retry? | networking layer? | role |
|---|---|---|---|---|
| 1 | `urllib3/connectionpool.py` | yes (no backoff) | yes | strong match; survives query + AND |
| 2 | `requests/adapters.py` | yes (no backoff) | yes | strong match; survives query + AND |
| 3 | `aiohttp/client.py` | yes (no backoff) | yes | strong match; survives query + AND |
| 4 | `app/db_session.py` | yes (has backoff) | **no** | query survivor; **drops on the AND** |
| 5 | `jobs/worker.py` | yes (no backoff) | **no** | query survivor; the **click-NOT** target |
| 6 | `Neural Retrieval for Code Search` | no | no | non-match (spread); IR-sense closer |
| 7 | `demo/ui.ts` | no | no | non-match (spread) |

Measured beat counts (pinned, `cut_line_trace.json`): query **7→5**, click-NOT **0 scored→4**,
AND **4 scored→3**, threshold **0 scored**, fresh-file **8 scanned, fresh chunk matches**.

---

## Operator commands (copy-paste)

```bash
# 0) Preflight — must print "GO ✓" (drives the full loop + records all canned fixtures)
PYTHON=python3 bash scripts/preload_demo.sh

# 1) PRIMARY: live mock backend
SCORER_BACKEND=mock python3 -m uvicorn backend.main:app --port 8000
#    frontend (built ahead of time on Node ≥20.19):
VITE_DATA_MODE=live VITE_API_BASE=/api npm --prefix frontend run dev    # proxies /api → :8000

# 2) FALLBACK A — canned SSE replay (live path stalled): point the frontend here
python3 -m scripts.replay_sse serve --port 8090
#    set VITE_API_BASE=http://localhost:8090  (CORS is enabled on the replay server)

# 3) FALLBACK B — pure frontend, no backend at all
VITE_DATA_MODE=mock npm --prefix frontend run dev
```

## Fallback ladder (every live beat has a canned twin)

| Tier | How | Covers |
|---|---|---|
| Live mock backend | `SCORER_BACKEND=mock` | all beats, interactive (primary) |
| **Canned SSE replay** | `scripts/replay_sse serve` + `VITE_API_BASE` | beat 1 (`cut_line_query.sse`), beats 2-3 (`cut_line_refine.sse`), **beat 5** (`cut_line_fresh.sse`, auto-armed after a file drop) |
| Pure-frontend mock | `VITE_DATA_MODE=mock` | all beats, no backend |
| Manual staged loop | preloaded corpus + explicit op buttons | last resort |

> Beat 4 (threshold drag) is a client-side recut — no SSE, no fixture needed; it works on every tier.
> Real-vLLM replay fixture is **pending** the Phase 04 unblock; until then the canned twins are
> recorded from the mock backend (stated as such if asked).

## Eval slide
Frozen figures + captions + honest labels live in **[`eval/SLIDE.md`](eval/SLIDE.md)**. Lead with
Figure 1 (area-under-loop, measured-mock); cite Figure 3 (6×H100 throughput) with the **MFU
caveat**; the quality gate is **pending** the real-backend run — never present 6% MFU as the
architecture's MFU, and never present an unrun gate as passed.

## Preflight checks
- `bash scripts/preload_demo.sh` prints **GO ✓** and all fixtures exist (`tests/test_phase5_demo_lock.py`).
- `SCORER_BACKEND`, `VITE_DATA_MODE`, `VITE_API_BASE` set as intended.
- **Frontend `dist/` built ahead of time on Node ≥ 20.19** (Vite 8) — the venue box may not `npm install`.
- Two-person split: one drives the keyboard following the script, one narrates.

## Do Not Demo
- A single static query over a frozen corpus.
- Frozen ladder/throughput numbers as final — only the **small** gate is measured; the full freeze is pending.
- The real vLLM backend at threshold 0.5 (recall collapses to ~11%; use the calibrated ≈0.016).
- 6% MFU as "our MFU" (it's an under-saturated benchmark — see `eval/SLIDE.md`).
- 4-bit weight quantization as a raw scan-throughput multiplier (capacity lever; FP8 is throughput).
- Stretch features (Tier-2 cascade, second domain, editable chip algebra) — the H14 loop is the line.
