# CONTRACTS.md — the frozen seam

> **This is the single source of truth for every interface that crosses an owner boundary.**
> Four people build in parallel against mocks; that only works if everyone mocks the *same* schema.
> Three independent design reviews found the #1 risk was that each subsystem had quietly written a
> *different* "frozen" interface. This file reconciles them. **Freeze it at H1.** Any change requires
> the delivery owner's sign-off and a changelog entry at the bottom.

Status: **DRAFT → freeze at H1**. Last reconciled: 2026-06-19.

---

## 0. Golden rules

1. **One definition, many imports.** Shared Python types live in `inference/scorer.py` and
   `data/schema.py`. `backend/` and `eval/` **import** them — they never redefine them.
   The frontend hand-mirrors them in `frontend/src/lib/types.ts` (kept in sync by eye + a fixture test).
2. **The scorer is clause-agnostic.** It scores `(chunk_text, predicate)` → a float. It knows nothing
   about clauses, candidate sets, gating, or the UI. All of that is `backend/`.
3. **`result.score` over the wire is the FINAL combined relevance** (post gate × soft). The frontend
   caches that single number per chunk and re-cuts the threshold over it — so the client-side
   "drag = zero inference" recut provably matches what the server would return.
4. **`ScoreResult`, not `float`.** Rich result struct flows everywhere; backend reads `.score`, eval reads
   `.p_yes`/`.tier`/`.from_cache`, cascade reads `.tier`. A bare float is a premature narrowing.
5. **One MockScorer.** It lives in `inference/mock_scorer.py` and is imported by backend and eval.
   The frontend has its own TS `mockAdapter` (different language) but it must produce the *same*
   score distribution shape. There are **not** four mocks.

---

## 1. The scorer interface (`inference/scorer.py`)

```python
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal

# ---- the unit of work ----
@dataclass
class ScoreRequest:
    chunk_id: str
    chunk_text: str        # the raw chunk; scorer builds [instruction+chunk] prefix from this
    predicate: str         # the short changing suffix question

# ---- the unit of result (rich, not a float) ----
@dataclass
class ScoreResult:
    chunk_id: str
    score: float           # P(Yes)/(P(Yes)+P(No)), in [0,1]
    p_yes: float
    p_no: float
    tier: int = 1          # 1 = Tier-1 filter; 2 = escalated to Tier-2 (cascade, stretch)
    from_cache: bool = False
    latency_ms: float = 0.0

# ---- opaque warm handle ----
@dataclass
class PrefixState:
    corpus_id: str
    n_chunks: int
    warmed: bool
    model_id: str

class ScorerClient(ABC):
    @abstractmethod
    async def warm(self, corpus_id: str, chunks: list["Chunk"]) -> PrefixState: ...
    @abstractmethod
    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]: ...
    @abstractmethod
    async def health(self) -> dict: ...        # {ready: bool, replicas: [...], warmed_corpora: [...]}
    @abstractmethod
    def model_id(self) -> str: ...
```

* `score_batch` returns results **in the same order** as `items`.
* `make_scorer()` (in `inference/config.py`) is the **only** place backend/eval learn which impl they got.
* Yes/No normalization to `[0,1]` happens **inside the scorer** (in `inference/`), never in backend.

```python
# inference/config.py
SCORER_BACKEND = env("SCORER_BACKEND", "mock")   # "mock" | "vllm"  — THE swap point
def make_scorer() -> ScorerClient: ...           # returns MockScorer or VLLMScorer
```

---

## 2. The chunk schema (`data/schema.py`)

```python
from dataclasses import dataclass
from typing import Literal, Optional
import hashlib

@dataclass
class ChunkMeta:
    category: Optional[str]   # arXiv category for papers (cs.IR…); language name for code (python…)
    year: Optional[int]
    path: Optional[str]       # file path for code; None for papers
    lang: Optional[str]       # code language; None for papers
    repo: Optional[str]       # code repo; None for papers
    source: str               # PROVENANCE: "arxiv_snapshot" | "github" | "synthetic" | "dropped"

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str               # e.g. "2103.00020" or "repo:urllib3#src/connectionpool.py"
    type: Literal["paper", "code"]   # MODALITY — this is the paper-vs-code facet field
    title: str
    text: str
    meta: ChunkMeta

def chunk_id_of(doc_id: str, idx: int, text: str) -> str:
    """THE one id function. Imported by data/chunker.py AND backend/chunker.py.
    Text IS in the hash so identical content re-ingested collides correctly."""
    return hashlib.sha1(f"{doc_id}|{idx}|{text}".encode()).hexdigest()[:16]
```

> **Resolved naming forks (do not re-litigate):**
> * Modality (paper vs code) is **`chunk.type`** everywhere. The frontend renames its old
>   `meta.source` → `meta.type`.
> * Provenance (where it came from) is **`chunk.meta.source`**. Never overload `type` for this.
> * Identity is **`chunk_id`** everywhere (never `.id`).

---

## 3. HTTP + SSE API (`backend/main.py`)

All SSE frames are standard `data: <json>\n\n`. Two logical channels (`result`, `aggregate`) are
**multiplexed over ONE stream per request**, distinguished by the `type` field. There is **no**
second physical socket and **no** `/refine/stream/{turn_id}`.

### `POST /ingest`
```
req:  { "corpus_id": "demo" }
resp: { "corpus_id": "demo", "n_chunks": 18234,
        "facets": { "type": [FacetBucket], "category": [FacetBucket], "year": [FacetBucket] },
        "warm_eta_s": 12.0 }
# side effect: kicks off the async warm pass
```

### `POST /query`  → `text/event-stream`
```
req: { "predicate": "every place we retry a network call without backoff", "threshold": 0.5 }
```
Stream events (each a JSON object on a `data:` line, routed by `.type`):

```jsonc
// result — emitted best-first within a reorder window
{ "type": "result", "chunk_id": "…", "score": 0.91,
  "meta": { "type": "code", "title": "urllib3/connectionpool.py", "category": "python",
            "year": 2023, "path": "src/urllib3/connectionpool.py", "lang": "python", "repo": "urllib3" },
  "rank": 0, "rationale": null }

// aggregate — coalesced (latest-wins under backpressure); reflects current threshold
{ "type": "aggregate", "scanned": 8192, "matched": 244,
  "histogram": [ { "lo": 0.0, "hi": 0.05, "count": 31 }, … 20 bins … ],
  "facets": { "type":     [ {"key":"paper","relevant":120,"total":12000},
                            {"key":"code","relevant":124,"total":6234} ],
              "category": [ {"key":"cs.IR","relevant":40,"total":1800}, … ],
              "year":     [ {"key":"2024","relevant":60,"total":3000}, … ] },
  "threshold": 0.5, "eta_ms": 1400 }

// done — terminates the stream
{ "type": "done", "scanned": 18234, "matched": 244, "elapsed_ms": 870, "warm": false,
  "summary": "18,234 scanned · 244 matched" }
```

### `POST /refine`  → `text/event-stream`
```
req: { "utterance": "only in the networking layer" }
   | { "click": { "chunk_id": "…", "sign": "-" } }      // "-" = drop, "+" = keep
   | { "brush": { "lo": 0.6, "hi": 1.0 } }
```
**FIRST event is the chip, then the diff, then aggregate, then done** — all on the same stream:
```jsonc
{ "type": "chip", "operation": "require",
  "chip": { "clause_id": "c3", "op": "require", "text": "in the networking layer",
            "label": "Require", "removable": true, "confidence": 0.86 },
  "refine_ms": 180, "latency_kind": "warm" }

{ "type": "diff", "added": [ {result…} ], "removed": ["chunk_id", …],
  "rescored": [ {"chunk_id":"…","score":0.74}, … ], "refine_ms": 180 }

{ "type": "aggregate", … }   // reflowed
{ "type": "done", … }
```

### `DELETE /clause/{clause_id}`  (chip removal — **zero inference**)
```
resp: { "removed": true, "refine_ms": 4 }   // survivors recomputed from cache; frontend re-pulls /results
```

### `GET /results?threshold=&top_k=`  (pure cache read — **zero inference**)
```
resp: { "items": [ {"chunk_id","score","meta"} ], "threshold": 0.55, "top_k": 50, "total_matched": 244 }
```
> The frontend usually computes this client-side from its own score cache. The endpoint exists for
> parity and for eval. Kept in the contract.

### `GET /healthz` → `{ "ready": bool, "scorer": "mock"|"vllm", "warmed": bool }`

---

## 4. Shared value objects

```jsonc
// FacetBucket — backend computes BOTH relevant (≥ threshold) and total (all scanned)
{ "key": "code", "relevant": 124, "total": 6234 }

// HistogramBin
{ "lo": 0.55, "hi": 0.60, "count": 18 }
```

* **`HIST_BINS = 20`** — one constant, server and client. The client-side recut uses the *same* 20 bins
  so the histogram never visibly jumps when it switches from server-seeded bins to client recut.
* **Chip / refine-op vocabulary (one enum, lowercase, used by classifier output AND chip AND frontend):**

  | wire op    | label     | clause semantics                          | typical trigger             |
  |------------|-----------|-------------------------------------------|-----------------------------|
  | `require`  | Require   | AND over current survivors                | "only…", "must…", "also requires…", keep-click |
  | `exclude`  | Exclude   | NOT over current survivors                | "not…", "without…", drop-click |
  | `include`  | Include   | OR over the complement                    | "also include…", "or…"      |
  | `refocus`  | Refocus   | rewrite — re-score over parent set        | "I meant…", "actually…", "in the … sense" |
  | `brush`    | Range     | client-side threshold/range filter        | on-histogram drag           |

  `backend/classifier.py` emits exactly these tokens; `frontend/src/lib/classify.ts` mirrors them so the
  optimistic chip matches the server chip. **A contract test runs the same 10 utterances through both
  and asserts equal ops.**

* **click payload key is `sign`** (`'+'|'-'`), never `dir`.
* **`rationale` is always-optional, never-required.** Tier-1-only builds emit `null`; the frontend
  falls back to a category-based one-liner. Only the one scripted wrong-match beat needs a real
  rationale and it may be canned.

---

## 5. clause_id ownership & the combine formula

* **Backend mints `clause_id`s.** The scorer never sees them.
* The score cache is keyed **`(chunk_id, clause_id) → float`** (`backend/cache.py`). This is the
  B2/B3 latency win: threshold drags and chip removals are pure dict reads + recombine, zero inference.
* `cache.missing(clause_id, candidate_ids)` returns exactly the chunks that still need inference — the
  refine cost is `|scoped missing|`, **not** `|corpus|`.
* **Combine** (`backend/clause.py`): `relevance = gate(hard) × soft`, where
  * `gate = 1` iff every **hard require** clause scores ≥ `G_HI` **and** every **hard exclude** clause
    scores ≤ `G_LO` (else 0);
  * `soft = Σ wᵢ·sᵢ / Σ wᵢ` over soft clauses.
  * New clauses default **soft** (graded, recoverable) unless a click marks them hard. `G_HI`/`G_LO`/`wᵢ`
    are config flags, tunable live.
* **The SSE `result.score` is the post-combine `relevance`.** A `rewrite`/`refocus` mints a *new*
  `clause_id` (invalidating only that column) and re-scores over the parent set; the superseded clause
  drops out of the combine sum.

---

## 6. Performance trace schema (`eval/bench.py` / `METRICS.md`)

Every query and refine turn emits a trace row. This is the bridge between runtime
instrumentation and the theoretical models in `performance/theory.py`.

```jsonc
{
  "run_id": "2026-06-20-h100-freeze",
  "commit": "git-sha",
  "corpus_id": "demo",
  "model_id": "Qwen/Qwen2.5-3B-Instruct-AWQ",
  "scorer_backend": "mock|vllm",
  "turn": 3,
  "operation": "query|require|exclude|include|refocus|brush|delete_clause",
  "threshold": 0.5,
  "n_chunks_total": 18234,
  "candidate_count": 4312,
  "chunks_scored": 4312,
  "chunks_served_from_cache": 13922,
  "survivor_count": 2368,
  "rho": 0.55,
  "elapsed_ms": 180.0,
  "model_ms": 142.0,
  "queue_ms": 18.0,
  "cache_hit_rate": 0.76,
  "warm_state": "cold|warm|cached",
  "latency_kind": "cold|warm|cached",
  "quality_slice": "demo-retry-without-backoff"
}
```

Rules:

* `chunks_scored` counts only cache misses that hit the scorer. This is the primary compute unit.
* `chunks_served_from_cache` counts cache hits used to recombine or recut scores.
* `rho = survivor_count / candidate_count` for refine turns; `null` is allowed for the first query.
* Threshold drag and chip deletion must emit `chunks_scored = 0`.
* Wall-clock fields are overlays. Do not use them as the x-axis for area-under-loop curves.
* Mock traces must be stamped `scorer_backend: "mock"` and any derived chart must be labeled projected.

---

## 7. Environment variables (one name each)

| Variable          | Side     | Values            | Meaning                                              |
|-------------------|----------|-------------------|------------------------------------------------------|
| `SCORER_BACKEND`  | Python   | `mock` \| `vllm`  | which `ScorerClient` `make_scorer()` returns         |
| `VLLM_REPLICAS`   | Python   | csv of base URLs  | e.g. `http://h100:8001,…,http://h100:8006`           |
| `VITE_DATA_MODE`  | Frontend | `mock` \| `live`  | pure-frontend mock adapter vs real backend           |
| `VITE_API_BASE`   | Frontend | path/URL          | `/api` (live backend) or the replay server's URL     |

> Backend imports `inference.config.make_scorer` — it does **not** have its own `GREP_SCORER`.
> The demo seed query is sourced from `data/predicates.yaml`, not hardcoded in the frontend.

---

## 8. Files that must exist before parallel work (H0–1)

* `inference/scorer.py` — `ScorerClient`, `ScoreRequest`, `ScoreResult`, `PrefixState`
* `inference/mock_scorer.py` — the one `MockScorer`
* `inference/config.py` — `SCORER_BACKEND`, `make_scorer()`
* `data/schema.py` — `Chunk`, `ChunkMeta`, `chunk_id_of`
* `backend/schemas.py` — pydantic models that **import** the above and define the SSE/HTTP wire models
* `frontend/src/lib/types.ts` — TS mirror of all of the above
* This file (`CONTRACTS.md`) — signed off by A, B, C, D
* `METRICS.md` + `performance/theory.py` — shared performance vocabulary and closed-form models

---

## Changelog
* 2026-06-20 — added performance trace contract and linked metrics/theory artifacts.
* 2026-06-19 — initial reconciliation of the six subsystem drafts into one contract (delivery owner).
