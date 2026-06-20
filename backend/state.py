from __future__ import annotations

from dataclasses import dataclass, field

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.scorer import PrefixState, ScoreResult

from backend.clause import ClauseRecord
from backend.schemas import FacetBucket, HIST_BINS, HistogramBin
from backend.schemas import FreshDocument


@dataclass
class BackendState:
    corpus_id: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    warm_state: PrefixState | None = None
    current_clause: str | None = None
    clauses: dict[str, ClauseRecord] = field(default_factory=dict)
    threshold: float = 0.5

    @property
    def warmed(self) -> bool:
        return bool(self.warm_state and self.warm_state.warmed)

    def load_demo(self) -> list[Chunk]:
        self.corpus_id = "demo"
        self.chunks = demo_chunks()
        self.current_clause = None
        self.clauses.clear()
        return self.chunks

    def chunks_by_id(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    def append_documents(self, documents: list[FreshDocument]) -> list[Chunk]:
        start = len(self.chunks)
        for offset, document in enumerate(documents):
            doc_id = f"fresh:{document.title}"
            meta = ChunkMeta(
                document.category,
                document.year,
                document.path,
                document.lang,
                document.repo,
                "fresh",
            )
            self.chunks.append(
                Chunk(
                    chunk_id=chunk_id_of(doc_id, start + offset, document.text),
                    doc_id=doc_id,
                    type=document.type,
                    title=document.title,
                    text=document.text,
                    meta=meta,
                )
            )
        self.current_clause = None
        self.clauses.clear()
        return self.chunks


def demo_chunks() -> list[Chunk]:
    # Curated cut-line corpus (Phase 03): pinned so the scripted demo beats have
    # known results. Chunks 1-3 are network retries in the networking layer
    # (survive the headline query AND the "networking layer" refine); 4-5 are
    # non-network retries (survive the query but drop on the networking refine —
    # the click-NOT / AND-narrow targets); 6-7 are clear non-matches for spread.
    #
    # WARNING: MockScorer applies ±0.09 stable jitter, so edits to these texts
    # must keep survivors clear of the 0.5 threshold. tests/test_phase3_cut_line.py
    # is the guard — if an edit flips a survivor, the cut-line loop goes RED.
    seeds = [
        (
            "repo:urllib3#connectionpool.py",
            "code",
            "urllib3/connectionpool.py",
            "In the networking layer, the HTTP connection pool retries a network call but never applies exponential backoff.",
            ChunkMeta("python", 2023, "src/urllib3/connectionpool.py", "python", "urllib3", "synthetic"),
        ),
        (
            "repo:requests#adapters.py",
            "code",
            "requests/adapters.py",
            "The networking layer HTTP adapter retries a request on transient connection failures; no backoff is configured.",
            ChunkMeta("python", 2024, "src/requests/adapters.py", "python", "requests", "synthetic"),
        ),
        (
            "repo:aiohttp#client.py",
            "code",
            "aiohttp/client.py",
            "An async client in the networking layer retries a network call and skips exponential backoff entirely.",
            ChunkMeta("python", 2024, "aiohttp/client.py", "python", "aiohttp", "synthetic"),
        ),
        (
            "repo:app#db_session.py",
            "code",
            "app/db_session.py",
            "The database session retries a failed transaction on deadlock with a capped exponential backoff.",
            ChunkMeta("python", 2023, "src/app/db_session.py", "python", "app", "synthetic"),
        ),
        (
            "repo:app#worker.py",
            "code",
            "jobs/worker.py",
            "A background job worker retries a failed task a few times before giving up.",
            ChunkMeta("python", 2022, "src/jobs/worker.py", "python", "app", "synthetic"),
        ),
        (
            "2103.00020",
            "paper",
            "Neural Retrieval for Code Search",
            "A paper on retrieval ranking metrics and semantic search quality for code.",
            ChunkMeta("cs.IR", 2021, None, None, None, "synthetic"),
        ),
        (
            "repo:demo#ui.ts",
            "code",
            "demo/ui.ts",
            "Histogram brushing updates thresholds without rescoring cached result chunks.",
            ChunkMeta("typescript", 2024, "src/ui.ts", "typescript", "demo", "synthetic"),
        ),
    ]
    return [
        Chunk(
            chunk_id=chunk_id_of(doc_id, idx, text),
            doc_id=doc_id,
            type=chunk_type,  # type: ignore[arg-type]
            title=title,
            text=text,
            meta=meta,
        )
        for idx, (doc_id, chunk_type, title, text, meta) in enumerate(seeds)
    ]


def facet_summary(chunks: list[Chunk], scores: dict[str, ScoreResult] | None = None, threshold: float = 0.0) -> dict[str, list[FacetBucket]]:
    def bucket_value(chunk: Chunk, name: str) -> str:
        if name == "type":
            return chunk.type
        value = getattr(chunk.meta, name)
        return str(value) if value is not None else "unknown"

    facets: dict[str, list[FacetBucket]] = {}
    for name in ("type", "category", "year"):
        counts: dict[str, tuple[int, int]] = {}
        for chunk in chunks:
            key = bucket_value(chunk, name)
            relevant, total = counts.get(key, (0, 0))
            score = scores.get(chunk.chunk_id) if scores else None
            is_relevant = score is None or score.score >= threshold
            counts[key] = (relevant + int(is_relevant), total + 1)
        facets[name] = [
            FacetBucket(key=key, relevant=relevant, total=total)
            for key, (relevant, total) in sorted(counts.items())
        ]
    return facets


def histogram(scores: list[ScoreResult]) -> list[HistogramBin]:
    counts = [0 for _ in range(HIST_BINS)]
    for score in scores:
        idx = min(int(score.score * HIST_BINS), HIST_BINS - 1)
        counts[idx] += 1
    return [
        HistogramBin(lo=idx / HIST_BINS, hi=(idx + 1) / HIST_BINS, count=count)
        for idx, count in enumerate(counts)
    ]
