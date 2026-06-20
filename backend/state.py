from __future__ import annotations

from dataclasses import dataclass, field

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.scorer import PrefixState, ScoreResult

from backend.schemas import FacetBucket, HIST_BINS, HistogramBin


@dataclass
class BackendState:
    corpus_id: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    warm_state: PrefixState | None = None
    current_clause: str | None = None

    @property
    def warmed(self) -> bool:
        return bool(self.warm_state and self.warm_state.warmed)

    def load_demo(self) -> list[Chunk]:
        self.corpus_id = "demo"
        self.chunks = demo_chunks()
        self.current_clause = None
        return self.chunks

    def chunks_by_id(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}


def demo_chunks() -> list[Chunk]:
    seeds = [
        (
            "repo:urllib3#connectionpool.py",
            "code",
            "urllib3/connectionpool.py",
            "HTTP connection pool retries network calls but does not always use exponential backoff.",
            ChunkMeta("python", 2023, "src/urllib3/connectionpool.py", "python", "urllib3", "synthetic"),
        ),
        (
            "repo:requests#adapters.py",
            "code",
            "requests/adapters.py",
            "Adapter retry handling for transient networking failures with configurable backoff.",
            ChunkMeta("python", 2024, "src/requests/adapters.py", "python", "requests", "synthetic"),
        ),
        (
            "2103.00020",
            "paper",
            "Neural Retrieval for Code Search",
            "A paper about retrieval ranking, semantic filters, and search quality evaluation.",
            ChunkMeta("cs.IR", 2021, None, None, None, "synthetic"),
        ),
        (
            "2401.01010",
            "paper",
            "Inference Caching for Interactive Search",
            "Prefix cache reuse can reduce warm query latency for interactive relevance filtering.",
            ChunkMeta("cs.LG", 2024, None, None, None, "synthetic"),
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
