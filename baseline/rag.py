"""RAG baseline — EVAL ONLY.

Times index build + retrieve so we can quantify "RAG: minutes to (re)index,
ours: 0." This is a pure-Python hashing-vectorizer + cosine fallback so it runs
on the Mac with no faiss/torch; on the H100 box swap in sentence-transformers +
faiss.IndexFlatIP for real magnitudes (the structure and the emitted timing
record stay identical).
"""
from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import time
from dataclasses import dataclass

DIM = 256
SparseVector = dict[int, float]


def _embed(text: str) -> SparseVector:
    vec: SparseVector = {}
    for token in text.lower().split():
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
        vec[bucket] = vec.get(bucket, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in vec.values())) or 1.0
    return {bucket: value / norm for bucket, value in vec.items()}


def _cosine(a: SparseVector, b: SparseVector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(bucket, 0.0) for bucket, value in a.items())


@dataclass
class IndexStats:
    n_docs: int
    embed_ms: float
    index_build_ms: float
    backend: str


class RagBaseline:
    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._vectors: list[SparseVector] = []

    def build_index(self, docs: list[tuple[str, str]]) -> IndexStats:
        embed_start = time.perf_counter()
        vectors = [_embed(text) for _, text in docs]
        embed_ms = (time.perf_counter() - embed_start) * 1000.0

        build_start = time.perf_counter()
        self._doc_ids = [doc_id for doc_id, _ in docs]
        self._vectors = vectors
        index_build_ms = (time.perf_counter() - build_start) * 1000.0

        return IndexStats(len(docs), embed_ms, index_build_ms, "numpy-fallback")

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict], dict]:
        embed_start = time.perf_counter()
        query_vec = _embed(query)
        query_embed_ms = (time.perf_counter() - embed_start) * 1000.0

        ann_start = time.perf_counter()
        scored = heapq.nlargest(
            top_k,
            ((_cosine(query_vec, vec), doc_id) for vec, doc_id in zip(self._vectors, self._doc_ids)),
            key=lambda item: item[0],
        )
        ann_ms = (time.perf_counter() - ann_start) * 1000.0

        hits = [{"doc_id": doc_id, "score": score} for score, doc_id in scored[:top_k]]
        return hits, {"query_embed_ms": query_embed_ms, "ann_ms": ann_ms, "rerank_ms": 0.0}


def _demo_docs() -> list[tuple[str, str]]:
    try:
        from backend.state import demo_chunks

        return [(chunk.doc_id, chunk.text) for chunk in demo_chunks()]
    except Exception:
        return [
            ("doc-1", "network retry without exponential backoff in the connection pool"),
            ("doc-2", "information retrieval ranking metrics for search"),
            ("doc-3", "prefix cache reuse for interactive relevance filtering"),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG baseline timing record (eval only)")
    parser.add_argument("--query", default="retry without backoff")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    docs = _demo_docs()
    rag = RagBaseline()
    stats = rag.build_index(docs)
    hits, retrieve_ms = rag.retrieve(args.query, args.top_k)

    record = {
        "backend": stats.backend,
        "n_docs": stats.n_docs,
        "embed_ms": round(stats.embed_ms, 3),
        "index_build_ms": round(stats.index_build_ms, 3),
        "query_embed_ms": round(retrieve_ms["query_embed_ms"], 3),
        "ann_ms": round(retrieve_ms["ann_ms"], 3),
        "rerank_ms": retrieve_ms["rerank_ms"],
        "retrieve_ms_total": round(
            retrieve_ms["query_embed_ms"] + retrieve_ms["ann_ms"] + retrieve_ms["rerank_ms"], 3
        ),
        "top_hits": hits,
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
