from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import uuid

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.config import make_scorer
from inference.scorer import ScoreRequest


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _seed_chunks() -> list[Chunk]:
    rows = [
        (
            "demo:retry",
            "urllib3 connectionpool retry",
            "code",
            "network retry happens here without exponential backoff",
            ChunkMeta("python", 2024, "urllib3/connectionpool.py", "python", "urllib3", "synthetic"),
        ),
        (
            "demo:ranking",
            "IR ranking abstract",
            "paper",
            "this paper studies ranking metrics for information retrieval",
            ChunkMeta("cs.IR", 2023, None, None, None, "synthetic"),
        ),
        (
            "demo:backoff",
            "httpx retry helper",
            "code",
            "http retry uses exponential backoff in the networking layer",
            ChunkMeta("python", 2024, "httpx/retry.py", "python", "httpx", "synthetic"),
        ),
    ]
    chunks: list[Chunk] = []
    for idx, (doc_id, title, chunk_type, text, meta) in enumerate(rows):
        chunks.append(
            Chunk(
                chunk_id=chunk_id_of(doc_id, idx, text),
                doc_id=doc_id,
                type=chunk_type,  # type: ignore[arg-type]
                title=title,
                text=text,
                meta=meta,
            )
        )
    return chunks


async def run_smoke() -> dict:
    scorer = make_scorer()
    chunks = _seed_chunks()
    predicate = "retry without backoff"
    threshold = 0.5
    requests = [ScoreRequest(c.chunk_id, c.text, predicate) for c in chunks]

    started = time.perf_counter()
    results = await scorer.score_batch(requests)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    survivor_count = sum(1 for result in results if result.score >= threshold)
    n_chunks_total = len(chunks)
    chunks_scored = len(requests)
    chunks_served_from_cache = 0

    return {
        "run_id": f"smoke-{uuid.uuid4().hex[:8]}",
        "commit": _git_commit(),
        "corpus_id": "phase0-smoke",
        "model_id": scorer.model_id(),
        "scorer_backend": "mock",
        "turn": 1,
        "operation": "query",
        "threshold": threshold,
        "n_chunks_total": n_chunks_total,
        "candidate_count": n_chunks_total,
        "chunks_scored": chunks_scored,
        "chunks_served_from_cache": chunks_served_from_cache,
        "survivor_count": survivor_count,
        "rho": survivor_count / n_chunks_total,
        "elapsed_ms": elapsed_ms,
        "model_ms": elapsed_ms,
        "queue_ms": 0.0,
        "cache_hit_rate": 0.0,
        "warm_state": "cold",
        "latency_kind": "cold",
        "quality_slice": "phase0-smoke-retry-without-backoff",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 evaluation harness")
    parser.add_argument("--smoke", action="store_true", help="emit one required trace row as JSON")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Phase 0 only supports --smoke")
    print(json.dumps(asyncio.run(run_smoke()), sort_keys=True))


if __name__ == "__main__":
    main()
