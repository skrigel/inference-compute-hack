from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from backend.state import demo_chunks
from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.config import make_scorer
from inference.scorer import ScoreRequest
from eval.cut_line import DEMO_QUERY, run_cut_line
from eval.trace import TurnTrace
from eval.weave_ops import DEFAULT_WEAVE_PROJECT, init_weave, weave_op


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
QUALITY_THRESHOLD = 0.5
MIN_F1 = 0.7

GOLD_PREDICATES = [
    {
        "predicate": DEMO_QUERY,
        "positive_titles": {
            "urllib3/connectionpool.py",
            "requests/adapters.py",
            "aiohttp/client.py",
            "app/db_session.py",
            "jobs/worker.py",
        },
    },
    {
        "predicate": "only in the networking layer",
        "positive_titles": {
            "urllib3/connectionpool.py",
            "requests/adapters.py",
            "aiohttp/client.py",
        },
    },
    {
        "predicate": "retrieval ranking metrics",
        "positive_titles": {"Neural Retrieval for Code Search"},
    },
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _active_backend() -> str:
    return os.environ.get("SCORER_BACKEND", "mock").lower()


def _set_backend(backend: str) -> None:
    os.environ["SCORER_BACKEND"] = backend


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
    backend = _active_backend()
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

    trace = TurnTrace(
        run_id=f"smoke-{uuid.uuid4().hex[:8]}",
        commit=_git_commit(),
        corpus_id="phase0-smoke",
        model_id=scorer.model_id(),
        scorer_backend=backend,
        turn=1,
        operation="query",
        threshold=threshold,
        n_chunks_total=n_chunks_total,
        candidate_count=n_chunks_total,
        chunks_scored=chunks_scored,
        chunks_served_from_cache=chunks_served_from_cache,
        survivor_count=survivor_count,
        elapsed_ms=elapsed_ms,
        model_ms=elapsed_ms,
        queue_ms=0.0,
        ttft_ms=0.0,
        cache_hit_rate=0.0,
        gpu_cache_usage_perc=0.0,
        quality_slice=None,
    )
    payload = trace.to_dict()
    payload["warm_state"] = "cold"
    payload["latency_kind"] = "cold"
    return payload


@weave_op(name="eval.run_smoke")
async def run_smoke_traced() -> dict:
    return await run_smoke()


def run_smoke_with_weave(project: str = DEFAULT_WEAVE_PROJECT) -> dict:
    init_weave(project)
    return asyncio.run(run_smoke_traced())


async def run_quality_gate(
    backend: str,
    *,
    artifact_dir: Path = ARTIFACT_DIR,
    threshold: float = QUALITY_THRESHOLD,
    force: bool = False,
) -> dict:
    _set_backend(backend)
    scorer = make_scorer()
    chunks = demo_chunks()
    rows = []
    y_true = []
    y_pred = []
    scores = []

    for spec in GOLD_PREDICATES:
        predicate = spec["predicate"]
        positive_titles = spec["positive_titles"]
        requests = [ScoreRequest(chunk.chunk_id, chunk.text, predicate) for chunk in chunks]
        results = await scorer.score_batch(requests)
        by_id = {result.chunk_id: result for result in results}
        for chunk in chunks:
            result = by_id[chunk.chunk_id]
            expected = chunk.title in positive_titles
            predicted = result.score >= threshold
            y_true.append(expected)
            y_pred.append(predicted)
            scores.append(result.score)
            rows.append(
                {
                    "predicate": predicate,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "expected": expected,
                    "score": result.score,
                    "predicted": predicted,
                }
            )

    quality = _classification_metrics(y_true, y_pred)
    payload = {
        "run_id": f"quality-{uuid.uuid4().hex[:8]}",
        "commit": _git_commit(),
        "scorer_backend": backend,
        "model_id": scorer.model_id(),
        "corpus_id": "demo",
        "corpus_size": len(chunks),
        "threshold": threshold,
        "quality": quality,
        "small_gate": True,
        "n_cases": len(rows),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "rows": rows,
        "label": "measured" if backend in {"vllm", "modal"} else "mock/projected",
    }
    if quality["f1"] < MIN_F1 and not force:
        _write_quality_artifacts(payload, artifact_dir)
        raise RuntimeError(f"Phase 04 quality gate failed: F1 {quality['f1']:.3f} < {MIN_F1:.3f}")
    _write_quality_artifacts(payload, artifact_dir)
    return payload


@weave_op(name="eval.run_quality_gate")
async def run_quality_gate_traced(backend: str, *, artifact_dir: Path = ARTIFACT_DIR, force: bool = False) -> dict:
    return await run_quality_gate(backend, artifact_dir=artifact_dir, force=force)


def run_quality_gate_with_weave(
    backend: str,
    project: str = DEFAULT_WEAVE_PROJECT,
    *,
    artifact_dir: Path = ARTIFACT_DIR,
    force: bool = False,
) -> dict:
    init_weave(project)
    return asyncio.run(run_quality_gate_traced(backend, artifact_dir=artifact_dir, force=force))


async def run_freeze(
    backend: str,
    *,
    artifact_dir: Path = ARTIFACT_DIR,
    tag: str = "freeze",
    force: bool = False,
) -> dict:
    gate = await run_quality_gate(backend, artifact_dir=artifact_dir, force=force)
    _set_backend(backend)
    scorer = make_scorer()
    metrics = await _collect_metrics(scorer)
    result = run_cut_line(scorer, label=f"measured ({backend})")
    if not result.green:
        raise RuntimeError(f"Phase 04 freeze loop failed: {result.failures}")

    run_id = f"phase04-{uuid.uuid4().hex[:8]}"
    gpu_cache_usage = _gpu_cache_usage(metrics)
    rows = [
        _trace_row(
            run_id=run_id,
            backend=backend,
            model_id=scorer.model_id(),
            n_chunks=result.n_chunks,
            step=step,
            gpu_cache_usage_perc=gpu_cache_usage,
        )
        for step in result.steps
    ]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "phase04_vllm_trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    payload = {
        "run_id": run_id,
        "tag": tag,
        "commit": _git_commit(),
        "scorer_backend": backend,
        "model_id": scorer.model_id(),
        "corpus_id": "demo",
        "corpus_size": result.n_chunks,
        "quality_gate": {
            "f1": gate["quality"]["f1"],
            "threshold": gate["threshold"],
            "artifact": "phase04_quality_gate.json",
        },
        "scoped_cumulative_chunks": result.area_under_loop["scoped_total"],
        "full_cumulative_chunks": result.area_under_loop["full_total"],
        "area_under_loop": result.area_under_loop,
        "fresh_vs_rag": result.fresh_vs_rag,
        "vllm_metrics": metrics,
        "label": "measured" if backend in {"vllm", "modal"} else "mock/projected",
        "caveats": _freeze_caveats(backend),
    }
    (artifact_dir / "phase04_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (artifact_dir / "phase04_gpu_memory_sweep.json").write_text(
        json.dumps(
            [
                {
                    "gpu_memory_utilization": os.environ.get("GPU_MEMORY_UTILIZATION"),
                    "backend": backend,
                    "metrics": metrics,
                    "note": "Populate with Modal/vLLM sweep results after each real run.",
                }
            ],
            indent=2,
            sort_keys=True,
        )
    )
    _write_environment_md(artifact_dir / "phase04_environment.md", backend, scorer.model_id(), metrics)
    return payload


@weave_op(name="eval.run_freeze")
async def run_freeze_traced(backend: str, *, artifact_dir: Path = ARTIFACT_DIR, tag: str = "freeze", force: bool = False) -> dict:
    return await run_freeze(backend, artifact_dir=artifact_dir, tag=tag, force=force)


def run_freeze_with_weave(
    backend: str,
    project: str = DEFAULT_WEAVE_PROJECT,
    *,
    artifact_dir: Path = ARTIFACT_DIR,
    tag: str = "freeze",
    force: bool = False,
) -> dict:
    init_weave(project)
    return asyncio.run(run_freeze_traced(backend, artifact_dir=artifact_dir, tag=tag, force=force))


def _classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected and predicted)
    fp = sum(1 for expected, predicted in zip(y_true, y_pred) if not expected and predicted)
    fn = sum(1 for expected, predicted in zip(y_true, y_pred) if expected and not predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _write_quality_artifacts(payload: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "phase04_quality_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    quality = payload["quality"]
    md = "\n".join(
        [
            "# Phase 04 Quality Gate",
            "",
            f"- backend: `{payload['scorer_backend']}`",
            f"- model: `{payload['model_id']}`",
            f"- commit: `{payload['commit']}`",
            f"- corpus: `{payload['corpus_id']}` ({payload['corpus_size']} chunks)",
            f"- threshold: `{payload['threshold']}`",
            f"- precision: `{quality['precision']}`",
            f"- recall: `{quality['recall']}`",
            f"- f1: `{quality['f1']}`",
            f"- label: `{payload['label']}`",
            "",
            "This is a small Phase 04 gate over the pinned demo corpus and scripted predicates.",
            "",
        ]
    )
    (artifact_dir / "phase04_quality_gate.md").write_text(md)


async def _collect_metrics(scorer: Any) -> dict:
    collect = getattr(scorer, "collect_metrics", None)
    if collect is None:
        return {}
    try:
        return await collect()
    except Exception as exc:
        return {"collection_error": str(exc)}


def _gpu_cache_usage(metrics: dict) -> float:
    values = []
    for replica_metrics in metrics.values() if isinstance(metrics, dict) else []:
        if not isinstance(replica_metrics, dict):
            continue
        for key in ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"):
            if key in replica_metrics:
                values.append(float(replica_metrics[key]))
    return sum(values) / len(values) if values else 0.0


def _trace_row(
    *,
    run_id: str,
    backend: str,
    model_id: str,
    n_chunks: int,
    step,
    gpu_cache_usage_perc: float,
) -> dict:
    candidate_count = max(step.candidate_count, step.matched, step.chunks_scored)
    n_chunks_total = max(n_chunks, candidate_count)
    chunks_from_cache = max(0, candidate_count - step.chunks_scored)
    trace = TurnTrace(
        run_id=run_id,
        commit=_git_commit(),
        corpus_id="demo",
        model_id=model_id,
        scorer_backend=backend,
        turn=step.step,
        operation=step.operation,
        threshold=QUALITY_THRESHOLD,
        n_chunks_total=n_chunks_total,
        candidate_count=candidate_count,
        chunks_scored=step.chunks_scored,
        chunks_served_from_cache=chunks_from_cache,
        survivor_count=step.matched,
        elapsed_ms=float(step.refine_ms),
        model_ms=float(step.refine_ms if step.chunks_scored else 0.0),
        queue_ms=0.0,
        ttft_ms=0.0,
        cache_hit_rate=chunks_from_cache / candidate_count if candidate_count else 0.0,
        gpu_cache_usage_perc=gpu_cache_usage_perc,
        quality_slice=None,
    )
    payload = trace.to_dict()
    payload["warm_state"] = "cold" if step.latency_kind == "cold" else "warm"
    payload["latency_kind"] = step.latency_kind
    payload["note"] = step.note
    return payload


def _freeze_caveats(backend: str) -> list[str]:
    if backend in {"vllm", "modal"}:
        return [
            "Trace rows are measured on the selected real scorer backend.",
            "GPU/cache metrics are present only when the backend exposes vLLM /metrics.",
        ]
    return ["Mock backend artifact; use only as structural/projected proof, not a real vLLM speed claim."]


def _write_environment_md(path: Path, backend: str, model_id: str, metrics: dict) -> None:
    lines = [
        "# Phase 04 Environment",
        "",
        f"- backend: `{backend}`",
        f"- model: `{model_id}`",
        f"- commit: `{_git_commit()}`",
        f"- recorded_at_unix: `{time.time():.3f}`",
        f"- python: `{os.sys.version.split()[0]}`",
        f"- vllm_replicas: `{os.environ.get('VLLM_REPLICAS', '')}`",
        f"- gpu_memory_utilization: `{os.environ.get('GPU_MEMORY_UTILIZATION', '')}`",
        f"- metrics_available: `{bool(metrics)}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 evaluation harness")
    parser.add_argument("--smoke", action="store_true", help="emit one required trace row as JSON")
    parser.add_argument("--weave", action="store_true", help="log the smoke eval as a W&B Weave trace")
    parser.add_argument("--backend", choices=["mock", "modal", "vllm"], default=os.environ.get("SCORER_BACKEND", "mock"))
    parser.add_argument("--gate-only", action="store_true", help="run the Phase 04 quality gate only")
    parser.add_argument("--tag", default=None, help="run a named Phase 04 freeze, e.g. --tag freeze")
    parser.add_argument("--force", action="store_true", help="write artifacts even if the quality gate is below threshold")
    parser.add_argument(
        "--weave-project",
        default=os.environ.get("WEAVE_PROJECT", DEFAULT_WEAVE_PROJECT),
        help="W&B team/project for Weave traces",
    )
    args = parser.parse_args()
    _set_backend(args.backend)
    try:
        if args.gate_only:
            payload = (
                run_quality_gate_with_weave(args.backend, args.weave_project, force=args.force)
                if args.weave
                else asyncio.run(run_quality_gate(args.backend, force=args.force))
            )
        elif args.tag is not None:
            payload = (
                run_freeze_with_weave(args.backend, args.weave_project, tag=args.tag, force=args.force)
                if args.weave
                else asyncio.run(run_freeze(args.backend, tag=args.tag, force=args.force))
            )
        elif args.smoke:
            payload = run_smoke_with_weave(args.weave_project) if args.weave else asyncio.run(run_smoke())
        else:
            parser.error("expected --smoke, --gate-only, or --tag freeze")
    except RuntimeError as exc:
        parser.exit(2, f"eval.bench: {exc}\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
