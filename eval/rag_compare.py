from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from backend.state import demo_chunks
from baseline.rag import RagBaseline
from eval.bench import _git_commit


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODAL_ARTIFACT = ARTIFACT_DIR / "phase04_modal_openai_server_benchmark.json"
DEFAULT_OUTPUT_JSON = ARTIFACT_DIR / "phase04_rag_vs_6xh100.json"
DEFAULT_OUTPUT_MD = ARTIFACT_DIR / "phase04_rag_vs_6xh100.md"
DEFAULT_QUERY = "GPU queue saturation and throughput metrics"
DEFAULT_SIZES = [7, 100, 1_000, 5_000, 10_000, 25_000]


def run_rag_vs_6xh100(
    *,
    modal_artifact: Path = DEFAULT_MODAL_ARTIFACT,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    query: str = DEFAULT_QUERY,
    top_k: int = 5,
    sizes: list[int] | None = None,
    runs: int = 7,
) -> dict[str, Any]:
    six_h100 = json.loads(modal_artifact.read_text())
    rows = [
        _measure_rag_size(n_docs=size, query=query, top_k=top_k, runs=runs)
        for size in (sizes or DEFAULT_SIZES)
    ]
    payload = _comparison_payload(
        six_h100=six_h100,
        rows=rows,
        modal_artifact=modal_artifact,
        query=query,
        top_k=top_k,
        runs=runs,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown_report(payload) + "\n")
    return payload


def _measure_rag_size(*, n_docs: int, query: str, top_k: int, runs: int) -> dict[str, Any]:
    docs = _scaled_docs(n_docs)
    rag = RagBaseline()
    stats = rag.build_index(docs)
    index_total_ms = stats.embed_ms + stats.index_build_ms

    retrieve_runs_ms = []
    hits: list[dict[str, Any]] = []
    for _ in range(runs):
        run_start = time.perf_counter()
        hits, retrieve_ms = rag.retrieve(query, top_k)
        wall_ms = (time.perf_counter() - run_start) * 1000.0
        retrieve_runs_ms.append(
            {
                "query_embed_ms": retrieve_ms["query_embed_ms"],
                "ann_ms": retrieve_ms["ann_ms"],
                "rerank_ms": retrieve_ms["rerank_ms"],
                "retrieve_ms_total": retrieve_ms["query_embed_ms"] + retrieve_ms["ann_ms"] + retrieve_ms["rerank_ms"],
                "wall_ms": wall_ms,
            }
        )

    retrieve_totals = [run["retrieve_ms_total"] for run in retrieve_runs_ms]
    return {
        "backend": stats.backend,
        "n_docs": stats.n_docs,
        "index_embed_ms": stats.embed_ms,
        "index_build_ms": stats.index_build_ms,
        "index_total_ms": index_total_ms,
        "retrieve_ms_p50": statistics.median(retrieve_totals),
        "retrieve_ms_min": min(retrieve_totals),
        "retrieve_ms_max": max(retrieve_totals),
        "retrieve_ms_mean": statistics.fmean(retrieve_totals),
        "retrieve_runs_ms": retrieve_runs_ms,
        "single_process_retrieve_qps_p50": 1000.0 / max(statistics.median(retrieve_totals), 1e-9),
        "fresh_file_total_ms": index_total_ms + statistics.median(retrieve_totals),
        "top_hits": hits,
    }


def _scaled_docs(n_docs: int) -> list[tuple[str, str]]:
    base = [(chunk.doc_id, chunk.text) for chunk in demo_chunks()]
    return [
        (f"{doc_id}:copy-{idx}", text)
        for idx in range((n_docs + len(base) - 1) // len(base))
        for doc_id, text in base
    ][:n_docs]


def _comparison_payload(
    *,
    six_h100: dict[str, Any],
    rows: list[dict[str, Any]],
    modal_artifact: Path,
    query: str,
    top_k: int,
    runs: int,
) -> dict[str, Any]:
    vllm_client = six_h100["aggregate_client"]
    vllm_server = six_h100["aggregate_server"]
    vllm_p50_ms = float(vllm_client["latency_ms_p50_mean"])
    vllm_requests_per_s = float(vllm_client["requests_per_s"])
    compared_rows = []
    for row in rows:
        compared = dict(row)
        compared["rag_retrieve_latency_vs_vllm_p50"] = row["retrieve_ms_p50"] / vllm_p50_ms
        compared["rag_fresh_file_total_vs_vllm_p50"] = row["fresh_file_total_ms"] / vllm_p50_ms
        compared["rag_single_process_qps_vs_6xh100_vllm_qps"] = (
            row["single_process_retrieve_qps_p50"] / vllm_requests_per_s
        )
        compared_rows.append(compared)

    return {
        "run_id": f"rag-vs-6xh100-{int(time.time())}",
        "commit": _git_commit(),
        "query": query,
        "top_k": top_k,
        "runs_per_size": runs,
        "rag_backend": rows[0]["backend"] if rows else "unknown",
        "six_h100_reference": {
            "artifact": str(modal_artifact),
            "run_id": six_h100.get("run_id"),
            "replicas": six_h100.get("replicas"),
            "model": six_h100.get("model"),
            "vllm_version": six_h100.get("vllm_version"),
            "gpu_memory_utilization": six_h100.get("gpu_memory_utilization"),
            "requests_per_s": vllm_requests_per_s,
            "total_tokens_per_s": vllm_client.get("total_tokens_per_s"),
            "latency_ms_p50_mean": vllm_p50_ms,
            "latency_ms_p95_max": vllm_client.get("latency_ms_p95_max"),
            "derived_mfu_bf16_peak_mean": vllm_server.get("derived_mfu_bf16_peak_mean"),
            "estimated_tflops_per_gpu_mean": vllm_server.get("estimated_tflops_per_gpu_mean"),
        },
        "rag_rows": compared_rows,
        "caveats": [
            "RAG backend is the repo's pure-Python hashing-vectorizer fallback, not tuned FAISS or a neural embedding model.",
            "RAG numbers are retrieve-only: no LLM answer generation, no cross-encoder rerank, and no semantic continuity across refine turns.",
            "The 6xH100 reference is the vLLM OpenAI-server semantic scoring benchmark with max_tokens=1 and MFU metrics enabled.",
            "Fresh-file total models the structural RAG requirement to rebuild embeddings/index before the new document is retrievable.",
        ],
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    ref = payload["six_h100_reference"]
    lines = [
        "# Phase 04 RAG vs 6xH100 Baseline",
        "",
        f"- commit: `{payload['commit']}`",
        f"- query: `{payload['query']}`",
        f"- RAG backend: `{payload['rag_backend']}`",
        f"- 6xH100 run: `{ref['run_id']}`",
        f"- 6xH100 throughput: `{ref['requests_per_s']:.3f}` req/s, `{ref['total_tokens_per_s']:.3f}` tok/s",
        f"- 6xH100 latency: p50 mean `{ref['latency_ms_p50_mean']:.3f}` ms, p95 max `{ref['latency_ms_p95_max']:.3f}` ms",
        f"- 6xH100 derived MFU: `{ref['derived_mfu_bf16_peak_mean']:.6f}` of H100 BF16 peak",
        "",
        "| docs | RAG index ms | RAG retrieve p50 ms | RAG retrieve qps | fresh-file total ms | retrieve/vLLM p50 | fresh/vLLM p50 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rag_rows"]:
        lines.append(
            "| "
            f"{row['n_docs']} | "
            f"{row['index_total_ms']:.3f} | "
            f"{row['retrieve_ms_p50']:.3f} | "
            f"{row['single_process_retrieve_qps_p50']:.3f} | "
            f"{row['fresh_file_total_ms']:.3f} | "
            f"{row['rag_retrieve_latency_vs_vllm_p50']:.3f}x | "
            f"{row['rag_fresh_file_total_vs_vllm_p50']:.3f}x |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in payload["caveats"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG baseline timing against the 6xH100 vLLM benchmark.")
    parser.add_argument("--modal-artifact", type=Path, default=DEFAULT_MODAL_ARTIFACT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()

    payload = run_rag_vs_6xh100(
        modal_artifact=args.modal_artifact,
        output_json=args.output_json,
        output_md=args.output_md,
        query=args.query,
        top_k=args.top_k,
        sizes=args.sizes,
        runs=args.runs,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
