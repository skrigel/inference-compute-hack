from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from backend.state import demo_chunks
from baseline.rag import RagBaseline
from data.browsecomp_loader import browsecomp_docs
from eval.bench import GOLD_PREDICATES, _classification_metrics, _git_commit


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MODAL_ARTIFACT = ARTIFACT_DIR / "phase04_modal_openai_server_benchmark.json"
DEFAULT_OURS_QUALITY_ARTIFACT = ARTIFACT_DIR / "phase04_quality_gate.json"
DEFAULT_OUTPUT_JSON = ARTIFACT_DIR / "phase04_rag_vs_6xh100.json"
DEFAULT_OUTPUT_MD = ARTIFACT_DIR / "phase04_rag_vs_6xh100.md"
DEFAULT_QUERY = "GPU queue saturation and throughput metrics"
DEFAULT_SIZES = [7, 100, 1_000, 5_000, 10_000, 25_000, 100_000]


def run_rag_vs_6xh100(
    *,
    modal_artifact: Path = DEFAULT_MODAL_ARTIFACT,
    ours_quality_artifact: Path = DEFAULT_OURS_QUALITY_ARTIFACT,
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
    quality_comparison = _quality_comparison(
        rag_quality=evaluate_rag_quality(top_k=top_k),
        ours_quality_artifact=ours_quality_artifact,
    )
    payload = _comparison_payload(
        six_h100=six_h100,
        rows=rows,
        modal_artifact=modal_artifact,
        quality_comparison=quality_comparison,
        query=query,
        top_k=top_k,
        runs=runs,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown_report(payload) + "\n")
    return payload


def evaluate_rag_quality(*, top_k: int = 5) -> dict[str, Any]:
    chunks = demo_chunks()
    docs = [(chunk.doc_id, chunk.text) for chunk in chunks]
    title_by_doc_id = {chunk.doc_id: chunk.title for chunk in chunks}
    rag = RagBaseline()
    stats = rag.build_index(docs)

    y_true: list[bool] = []
    y_pred: list[bool] = []
    rows = []
    for spec in GOLD_PREDICATES:
        predicate = spec["predicate"]
        positive_titles = set(spec["positive_titles"])
        hits, timing = rag.retrieve(predicate, top_k=top_k)
        hit_doc_ids = {hit["doc_id"] for hit in hits}
        hit_titles = [title_by_doc_id[hit["doc_id"]] for hit in hits if hit["doc_id"] in title_by_doc_id]

        for chunk in chunks:
            y_true.append(chunk.title in positive_titles)
            y_pred.append(chunk.doc_id in hit_doc_ids)

        tp = len(set(hit_titles) & positive_titles)
        precision = tp / len(hit_titles) if hit_titles else 0.0
        recall = tp / len(positive_titles) if positive_titles else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "predicate": predicate,
                "positive_titles": sorted(positive_titles),
                "hit_titles": hit_titles,
                "precision_at_k": round(precision, 6),
                "recall_at_k": round(recall, 6),
                "f1_at_k": round(f1, 6),
                "retrieve_ms_total": timing["query_embed_ms"] + timing["ann_ms"] + timing["rerank_ms"],
            }
        )

    return {
        "backend": stats.backend,
        "corpus_size": len(chunks),
        "top_k": top_k,
        "quality": _classification_metrics(y_true, y_pred),
        "rows": rows,
    }


def _quality_comparison(*, rag_quality: dict[str, Any], ours_quality_artifact: Path) -> dict[str, Any]:
    ours = _load_ours_quality(ours_quality_artifact)
    payload: dict[str, Any] = {"rag": rag_quality, "ours": ours}
    if ours.get("status") == "measured":
        payload["delta"] = {
            "recall": ours["quality"]["recall"] - rag_quality["quality"]["recall"],
            "precision": ours["quality"]["precision"] - rag_quality["quality"]["precision"],
            "f1": ours["quality"]["f1"] - rag_quality["quality"]["f1"],
        }
    return payload


def _load_ours_quality(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "artifact": str(path),
            "note": "Run `python -m eval.bench --backend modal --gate-only --force` to populate measured ours precision/recall/F1.",
        }
    payload = json.loads(path.read_text())
    return {
        "status": "measured",
        "artifact": str(path),
        "run_id": payload.get("run_id"),
        "backend": payload.get("scorer_backend"),
        "model_id": payload.get("model_id"),
        "threshold": payload.get("threshold"),
        "quality": payload.get("quality", {}),
        "recommended_threshold": payload.get("recommended_threshold"),
        "threshold_sweep_best": (payload.get("threshold_sweep") or {}).get("best"),
    }


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
    """Return n_docs (doc_id, text) tuples from BrowseComp-Plus corpus.

    Uses the ~100k BrowseComp-Plus corpus directly. If n_docs exceeds corpus
    size, replicates with copy suffixes.
    """
    base = browsecomp_docs()
    if n_docs <= len(base):
        return base[:n_docs]
    # Replicate if needed (unlikely given 100k corpus)
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
    quality_comparison: dict[str, Any],
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
        "quality_comparison": quality_comparison,
        "optimization_findings": identify_optimization_findings(six_h100, quality_comparison),
        "rag_rows": compared_rows,
        "caveats": [
            "RAG backend is the repo's pure-Python hashing-vectorizer fallback, not tuned FAISS or a neural embedding model.",
            "RAG numbers are retrieve-only: no LLM answer generation, no cross-encoder rerank, and no semantic continuity across refine turns.",
            "The 6xH100 reference is the vLLM OpenAI-server semantic scoring benchmark with max_tokens=1 and MFU metrics enabled.",
            "Fresh-file total models the structural RAG requirement to rebuild embeddings/index before the new document is retrievable.",
        ],
    }


def identify_optimization_findings(six_h100: dict[str, Any], quality_comparison: dict[str, Any]) -> list[dict[str, Any]]:
    client = six_h100.get("aggregate_client", {})
    server = six_h100.get("aggregate_server", {})
    findings = []

    mfu = server.get("derived_mfu_bf16_peak_mean")
    if mfu is not None and float(mfu) < 0.20:
        findings.append(
            {
                "name": "low_mfu_short_request_overhead",
                "evidence": {
                    "derived_mfu_bf16_peak_mean": mfu,
                    "server_queue_avg_ms_mean": server.get("server_queue_avg_ms_mean"),
                    "server_prefill_avg_ms_mean": server.get("server_prefill_avg_ms_mean"),
                    "server_e2e_latency_avg_ms_mean": server.get("server_e2e_latency_avg_ms_mean"),
                },
                "bottleneck": "The H100s are not queue-bound; short one-token scoring requests leave most peak FLOPs unused.",
                "fix": "Use compact scoring prompts, then sweep concurrency/max-num-batched-tokens upward after the quality gate passes.",
                "status": "compact benchmark prompt added; production prompt should only change after measured quality is present.",
            }
        )

    if quality_comparison["ours"].get("status") != "measured":
        findings.append(
            {
                "name": "quality_gate_missing_for_speed_claim",
                "evidence": quality_comparison["ours"],
                "bottleneck": "The six-H100 throughput artifact does not prove recall, so optimization could speed up the wrong behavior.",
                "fix": "Run the Modal quality gate and feed `phase04_quality_gate.json` into this comparison.",
                "status": "comparison now records the missing quality artifact explicitly.",
            }
        )
    else:
        quality = quality_comparison["ours"].get("quality", {})
        best = quality_comparison["ours"].get("threshold_sweep_best") or {}
        best_quality = best.get("quality") or {}
        current_threshold = quality_comparison["ours"].get("threshold")
        best_threshold = best.get("threshold")
        threshold_delta = abs(float(current_threshold) - float(best_threshold)) if current_threshold is not None and best_threshold is not None else None
        if best_quality.get("f1", 0.0) > quality.get("f1", 0.0) or threshold_delta == 0.0:
            findings.append(
                {
                    "name": "threshold_calibration",
                    "evidence": {
                        "current_threshold": current_threshold,
                        "current_quality": quality,
                        "recommended_threshold": best_threshold,
                        "recommended_quality": best_quality,
                    },
                    "bottleneck": "The model separates positives, but the default 0.5 threshold throws away recall.",
                    "fix": "Use the recommended threshold for the demo operating point, then validate on a larger gold set.",
                    "status": "applied" if threshold_delta == 0.0 else "threshold sweep added; rerun quality gate with the recommended threshold to make it active.",
                }
            )

    prompt_tokens_per_request = None
    if client.get("prompt_tokens_per_s") and client.get("requests_per_s"):
        prompt_tokens_per_request = client["prompt_tokens_per_s"] / client["requests_per_s"]
    if prompt_tokens_per_request is not None and prompt_tokens_per_request > 32:
        findings.append(
            {
                "name": "prompt_prefill_cost",
                "evidence": {"prompt_tokens_per_request": prompt_tokens_per_request},
                "bottleneck": "Most work is prompt prefill for a one-token classifier.",
                "fix": "Default future benchmark runs to a compact prompt variant and report prompt tokens per request.",
                "status": "applied in Modal OpenAI-server benchmark path.",
            }
        )
    return findings


def _markdown_report(payload: dict[str, Any]) -> str:
    ref = payload["six_h100_reference"]
    rag_quality = payload["quality_comparison"]["rag"]["quality"]
    ours_quality = payload["quality_comparison"]["ours"]
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
        f"- RAG recall@{payload['top_k']}: `{rag_quality['recall']:.6f}`; precision@{payload['top_k']}: `{rag_quality['precision']:.6f}`; F1@{payload['top_k']}: `{rag_quality['f1']:.6f}`",
    ]
    if ours_quality.get("status") == "measured":
        quality = ours_quality["quality"]
        lines.append(
            f"- Ours quality: recall `{quality['recall']:.6f}`; precision `{quality['precision']:.6f}`; F1 `{quality['f1']:.6f}`"
        )
        best = ours_quality.get("threshold_sweep_best") or {}
        if best:
            best_quality = best["quality"]
            lines.append(
                f"- Ours recommended threshold: `{best['threshold']}` gives recall `{best_quality['recall']:.6f}`; precision `{best_quality['precision']:.6f}`; F1 `{best_quality['f1']:.6f}`"
            )
    else:
        lines.append(f"- Ours quality: `{ours_quality['status']}` ({ours_quality['note']})")
    lines.extend(
        [
            "",
            "| docs | RAG index ms | RAG retrieve p50 ms | RAG retrieve qps | fresh-file total ms | retrieve/vLLM p50 | fresh/vLLM p50 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
    lines.extend(["", "## Optimization Findings", ""])
    for finding in payload["optimization_findings"]:
        lines.append(f"- **{finding['name']}**: {finding['bottleneck']} Fix: {finding['fix']} Status: {finding['status']}")
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in payload["caveats"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG baseline timing against the 6xH100 vLLM benchmark.")
    parser.add_argument("--modal-artifact", type=Path, default=DEFAULT_MODAL_ARTIFACT)
    parser.add_argument("--ours-quality-artifact", type=Path, default=DEFAULT_OURS_QUALITY_ARTIFACT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()

    payload = run_rag_vs_6xh100(
        modal_artifact=args.modal_artifact,
        ours_quality_artifact=args.ours_quality_artifact,
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
