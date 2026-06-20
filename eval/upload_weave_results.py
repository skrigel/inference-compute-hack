from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.bench import _git_commit
from eval.weave_ops import DEFAULT_WEAVE_PROJECT, init_weave, weave_op


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_MATRIX_ARTIFACT = ARTIFACT_DIR / "phase04_h100_rag_matrix.json"
DEFAULT_QUALITY_ARTIFACT = ARTIFACT_DIR / "phase04_quality_gate.json"
DEFAULT_RAG_VS_ARTIFACT = ARTIFACT_DIR / "phase04_rag_vs_6xh100.json"
DEFAULT_RECEIPT = ARTIFACT_DIR / "phase04_weave_upload_receipt.json"


@weave_op(name="eval.phase04.upload_results_trace", kind="agent", color="purple")
def trace_phase04_results(
    summary: dict[str, Any],
    h100_rows: list[dict[str, Any]],
    rag_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    for row in h100_rows:
        trace_h100_scenario(row)
    for row in rag_rows:
        trace_rag_reference(row)
    for row in comparison_rows:
        trace_h100_rag_comparison(row)
    return summary


@weave_op(name="eval.phase04.h100_scenario", kind="tool", color="blue")
def trace_h100_scenario(row: dict[str, Any]) -> dict[str, Any]:
    return row


@weave_op(name="eval.phase04.rag_reference", kind="search", color="green")
def trace_rag_reference(row: dict[str, Any]) -> dict[str, Any]:
    return row


@weave_op(name="eval.phase04.h100_rag_comparison", kind="tool", color="orange")
def trace_h100_rag_comparison(row: dict[str, Any]) -> dict[str, Any]:
    return row


def load_phase04_payloads(
    *,
    matrix_artifact: Path = DEFAULT_MATRIX_ARTIFACT,
    quality_artifact: Path = DEFAULT_QUALITY_ARTIFACT,
    rag_vs_artifact: Path = DEFAULT_RAG_VS_ARTIFACT,
) -> dict[str, Any]:
    matrix = json.loads(matrix_artifact.read_text())
    return {
        "matrix": matrix,
        "quality": _load_optional_json(quality_artifact),
        "rag_vs_6xh100": _load_optional_json(rag_vs_artifact),
        "artifact_paths": {
            "matrix": _portable_path(matrix_artifact),
            "quality": _portable_path(quality_artifact),
            "rag_vs_6xh100": _portable_path(rag_vs_artifact),
        },
    }


def build_upload_bundle(payloads: dict[str, Any]) -> dict[str, Any]:
    matrix = payloads["matrix"]
    quality = payloads.get("quality") or {}
    rag_vs = payloads.get("rag_vs_6xh100") or {}
    h100_rows = _h100_rows(matrix)
    rag_rows = _rag_rows(matrix)
    comparison_rows = _comparison_rows(matrix)
    summary = _summary(matrix=matrix, quality=quality, rag_vs=rag_vs, h100_rows=h100_rows, rag_rows=rag_rows)
    return {
        "summary": summary,
        "h100_rows": h100_rows,
        "rag_rows": rag_rows,
        "comparison_rows": comparison_rows,
        "artifact_paths": payloads["artifact_paths"],
    }


def upload_phase04_results(
    bundle: dict[str, Any],
    *,
    project: str = DEFAULT_WEAVE_PROJECT,
) -> dict[str, Any]:
    import weave

    summary = bundle["summary"]
    global_attributes = {
        "project_area": "phase04-performance",
        "result_type": "h100-rag-matrix",
        "repo_commit": summary["commit"],
        "matrix_run_id": summary["matrix_run_id"],
        "model": summary["model"],
        "vllm_version": summary["vllm_version"],
        "weave_schema_version": "phase04.results.v1",
    }
    init_weave(project, global_attributes=global_attributes)

    with weave.attributes({"weave_upload_kind": "trace", **global_attributes}):
        trace_phase04_results(
            bundle["summary"],
            bundle["h100_rows"],
            bundle["rag_rows"],
            bundle["comparison_rows"],
        )

    eval_name = summary["matrix_run_id"]
    evaluation = weave.EvaluationLogger(
        name=eval_name,
        model=summary["model"],
        dataset="phase04-h100-rag-static-dynamic-matrix",
        eval_attributes={**global_attributes, "weave_upload_kind": "evaluation"},
        scorers=[
            "requests_per_s",
            "latency_ms_p50",
            "latency_ms_p95",
            "gpu_utilization_pct_mean",
            "derived_mfu_bf16_peak",
            "rag_latency_over_h100_p50",
            "h100_qps_over_rag_single_process_qps",
        ],
    )
    for row in bundle["h100_rows"]:
        score_logger = evaluation.log_prediction(
            inputs={
                "row_type": "h100_scenario",
                "scenario": row["scenario"],
                "dataset_mode": row["dataset_mode"],
                "gpu_count": row["gpu_count"],
                "concurrency": row["concurrency"],
                "num_requests": row["num_requests"],
            },
            output=row,
        )
        score_logger.log_score("requests_per_s", row["requests_per_s"])
        score_logger.log_score("latency_ms_p50", row["latency_ms_p50"])
        score_logger.log_score("latency_ms_p95", row["latency_ms_p95"])
        score_logger.log_score("gpu_utilization_pct_mean", row["gpu_utilization_pct_mean"])
        score_logger.log_score("derived_mfu_bf16_peak", row["derived_mfu_bf16_peak"])

    for row in bundle["comparison_rows"]:
        score_logger = evaluation.log_prediction(
            inputs={
                "row_type": "h100_rag_comparison",
                "scenario": row["scenario"],
                "gpu_count": row["h100_replicas"],
                "rag_n_docs": row["rag_n_docs"],
                "rag_metric": row["rag_metric"],
            },
            output=row,
        )
        score_logger.log_score("rag_latency_over_h100_p50", row["rag_latency_over_h100_p50"])
        score_logger.log_score("h100_qps_over_rag_single_process_qps", row["h100_qps_over_rag_single_process_qps"])

    evaluation.log_summary(summary)
    return {
        "project": project,
        "eval_name": eval_name,
        "matrix_run_id": summary["matrix_run_id"],
        "h100_rows_logged": len(bundle["h100_rows"]),
        "rag_rows_logged": len(bundle["rag_rows"]),
        "comparison_rows_logged": len(bundle["comparison_rows"]),
        "uploaded_at_unix": int(time.time()),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _h100_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_by_name = {scenario["name"]: scenario for scenario in matrix["scenarios"]}
    rows = []
    for scenario_name, by_gpu_count in sorted(matrix["h100_results"].items()):
        scenario = scenario_by_name[scenario_name]
        for gpu_count, result in sorted(by_gpu_count.items(), key=lambda item: int(item[0])):
            client = result["aggregate_client"]
            server = result["aggregate_server"]
            rows.append(
                {
                    "scenario": scenario_name,
                    "dataset_mode": scenario["dataset_mode"],
                    "gpu_count": int(gpu_count),
                    "num_requests": scenario["num_requests"],
                    "concurrency": scenario["concurrency"],
                    "requests_per_s": client["requests_per_s"],
                    "prompt_tokens_per_s": client["prompt_tokens_per_s"],
                    "total_tokens_per_s": client["total_tokens_per_s"],
                    "latency_ms_p50": client["latency_ms_p50_mean"],
                    "latency_ms_p95": client["latency_ms_p95_max"],
                    "latency_ms_p99": client["latency_ms_p99_max"],
                    "derived_mfu_bf16_peak": server.get("derived_mfu_bf16_peak_mean"),
                    "estimated_tflops_per_gpu": server.get("estimated_tflops_per_gpu_mean"),
                    "gpu_utilization_pct_mean": server.get("gpu_utilization_pct_mean"),
                    "gpu_utilization_pct_max": server.get("gpu_utilization_pct_max"),
                    "gpu_memory_used_mb_max": server.get("gpu_memory_used_mb_max"),
                    "gpu_memory_utilization_pct_max": server.get("gpu_memory_utilization_pct_max"),
                    "gpu_power_w_mean": server.get("gpu_power_w_mean"),
                    "gpu_power_w_max": server.get("gpu_power_w_max"),
                    "kv_cache_usage_perc_max": server.get("kv_cache_usage_perc_max"),
                    "server_queue_avg_ms": server.get("server_queue_avg_ms_mean"),
                    "server_prefill_avg_ms": server.get("server_prefill_avg_ms_mean"),
                    "server_ttft_avg_ms": server.get("server_ttft_avg_ms_mean"),
                }
            )
    return rows


def _rag_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in matrix["rag_reference"]["rows"]:
        rows.append(
            {
                "backend": row["backend"],
                "n_docs": row["n_docs"],
                "retrieve_ms_p50": row["retrieve_ms_p50"],
                "fresh_file_total_ms": row["fresh_file_total_ms"],
                "single_process_retrieve_qps_p50": row["single_process_retrieve_qps_p50"],
                "index_total_ms": row["index_total_ms"],
            }
        )
    return rows


def _comparison_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "scenario": row["scenario"],
                "h100_replicas": row["h100_replicas"],
                "rag_n_docs": row["rag_n_docs"],
                "rag_metric": row["rag_metric"],
                "h100_latency_ms_p50": row["h100_latency_ms_p50"],
                "rag_latency_ms": row["rag_latency_ms"],
                "rag_latency_over_h100_p50": row["rag_latency_over_h100_p50"],
                "h100_requests_per_s": row["h100_requests_per_s"],
                "rag_single_process_qps": row["rag_single_process_qps"],
                "h100_qps_over_rag_single_process_qps": row["h100_qps_over_rag_single_process_qps"],
            }
            for row in matrix["comparisons"]
        ),
        key=lambda row: (row["scenario"], row["h100_replicas"], row["rag_n_docs"]),
    )


def _summary(
    *,
    matrix: dict[str, Any],
    quality: dict[str, Any],
    rag_vs: dict[str, Any],
    h100_rows: list[dict[str, Any]],
    rag_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best_difference = max(matrix["comparisons"], key=lambda row: row["rag_latency_over_h100_p50"])
    best_throughput = max(h100_rows, key=lambda row: row["requests_per_s"])
    best_rag = max(rag_rows, key=lambda row: row["fresh_file_total_ms"])
    return {
        "commit": _git_commit(),
        "matrix_run_id": matrix["run_id"],
        "model": matrix["model"],
        "vllm_version": matrix["vllm_version"],
        "gpu_counts": matrix["gpu_counts"],
        "prompt_variant": matrix["prompt_variant"],
        "gpu_memory_utilization": matrix["gpu_memory_utilization"],
        "max_num_batched_tokens": matrix["max_num_batched_tokens"],
        "h100_row_count": len(h100_rows),
        "rag_row_count": len(rag_rows),
        "comparison_row_count": len(matrix["comparisons"]),
        "best_h100_throughput_scenario": best_throughput["scenario"],
        "best_h100_throughput_gpu_count": best_throughput["gpu_count"],
        "best_h100_requests_per_s": best_throughput["requests_per_s"],
        "largest_rag_latency_scenario": best_difference["scenario"],
        "largest_rag_latency_gpu_count": best_difference["h100_replicas"],
        "largest_rag_latency_docs": best_difference["rag_n_docs"],
        "largest_rag_latency_over_h100_p50": best_difference["rag_latency_over_h100_p50"],
        "largest_rag_fresh_file_total_ms": best_rag["fresh_file_total_ms"],
        "quality_run_id": quality.get("run_id"),
        "quality_precision": (quality.get("quality") or {}).get("precision"),
        "quality_recall": (quality.get("quality") or {}).get("recall"),
        "quality_f1": (quality.get("quality") or {}).get("f1"),
        "quality_threshold": quality.get("threshold"),
        "rag_vs_6xh100_run_id": rag_vs.get("run_id"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Upload Phase 04 H100/RAG results to W&B Weave.")
    parser.add_argument("--project", default=DEFAULT_WEAVE_PROJECT)
    parser.add_argument("--matrix-artifact", type=Path, default=DEFAULT_MATRIX_ARTIFACT)
    parser.add_argument("--quality-artifact", type=Path, default=DEFAULT_QUALITY_ARTIFACT)
    parser.add_argument("--rag-vs-artifact", type=Path, default=DEFAULT_RAG_VS_ARTIFACT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--dry-run", action="store_true", help="Build and print the upload bundle without initializing Weave.")
    args = parser.parse_args(argv)

    payloads = load_phase04_payloads(
        matrix_artifact=args.matrix_artifact,
        quality_artifact=args.quality_artifact,
        rag_vs_artifact=args.rag_vs_artifact,
    )
    bundle = build_upload_bundle(payloads)
    if args.dry_run:
        print(json.dumps(bundle["summary"], indent=2, sort_keys=True))
        return

    receipt = upload_phase04_results(bundle, project=args.project)
    receipt["artifact_paths"] = bundle["artifact_paths"]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
