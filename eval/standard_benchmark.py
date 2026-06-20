from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval.bench import _git_commit
from eval.rag_compare import DEFAULT_QUERY, _measure_rag_size
from eval.upload_weave_results import _h100_rows, _portable_path


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_BASELINE_MATRIX = ARTIFACT_DIR / "phase04_h100_rag_matrix.json"
EXPERIMENT_ROOT = ARTIFACT_DIR / "experiment_results"
SUMMARY_ROOT = ARTIFACT_DIR / "experiment_summaries"
DEFAULT_DATASET_SIZES = [7, 100, 1_000, 10_000, 25_000, 100_000]
EXTREME_DATASET_SIZES = [250_000]
PRIMARY_METRICS = [
    "requests_per_s",
    "latency_ms_p50",
    "latency_ms_p95",
    "gpu_utilization_pct_mean",
    "derived_mfu_bf16_peak",
]
METRIC_DIRECTION = {
    "requests_per_s": "higher",
    "latency_ms_p50": "lower",
    "latency_ms_p95": "lower",
    "gpu_utilization_pct_mean": "higher",
    "derived_mfu_bf16_peak": "higher",
}


def run_standard_benchmark(
    *,
    opt_id: str,
    name: str,
    baseline_artifacts: list[Path],
    candidate_artifacts: list[Path],
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    dataset_sizes: list[int] | None = None,
    rag_runs: int = 3,
    top_k: int = 5,
    query: str = DEFAULT_QUERY,
    skip_rag: bool = False,
    warmup_excluded: bool = False,
    agent: str = "agent",
    notes: str = "",
    command: str | None = None,
    regression_tolerance_pct: float = 5.0,
    min_improvement_pct: float = 5.0,
) -> dict[str, Any]:
    dataset_sizes = dataset_sizes or DEFAULT_DATASET_SIZES
    exp_dir = EXPERIMENT_ROOT / opt_id
    runs_dir = exp_dir / "runs"
    plots_dir = exp_dir / "plots"
    summary_dir = SUMMARY_ROOT
    for path in (runs_dir, plots_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = {
        "opt_id": opt_id,
        "name": name,
        "agent": agent,
        "commit": _git_commit(),
        "generated_at_unix": int(time.time()),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_artifacts": [_portable_path(path) for path in baseline_artifacts],
        "candidate_artifacts": [_portable_path(path) for path in candidate_artifacts],
        "dataset_sizes": dataset_sizes,
        "rag_runs": rag_runs,
        "top_k": top_k,
        "query": query,
        "skip_rag": skip_rag,
        "warmup_excluded": warmup_excluded,
        "regression_tolerance_pct": regression_tolerance_pct,
        "min_improvement_pct": min_improvement_pct,
        "command": command or " ".join(sys.argv),
        "notes": notes,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    baseline_runs = _load_matrix_runs(baseline_artifacts, baseline_label, runs_dir)
    candidate_runs = _load_matrix_runs(candidate_artifacts, candidate_label, runs_dir)
    baseline_agg = _aggregate_matrix_runs(baseline_runs)
    candidate_agg = _aggregate_matrix_runs(candidate_runs)
    comparisons = _compare_aggregates(
        baseline_agg,
        candidate_agg,
        regression_tolerance_pct=regression_tolerance_pct,
        min_improvement_pct=min_improvement_pct,
    )
    rag_rows = [] if skip_rag else _run_rag_ladder(dataset_sizes, query=query, top_k=top_k, runs=rag_runs)
    scaling = _scaling_analysis(rag_rows)

    aggregated = {
        "opt_id": opt_id,
        "name": name,
        "generated_at_unix": int(time.time()),
        "baseline": baseline_agg,
        "candidate": candidate_agg,
        "comparisons": comparisons,
        "rag_rows": rag_rows,
        "regression_summary": _regression_summary(comparisons),
    }
    (exp_dir / "aggregated.json").write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n")
    (exp_dir / "scaling_analysis.json").write_text(json.dumps(scaling, indent=2, sort_keys=True) + "\n")

    ledger_entry = _ledger_entry_markdown(config, aggregated, scaling)
    summary = _summary_markdown(config, aggregated, scaling)
    (exp_dir / "ledger_entry.md").write_text(ledger_entry + "\n")
    (summary_dir / f"{opt_id}_summary.md").write_text(summary + "\n")
    return {
        "config": config,
        "aggregated": aggregated,
        "scaling": scaling,
        "paths": {
            "config": _portable_path(exp_dir / "config.json"),
            "aggregated": _portable_path(exp_dir / "aggregated.json"),
            "scaling_analysis": _portable_path(exp_dir / "scaling_analysis.json"),
            "ledger_entry": _portable_path(exp_dir / "ledger_entry.md"),
            "summary": _portable_path(summary_dir / f"{opt_id}_summary.md"),
        },
    }


def run_modal_matrix(args: argparse.Namespace, artifact_prefix: str) -> Path:
    cmd = _modal_command_prefix() + [
        "run",
        "inference/modal_app.py::benchmark_h100_rag_matrix",
        "--gpu-counts",
        args.gpu_counts,
        "--rag-sizes",
        ",".join(str(size) for size in args.dataset_sizes),
        "--single-requests",
        str(args.single_requests),
        "--multi-requests",
        str(args.multi_requests),
        "--single-concurrency",
        str(args.single_concurrency),
        "--multi-concurrency",
        str(args.multi_concurrency),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--prompt-variant",
        args.prompt_variant,
        "--rag-runs",
        str(args.rag_runs),
        "--artifact-prefix",
        artifact_prefix,
    ]
    subprocess.run(cmd, check=True)
    return ARTIFACT_DIR / f"{artifact_prefix}.json"


def _modal_command_prefix() -> list[str]:
    venv_modal = Path(sys.executable).with_name("modal")
    if venv_modal.exists():
        return [str(venv_modal)]
    return ["modal"]


def _load_matrix_runs(artifacts: list[Path], label: str, runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for idx, artifact in enumerate(artifacts, start=1):
        matrix = json.loads(artifact.read_text())
        run = {
            "run_id": matrix.get("run_id", f"{label}-{idx:03d}"),
            "label": label,
            "artifact": _portable_path(artifact),
            "model": matrix.get("model"),
            "vllm_version": matrix.get("vllm_version"),
            "prompt_variant": matrix.get("prompt_variant"),
            "gpu_memory_utilization": matrix.get("gpu_memory_utilization"),
            "rows": _h100_rows(matrix),
        }
        runs.append(run)
        (runs_dir / f"{label}_matrix_run_{idx:03d}.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    return runs


def _aggregate_matrix_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for run in runs:
        for row in run["rows"]:
            key = _row_key(row)
            entry = grouped.setdefault(
                key,
                {
                    "scenario": row["scenario"],
                    "dataset_mode": row["dataset_mode"],
                    "gpu_count": row["gpu_count"],
                    "num_requests": row["num_requests"],
                    "concurrency": row["concurrency"],
                    "metrics": {metric: [] for metric in PRIMARY_METRICS},
                },
            )
            for metric in PRIMARY_METRICS:
                value = row.get(metric)
                if value is not None:
                    entry["metrics"][metric].append(float(value))

    by_workload = {}
    for key, entry in grouped.items():
        by_workload[key] = {
            **{field: entry[field] for field in ("scenario", "dataset_mode", "gpu_count", "num_requests", "concurrency")},
            "metrics": {metric: _stats(values) for metric, values in entry["metrics"].items()},
        }
    return {"run_count": len(runs), "by_workload": by_workload}


def _compare_aggregates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    regression_tolerance_pct: float,
    min_improvement_pct: float,
) -> list[dict[str, Any]]:
    rows = []
    for key, base_row in sorted(baseline["by_workload"].items()):
        candidate_row = candidate["by_workload"].get(key)
        if candidate_row is None:
            continue
        for metric in PRIMARY_METRICS:
            base_stats = base_row["metrics"][metric]
            candidate_stats = candidate_row["metrics"][metric]
            if base_stats["n"] == 0 or candidate_stats["n"] == 0:
                continue
            base_mean = base_stats["mean"]
            candidate_mean = candidate_stats["mean"]
            rel_delta_pct = (candidate_mean - base_mean) / max(abs(base_mean), 1e-9) * 100.0
            direction = METRIC_DIRECTION[metric]
            improvement_pct = rel_delta_pct if direction == "higher" else -rel_delta_pct
            p_value = _welch_normal_p_value(base_stats["values"], candidate_stats["values"])
            rows.append(
                {
                    "workload_key": key,
                    "scenario": base_row["scenario"],
                    "dataset_mode": base_row["dataset_mode"],
                    "gpu_count": base_row["gpu_count"],
                    "metric": metric,
                    "direction": direction,
                    "baseline_mean": base_mean,
                    "baseline_std": base_stats["std"],
                    "candidate_mean": candidate_mean,
                    "candidate_std": candidate_stats["std"],
                    "abs_delta": candidate_mean - base_mean,
                    "rel_delta_pct": rel_delta_pct,
                    "improvement_pct": improvement_pct,
                    "p_value_normal_approx": p_value,
                    "significant_approx": p_value is not None and p_value < 0.05,
                    "verdict": _metric_verdict(
                        improvement_pct,
                        regression_tolerance_pct=regression_tolerance_pct,
                        min_improvement_pct=min_improvement_pct,
                    ),
                }
            )
    return rows


def _run_rag_ladder(dataset_sizes: list[int], *, query: str, top_k: int, runs: int) -> list[dict[str, Any]]:
    rows = []
    for size in dataset_sizes:
        row = _measure_rag_size(n_docs=size, query=query, top_k=top_k, runs=runs)
        rows.append(row)
    return rows


def _scaling_analysis(rag_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rag_rows, key=lambda row: row["n_docs"])
    transitions = []
    for before, after in zip(ordered, ordered[1:]):
        doc_factor = after["n_docs"] / max(before["n_docs"], 1)
        retrieve_factor = after["retrieve_ms_p50"] / max(before["retrieve_ms_p50"], 1e-9)
        fresh_factor = after["fresh_file_total_ms"] / max(before["fresh_file_total_ms"], 1e-9)
        transitions.append(
            {
                "from_docs": before["n_docs"],
                "to_docs": after["n_docs"],
                "doc_factor": doc_factor,
                "retrieve_latency_factor": retrieve_factor,
                "fresh_file_latency_factor": fresh_factor,
                "retrieve_scaling_exponent": _scaling_exponent(retrieve_factor, doc_factor),
                "fresh_file_scaling_exponent": _scaling_exponent(fresh_factor, doc_factor),
            }
        )
    return {"rag_rows": ordered, "transitions": transitions}


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "values": [], "mean": None, "std": None, "min": None, "max": None, "ci_95": [None, None]}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci_95": [mean - half_width, mean + half_width],
    }


def _welch_normal_p_value(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    denom = math.sqrt(var_a / len(a) + var_b / len(b))
    if denom == 0:
        return 1.0 if mean_a == mean_b else 0.0
    z_score = (mean_b - mean_a) / denom
    return math.erfc(abs(z_score) / math.sqrt(2.0))


def _metric_verdict(improvement_pct: float, *, regression_tolerance_pct: float, min_improvement_pct: float) -> str:
    if improvement_pct <= -regression_tolerance_pct:
        return "regression"
    if improvement_pct >= min_improvement_pct:
        return "improved"
    return "neutral"


def _regression_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    regressions = [row for row in comparisons if row["verdict"] == "regression"]
    improvements = [row for row in comparisons if row["verdict"] == "improved"]
    return {
        "comparison_count": len(comparisons),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "verdict": "regression" if regressions else "pass",
    }


def _row_key(row: dict[str, Any]) -> str:
    return f"{row['scenario']}|gpu={row['gpu_count']}|concurrency={row['concurrency']}|requests={row['num_requests']}"


def _scaling_exponent(metric_factor: float, doc_factor: float) -> float | None:
    if metric_factor <= 0 or doc_factor <= 1:
        return None
    return math.log(metric_factor) / math.log(doc_factor)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ledger_entry_markdown(config: dict[str, Any], aggregated: dict[str, Any], scaling: dict[str, Any]) -> str:
    comparisons = aggregated["comparisons"]
    summary = aggregated["regression_summary"]
    throughput_rows = [row for row in comparisons if row["metric"] == "requests_per_s"]
    utilization_rows = [row for row in comparisons if row["metric"] in {"gpu_utilization_pct_mean", "derived_mfu_bf16_peak"}]
    lines = [
        f"### {config['opt_id']}: {config['name']}",
        "",
        "- status: proposed | running | applied | rejected | reverted",
        f"- owner: {config['agent']}",
        f"- date: {time.strftime('%Y-%m-%d', time.gmtime(config['generated_at_unix']))}",
        f"- commit: `{config['commit']}`",
        "- artifacts:",
        f"  - `eval/artifacts/experiment_results/{config['opt_id']}/config.json`",
        f"  - `eval/artifacts/experiment_results/{config['opt_id']}/aggregated.json`",
        f"  - `eval/artifacts/experiment_results/{config['opt_id']}/scaling_analysis.json`",
        f"  - `eval/artifacts/experiment_summaries/{config['opt_id']}_summary.md`",
        "- Weave run/eval: upload with `python -m eval.upload_weave_results` after freezing a matrix artifact",
        "- hypothesis: fill in before accepting this optimization",
        "- change: fill in exact code/config change",
        "- expected mechanism: fill in why the metric should move",
        "",
        "#### Experiment Configuration",
        f"- repetitions: baseline artifacts n={len(config['baseline_artifacts'])}; candidate artifacts n={len(config['candidate_artifacts'])}; RAG runs={config['rag_runs']}",
        f"- warmup excluded: {'yes' if config['warmup_excluded'] else 'no'}",
        "- dataset sizes tested:",
        "  | size tier | doc count | notes |",
        "  |---|---:|---|",
    ]
    for size in config["dataset_sizes"]:
        lines.append(f"  | {_size_tier(size)} | {size} | standardized RAG scaling ladder |")
    lines.extend(
        [
            "",
            "#### Quality Gate",
            "- precision: carry forward from `phase04_quality_gate.json` unless rerun",
            "- recall: carry forward from `phase04_quality_gate.json` unless rerun",
            "- F1: must remain >= 0.7",
            "- threshold: carry forward or document new threshold",
            "- verdict: pass | fail | not rerun",
            "",
            "#### Performance Delta (with variance)",
            "| workload | dataset | baseline mean +/- std | candidate mean +/- std | delta | p-value | verdict |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in throughput_rows:
        lines.append(
            "| "
            f"{row['scenario']} ({row['gpu_count']} H100) | matrix | "
            f"{_fmt(row['baseline_mean'])} +/- {_fmt(row['baseline_std'])} | "
            f"{_fmt(row['candidate_mean'])} +/- {_fmt(row['candidate_std'])} | "
            f"{_fmt(row['improvement_pct'])}% | {_fmt(row['p_value_normal_approx'])} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "#### Utilization (with variance)",
            "| workload | dataset | metric | baseline mean +/- std | candidate mean +/- std | delta | verdict |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in utilization_rows:
        lines.append(
            "| "
            f"{row['scenario']} ({row['gpu_count']} H100) | matrix | {row['metric']} | "
            f"{_fmt(row['baseline_mean'])} +/- {_fmt(row['baseline_std'])} | "
            f"{_fmt(row['candidate_mean'])} +/- {_fmt(row['candidate_std'])} | "
            f"{_fmt(row['improvement_pct'])}% | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "#### Scaling Analysis",
            "| transition | retrieve latency factor | fresh-file latency factor | retrieve exponent | fresh-file exponent |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in scaling["transitions"]:
        lines.append(
            "| "
            f"{row['from_docs']} -> {row['to_docs']} docs | "
            f"{_fmt(row['retrieve_latency_factor'])}x | {_fmt(row['fresh_file_latency_factor'])}x | "
            f"{_fmt(row['retrieve_scaling_exponent'])} | {_fmt(row['fresh_file_scaling_exponent'])} |"
        )
    lines.extend(
        [
            "",
            "#### Summary",
            f"- regression threshold: {config['regression_tolerance_pct']}%",
            f"- decision: {summary['verdict']}",
            "- caveats: fill in infra anomalies, OOMs, JIT/warmup, or missing quality reruns",
            "- next action: fill in",
            "- rollback: fill in",
        ]
    )
    return "\n".join(lines)


def _summary_markdown(config: dict[str, Any], aggregated: dict[str, Any], scaling: dict[str, Any]) -> str:
    summary = aggregated["regression_summary"]
    lines = [
        f"# Experiment Summary: {config['opt_id']} - {config['name']}",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(config['generated_at_unix']))}",
        f"Agent: {config['agent']}",
        f"Commit: {config['commit']}",
        "",
        "## Executive Summary",
        "",
        (
            f"Compared `{config['candidate_label']}` against `{config['baseline_label']}` "
            f"across {len(aggregated['comparisons'])} matrix metric comparisons and "
            f"{len(scaling['rag_rows'])} RAG dataset sizes."
        ),
        "",
        f"**Verdict:** {'REJECTED' if summary['verdict'] == 'regression' else 'INCONCLUSIVE'}",
        "**Confidence:** LOW until repeated candidate/baseline runs and quality reruns are present.",
        "",
        "## Dataset Configuration",
        "",
        "| size tier | doc count | corpus description |",
        "|---|---:|---|",
    ]
    for size in config["dataset_sizes"]:
        lines.append(f"| {_size_tier(size)} | {size} | scaled demo corpus for RAG timing |")
    lines.extend(
        [
            "",
            "## Aggregated Results",
            "",
            "| workload | metric | baseline mean | candidate mean | improvement % | verdict |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in aggregated["comparisons"]:
        lines.append(
            f"| {row['scenario']} ({row['gpu_count']} H100) | {row['metric']} | "
            f"{_fmt(row['baseline_mean'])} | {_fmt(row['candidate_mean'])} | "
            f"{_fmt(row['improvement_pct'])} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Scaling Factors",
            "",
            "| transition | retrieve latency factor | fresh-file latency factor |",
            "|---|---:|---:|",
        ]
    )
    for row in scaling["transitions"]:
        lines.append(
            f"| {row['from_docs']} -> {row['to_docs']} docs | "
            f"{_fmt(row['retrieve_latency_factor'])}x | {_fmt(row['fresh_file_latency_factor'])}x |"
        )
    lines.extend(
        [
            "",
            "## Command to Reproduce",
            "",
            "```bash",
            config["command"],
            "```",
            "",
            "## Artifacts",
            "",
            "| artifact | path |",
            "|---|---|",
            f"| config | `eval/artifacts/experiment_results/{config['opt_id']}/config.json` |",
            f"| aggregated results | `eval/artifacts/experiment_results/{config['opt_id']}/aggregated.json` |",
            f"| scaling analysis | `eval/artifacts/experiment_results/{config['opt_id']}/scaling_analysis.json` |",
            f"| ledger entry | `eval/artifacts/experiment_results/{config['opt_id']}/ledger_entry.md` |",
        ]
    )
    return "\n".join(lines)


def _size_tier(size: int) -> str:
    if size <= 100:
        return "small"
    if size <= 1_000:
        return "medium"
    if size <= 10_000:
        return "large"
    if size <= 25_000:
        return "xlarge"
    if size <= 100_000:
        return "xxlarge"
    return "huge"


def _parse_sizes(raw: list[int], include_extreme: bool) -> list[int]:
    sizes = list(dict.fromkeys(raw + (EXTREME_DATASET_SIZES if include_extreme else [])))
    return sorted(sizes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run standardized optimization benchmarks and write ledger-ready artifacts.")
    parser.add_argument("--opt-id", required=True, help="Optimization id, e.g. OPT-003")
    parser.add_argument("--name", required=True, help="Short optimization name")
    parser.add_argument("--agent", default="agent")
    parser.add_argument("--baseline-artifact", type=Path, action="append", default=None)
    parser.add_argument("--candidate-artifact", type=Path, action="append", default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=DEFAULT_DATASET_SIZES)
    parser.add_argument("--include-extreme", action="store_true", help="Also test the optional 250K-doc RAG tier.")
    parser.add_argument("--rag-runs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--warmup-excluded", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--regression-tolerance-pct", type=float, default=5.0)
    parser.add_argument("--min-improvement-pct", type=float, default=5.0)
    parser.add_argument("--run-modal", action="store_true", help="Run the Modal H100 matrix before comparing.")
    parser.add_argument("--gpu-counts", default="1,6")
    parser.add_argument("--single-requests", type=int, default=32)
    parser.add_argument("--multi-requests", type=int, default=96)
    parser.add_argument("--single-concurrency", type=int, default=1)
    parser.add_argument("--multi-concurrency", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--prompt-variant", default="compact")
    args = parser.parse_args(argv)

    args.dataset_sizes = _parse_sizes(args.dataset_sizes, args.include_extreme)
    baseline_artifacts = args.baseline_artifact or [DEFAULT_BASELINE_MATRIX]
    candidate_artifacts = args.candidate_artifact or [DEFAULT_BASELINE_MATRIX]
    if args.run_modal:
        artifact_prefix = f"experiment_results/{args.opt_id}/candidate_h100_rag_matrix"
        candidate_artifacts = [run_modal_matrix(args, artifact_prefix)]

    result = run_standard_benchmark(
        opt_id=args.opt_id,
        name=args.name,
        baseline_artifacts=baseline_artifacts,
        candidate_artifacts=candidate_artifacts,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        dataset_sizes=args.dataset_sizes,
        rag_runs=args.rag_runs,
        top_k=args.top_k,
        query=args.query,
        skip_rag=args.skip_rag,
        warmup_excluded=args.warmup_excluded,
        agent=args.agent,
        notes=args.notes,
        command=" ".join(sys.argv),
        regression_tolerance_pct=args.regression_tolerance_pct,
        min_improvement_pct=args.min_improvement_pct,
    )
    print(json.dumps(result["paths"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
