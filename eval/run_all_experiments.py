#!/usr/bin/env python
"""
Run all GPU memory optimization experiments automatically.

Usage:
    python -m eval.run_all_experiments              # Run all experiments
    python -m eval.run_all_experiments --dry-run    # Preview commands
    python -m eval.run_all_experiments --summarize  # Only generate summary from existing results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from eval.experiment_runner import EXPERIMENTS, run_experiment
from eval.weave_ops import DEFAULT_WEAVE_PROJECT, init_weave


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
EXPERIMENT_ROOT = ARTIFACT_DIR / "experiment_results"
SUMMARY_ROOT = ARTIFACT_DIR / "experiment_summaries"


def run_all_experiments(
    *,
    repetitions: int = 5,
    gpu_counts: str = "1,6",
    dataset_sizes: list[int] | None = None,
    dry_run: bool = False,
    experiments: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all experiments and return results."""
    results: dict[str, dict[str, Any]] = {}
    exp_ids = experiments or list(EXPERIMENTS.keys())

    print(f"\n{'='*60}")
    print(f"Running {len(exp_ids)} experiments")
    print(f"Repetitions: {repetitions}")
    print(f"GPU counts: {gpu_counts}")
    print(f"Dataset sizes: {dataset_sizes or 'default'}")
    print(f"{'='*60}\n")

    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n[{i}/{len(exp_ids)}] Starting {exp_id}: {EXPERIMENTS[exp_id]['name']}")
        print("-" * 40)

        start_time = time.time()
        result = run_experiment(
            exp_id,
            repetitions=repetitions,
            gpu_counts=gpu_counts,
            dataset_sizes=dataset_sizes,
            dry_run=dry_run,
        )
        elapsed = time.time() - start_time

        result["elapsed_seconds"] = elapsed
        results[exp_id] = result

        status = result.get("status", "unknown")
        print(f"[{i}/{len(exp_ids)}] {exp_id}: {status} ({elapsed:.1f}s)")

        # Save intermediate progress
        _save_progress(results)

    return results


def _save_progress(results: dict[str, dict[str, Any]]) -> None:
    """Save progress to a checkpoint file."""
    progress_file = EXPERIMENT_ROOT / "run_progress.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    # Serialize only the essential info
    progress = {
        exp_id: {
            "status": res.get("status"),
            "elapsed_seconds": res.get("elapsed_seconds"),
            "returncode": res.get("returncode"),
        }
        for exp_id, res in results.items()
    }
    progress_file.write_text(json.dumps(progress, indent=2, sort_keys=True))


def generate_consolidated_summary(results: dict[str, dict[str, Any]] | None = None) -> str:
    """Generate a consolidated summary of all experiment results."""
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    lines = [
        "# GPU Memory Optimization Experiments - Consolidated Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## Overview",
        "",
        "| Experiment | Name | Status | Hypothesis |",
        "|---|---|---|---|",
    ]

    for exp_id, config in EXPERIMENTS.items():
        exp_dir = EXPERIMENT_ROOT / exp_id
        agg_file = exp_dir / "aggregated.json"

        if agg_file.exists():
            agg = json.loads(agg_file.read_text())
            verdict = agg.get("regression_summary", {}).get("verdict", "unknown")
            status = f"complete ({verdict})"
        elif (exp_dir / "config.json").exists():
            status = "config only"
        else:
            status = "not started"

        lines.append(f"| {exp_id} | {config['name']} | {status} | {config['hypothesis'][:60]}... |")

    lines.extend([
        "",
        "## Experiment Details",
        "",
    ])

    for exp_id, config in EXPERIMENTS.items():
        exp_dir = EXPERIMENT_ROOT / exp_id
        summary_file = SUMMARY_ROOT / f"{exp_id}_summary.md"
        agg_file = exp_dir / "aggregated.json"

        lines.extend([
            f"### {exp_id}: {config['name']}",
            "",
            f"**Hypothesis:** {config['hypothesis']}",
            "",
            f"**Success Criteria:** {config['success_criteria']}",
            "",
            f"**Configuration:**",
            f"- Env vars: `{config['env_vars'] or 'none'}`",
            f"- Modal args: `{config.get('modal_args', []) or 'none'}`",
            "",
        ])

        if agg_file.exists():
            agg = json.loads(agg_file.read_text())
            summary = agg.get("regression_summary", {})

            lines.extend([
                "**Results:**",
                f"- Comparison count: {summary.get('comparison_count', 'N/A')}",
                f"- Improvements: {summary.get('improvement_count', 'N/A')}",
                f"- Regressions: {summary.get('regression_count', 'N/A')}",
                f"- Verdict: **{summary.get('verdict', 'unknown').upper()}**",
                "",
            ])

            # Add top comparisons
            comparisons = agg.get("comparisons", [])
            throughput_rows = [r for r in comparisons if r.get("metric") == "requests_per_s"]
            if throughput_rows:
                lines.extend([
                    "**Throughput Results:**",
                    "",
                    "| Workload | GPU | Baseline | Candidate | Delta | Verdict |",
                    "|---|---|---:|---:|---:|---|",
                ])
                for row in throughput_rows[:5]:
                    lines.append(
                        f"| {row.get('scenario', 'N/A')} | {row.get('gpu_count', '?')} H100 | "
                        f"{row.get('baseline_mean', 0):.2f} | {row.get('candidate_mean', 0):.2f} | "
                        f"{row.get('improvement_pct', 0):.1f}% | {row.get('verdict', '?')} |"
                    )
                lines.append("")
        else:
            lines.extend([
                "**Results:** Not yet available",
                "",
            ])

        lines.extend([
            f"**Artifacts:**",
            f"- Config: `eval/artifacts/experiment_results/{exp_id}/config.json`",
            f"- Results: `eval/artifacts/experiment_results/{exp_id}/aggregated.json`",
            f"- Summary: `eval/artifacts/experiment_summaries/{exp_id}_summary.md`",
            "",
            "---",
            "",
        ])

    # Recommendations section
    lines.extend([
        "## Recommendations",
        "",
        "Based on experiment results:",
        "",
    ])

    passed = []
    failed = []
    pending = []

    for exp_id in EXPERIMENTS:
        agg_file = EXPERIMENT_ROOT / exp_id / "aggregated.json"
        if agg_file.exists():
            agg = json.loads(agg_file.read_text())
            verdict = agg.get("regression_summary", {}).get("verdict", "unknown")
            if verdict == "pass":
                passed.append(exp_id)
            else:
                failed.append(exp_id)
        else:
            pending.append(exp_id)

    if passed:
        lines.append(f"**Ready to apply:** {', '.join(passed)}")
    if failed:
        lines.append(f"**Regression detected:** {', '.join(failed)}")
    if pending:
        lines.append(f"**Pending execution:** {', '.join(pending)}")

    lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review individual experiment summaries in `eval/artifacts/experiment_summaries/`",
        "2. Apply successful optimizations incrementally",
        "3. Re-run quality gates after applying changes",
        "4. Update `docs/optimization-results-ledger.md` with final decisions",
        "",
    ])

    content = "\n".join(lines)

    # Save consolidated summary
    consolidated_path = SUMMARY_ROOT / "CONSOLIDATED_SUMMARY.md"
    consolidated_path.write_text(content)
    print(f"\nConsolidated summary saved to: {consolidated_path}")

    return content


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all GPU memory optimization experiments")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per config")
    parser.add_argument("--gpu-counts", default="1,6", help="GPU counts to test")
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--summarize", action="store_true", help="Only generate summary from existing results")
    parser.add_argument("--weave", action="store_true", help="Enable Weave tracing")
    parser.add_argument("--weave-project", default=DEFAULT_WEAVE_PROJECT)
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run (default: all)")
    args = parser.parse_args(argv)

    if args.weave:
        init_weave(args.weave_project)

    if args.summarize:
        summary = generate_consolidated_summary()
        print(summary)
        return

    results = run_all_experiments(
        repetitions=args.repetitions,
        gpu_counts=args.gpu_counts,
        dataset_sizes=args.dataset_sizes,
        dry_run=args.dry_run,
        experiments=args.experiments,
    )

    # Generate consolidated summary
    summary = generate_consolidated_summary(results)

    # Final status
    print("\n" + "=" * 60)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 60)

    success = sum(1 for r in results.values() if r.get("status") == "success")
    failed = sum(1 for r in results.values() if r.get("status") == "failed")
    dry_run_count = sum(1 for r in results.values() if r.get("status") == "dry_run")

    print(f"Success: {success}")
    print(f"Failed: {failed}")
    if dry_run_count:
        print(f"Dry run: {dry_run_count}")
    print(f"\nConsolidated summary: eval/artifacts/experiment_summaries/CONSOLIDATED_SUMMARY.md")


if __name__ == "__main__":
    main()
