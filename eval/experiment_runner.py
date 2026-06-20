from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval.weave_ops import DEFAULT_WEAVE_PROJECT, init_weave, weave_op


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
EXPERIMENT_ROOT = ARTIFACT_DIR / "experiment_results"


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "EXP-FP8-001": {
        "name": "fp8 KV cache",
        "env_vars": {"KV_CACHE_DTYPE": "fp8"},
        "hypothesis": "fp8 KV cache halves memory, enables larger batches without OOM",
        "success_criteria": "F1 >= 0.7, throughput neutral or improved",
        "modal_args": [],
    },
    "EXP-BATCH-001": {
        "name": "increased batch sizes",
        "env_vars": {"QUERY_BATCH_SIZE": "128", "REFINE_BATCH_SIZE": "128"},
        "hypothesis": "Larger batches (128 vs 64) improve GPU utilization",
        "success_criteria": ">=5% throughput improvement, p95 regression <10%",
        "modal_args": [],
    },
    "EXP-MBT-001": {
        "name": "max batched tokens 12288",
        "env_vars": {},
        "hypothesis": "Larger max_num_batched_tokens improves prefill throughput",
        "success_criteria": ">=5% throughput improvement without OOM",
        "modal_args": ["--max-num-batched-tokens", "12288"],
    },
    "EXP-MBT-002": {
        "name": "max batched tokens 16384",
        "env_vars": {},
        "hypothesis": "Larger max_num_batched_tokens improves prefill throughput",
        "success_criteria": ">=5% throughput improvement without OOM",
        "modal_args": ["--max-num-batched-tokens", "16384"],
    },
    "EXP-SCHED-001": {
        "name": "time-window scheduling 15ms",
        "env_vars": {"BATCH_ACCUMULATE_MS": "15"},
        "hypothesis": "Accumulating requests for 15ms improves batch efficiency",
        "success_criteria": ">=10% throughput improvement to justify latency",
        "modal_args": [],
    },
    "EXP-LENBIN-001": {
        "name": "input-length binning",
        "env_vars": {"VLLM_ROUTING_MODE": "length_bin"},
        "hypothesis": "Routing similar-length inputs together reduces padding waste",
        "success_criteria": ">=5% throughput improvement, p95 latency improved",
        "modal_args": [],
    },
    "EXP-OVERLAP-001": {
        "name": "chunk overlap 10%",
        "env_vars": {"CHUNK_OVERLAP_RATIO": "0.1"},
        "hypothesis": "10% overlap improves recall at chunk boundaries",
        "success_criteria": "Recall improvement with acceptable throughput cost",
        "modal_args": [],
    },
    "EXP-OVERLAP-002": {
        "name": "chunk overlap 20%",
        "env_vars": {"CHUNK_OVERLAP_RATIO": "0.2"},
        "hypothesis": "20% overlap improves recall at chunk boundaries",
        "success_criteria": "Recall improvement with acceptable throughput cost",
        "modal_args": [],
    },
}


def _build_env(baseline: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Merge baseline environment with experiment overrides."""
    result = dict(baseline)
    result.update(overrides)
    return result


def _ensure_experiment_dir(exp_id: str) -> Path:
    """Create experiment artifact directory."""
    exp_dir = EXPERIMENT_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "runs").mkdir(exist_ok=True)
    return exp_dir


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_config(exp_dir: Path, exp_id: str, config: dict[str, Any]) -> None:
    """Write experiment configuration to JSON."""
    payload = {
        "exp_id": exp_id,
        "name": config["name"],
        "hypothesis": config["hypothesis"],
        "success_criteria": config["success_criteria"],
        "env_vars": config["env_vars"],
        "modal_args": config.get("modal_args", []),
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (exp_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


@weave_op(name="eval.experiment.run")
def run_experiment(
    exp_id: str,
    *,
    repetitions: int = 5,
    gpu_counts: str = "1,6",
    dataset_sizes: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single experiment with specified configuration."""
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_id}. Available: {list(EXPERIMENTS.keys())}")

    config = EXPERIMENTS[exp_id]
    exp_dir = _ensure_experiment_dir(exp_id)
    _write_config(exp_dir, exp_id, config)

    dataset_sizes = dataset_sizes or [7, 100, 1000, 10000, 25000, 100000]

    env = _build_env(dict(os.environ), config["env_vars"])

    cmd = [
        sys.executable, "-m", "eval.standard_benchmark",
        "--opt-id", exp_id,
        "--name", config["name"],
        "--run-modal",
        "--gpu-counts", gpu_counts,
        "--dataset-sizes", *[str(s) for s in dataset_sizes],
        "--rag-runs", str(repetitions),
        *config.get("modal_args", []),
    ]

    result = {
        "exp_id": exp_id,
        "config": config,
        "command": " ".join(cmd),
        "env_overrides": config["env_vars"],
        "dry_run": dry_run,
    }

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        print(f"[DRY RUN] With env overrides: {config['env_vars']}")
        result["status"] = "dry_run"
        return result

    print(f"Running experiment {exp_id}: {config['name']}")
    print(f"Command: {' '.join(cmd)}")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    result["returncode"] = proc.returncode
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    result["status"] = "success" if proc.returncode == 0 else "failed"

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GPU memory optimization experiments")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID to run (e.g., EXP-FP8-001)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per config")
    parser.add_argument("--gpu-counts", default="1,6", help="GPU counts to test")
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--weave", action="store_true", help="Enable Weave tracing")
    parser.add_argument("--weave-project", default=DEFAULT_WEAVE_PROJECT)
    args = parser.parse_args(argv)

    if args.list:
        print("Available experiments:")
        for exp_id, config in EXPERIMENTS.items():
            print(f"  {exp_id}: {config['name']}")
            print(f"    Hypothesis: {config['hypothesis']}")
            print(f"    Success: {config['success_criteria']}")
            print()
        return

    if args.weave:
        init_weave(args.weave_project)

    if args.all:
        for exp_id in EXPERIMENTS:
            run_experiment(
                exp_id,
                repetitions=args.repetitions,
                gpu_counts=args.gpu_counts,
                dataset_sizes=args.dataset_sizes,
                dry_run=args.dry_run,
            )
    elif args.exp_id:
        run_experiment(
            args.exp_id,
            repetitions=args.repetitions,
            gpu_counts=args.gpu_counts,
            dataset_sizes=args.dataset_sizes,
            dry_run=args.dry_run,
        )
    else:
        parser.error("Specify an experiment ID or --list or --all")


if __name__ == "__main__":
    main()
