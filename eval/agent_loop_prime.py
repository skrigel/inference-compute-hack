from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from eval.agent_loop import run_agent_loop_experiment


DEFAULT_CONFIG_DIR = Path("eval/configs/extension3_prime")
DEFAULT_ARTIFACT_DIR = Path("eval/artifacts/prime_readiness")
SCHEMA_VERSION = "extension3.prime_readiness.v1"


def build_prime_readiness_packet() -> dict[str, Any]:
    """Build the Prime training handoff packet without launching training."""
    return {
        "schema_version": SCHEMA_VERSION,
        "goal": (
            "Prepare Extension 3 for cohort-level RL/post-training by freezing the "
            "environment contract, cheap pre-training metrics, Prime config shape, "
            "and no-credit setup gates."
        ),
        "credit_policy": {
            "no_credit_smoke_required": True,
            "explicit_user_approval_required": True,
            "paid_training_command_prefix": "prime train run",
            "never_run_in_smoke": ["prime train run"],
            "approval_phrase": "User confirms Prime credits can be used for the named training run.",
        },
        "cohort_manifest": build_cohort_manifest(),
        "reward_contract": build_reward_contract(),
        "prime_training_config": build_prime_training_config(),
        "metric_to_lift_schema": build_metric_to_lift_schema(),
        "no_credit_smoke": build_no_credit_smoke_plan(),
    }


def build_cohort_manifest() -> dict[str, Any]:
    baseline = {
        "cohort_id": "c00_baseline_retry",
        "split": "train",
        "description": "Baseline retry/backoff query-refinement cohort.",
        "target_topic": "retry_backoff",
        "n_docs": 1_000,
        "positive_ratio": 0.08,
        "distractor_hardness": "medium",
        "max_steps": 5,
        "beam_width": 5,
        "reward_noise": 0.0,
        "dynamic_corpus": True,
        "single_changed_variable": "baseline",
        "baseline_cohort_id": None,
    }
    variants = [
        _cohort("c01_topic_ir", "train", baseline, target_topic="ir_retrieval", single_changed_variable="target_topic"),
        _cohort(
            "c02_topic_cache",
            "train",
            baseline,
            target_topic="cache_threshold",
            single_changed_variable="target_topic",
        ),
        _cohort(
            "c03_sparse_positive",
            "train",
            baseline,
            positive_ratio=0.04,
            single_changed_variable="positive_ratio",
        ),
        _cohort(
            "c04_dense_positive",
            "train",
            baseline,
            positive_ratio=0.16,
            single_changed_variable="positive_ratio",
        ),
        _cohort(
            "c05_hard_distractors",
            "train",
            baseline,
            distractor_hardness="high",
            single_changed_variable="distractor_hardness",
        ),
        _cohort(
            "c06_large_dynamic",
            "train",
            baseline,
            n_docs=10_000,
            single_changed_variable="n_docs",
        ),
        _cohort(
            "c07_long_rollout",
            "train",
            baseline,
            max_steps=8,
            single_changed_variable="max_steps",
        ),
        _cohort(
            "c08_reward_noise",
            "train",
            baseline,
            reward_noise=0.05,
            single_changed_variable="reward_noise",
        ),
        _cohort(
            "h00_heldout_retry",
            "heldout",
            baseline,
            n_docs=2_000,
            single_changed_variable="split",
        ),
        _cohort(
            "h01_heldout_ir",
            "heldout",
            baseline,
            target_topic="ir_retrieval",
            n_docs=2_000,
            single_changed_variable="split",
        ),
        _cohort(
            "h02_heldout_cache",
            "heldout",
            baseline,
            target_topic="cache_threshold",
            n_docs=2_000,
            single_changed_variable="split",
        ),
    ]
    return {
        "schema_version": "extension3.cohort_manifest.v1",
        "control_principle": "Hold cohort size and training config fixed unless single_changed_variable names the change.",
        "primary_predictors": [
            "mean_reward_variance",
            "trajectory_entropy",
            "mean_steps_to_threshold",
            "mean_memory_selectivity",
            "mean_movement_selectivity",
            "agent_vs_human_speedup_estimate",
        ],
        "cohorts": [baseline, *variants],
    }


def build_reward_contract() -> dict[str, Any]:
    return {
        "schema_version": "extension3.reward_contract.v1",
        "environment_id": "extension3-agent-loop",
        "track_mapping": {
            "T": "A query-refinement task over a labeled dynamic corpus.",
            "M": "The policy proposes the next refined query or stop action.",
            "V": "A verifier compares selected evidence ids to positive chunk ids.",
            "y": "The refined query, selected evidence set, and optional stop decision.",
            "r": "Scalar reward from quality, byte movement, and memory selectivity.",
        },
        "verifiers_shape": {
            "dataset": "Rows contain cohort_id, initial_query, target_topic, chunk records, and positive_chunk_ids.",
            "harness": "Multi-turn loop: policy proposes query, scorer selects chunks, verifier returns state and reward.",
            "rubric": "Reward functions score precision, recall, F1, byte movement, selected bytes, and stop quality.",
        },
        "reward_formula": {
            "score": "f1 + 0.25 * recall - 0.05 * movement_ratio - 0.10 * selected_ratio",
            "movement_ratio": "bytes_moved_this_rollout / total_corpus_bytes",
            "selected_ratio": "bytes_selected_final / total_corpus_bytes",
            "pass_gate": "recall >= 0.8 and precision >= 0.5",
        },
        "failure_modes_to_log": [
            "query_overbroad_high_recall_low_precision",
            "query_underspecified_low_recall",
            "early_stop_before_pass_gate",
            "byte_budget_exceeded",
            "reward_hacking_by_selecting_all_chunks",
        ],
    }


def build_prime_training_config() -> dict[str, Any]:
    return {
        "schema_version": "extension3.prime_train_config.v1",
        "run": {
            "project": "inference-compute-hack",
            "name": "extension3-agent-loop-grpo-pilot",
            "purpose": "Measure whether cheap pre-training cohort metrics predict post-RL lift.",
        },
        "model": {
            "candidate_base": "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter": "lora",
            "selection_reason": "Small enough for a pilot, still capable of short query rewriting and stop decisions.",
        },
        "hardware": {
            "target": "8x H100 80GB on Prime",
            "gpu_type": "H100",
            "gpu_count": 8,
            "note": "Keep this as the run request/cluster target; Hosted Training configs may map hardware through the Prime UI or run provisioning layer.",
        },
        "algorithm": {
            "name": "GRPO",
            "rollouts_per_example": 8,
            "max_turns": 5,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "checkpointing": {
            "checkpoint_interval": 25,
            "keep_cloud_checkpoints": True,
            "adapter_interval": 25,
            "adapter_keep_last": 4,
            "validation_interval": 25,
            "checkpoint_list_command": "prime train checkpoints <run-id>",
            "resume_field": "checkpoint_id",
            "resume_note": "Set top-level checkpoint_id to a listed checkpoint id to warm-start a resumed run.",
        },
        "environment": {
            "id": "extension3-agent-loop",
            "cohort_manifest": "eval/configs/extension3_prime/cohort_manifest.example.json",
            "train_split": "train",
            "heldout_split": "heldout",
        },
        "launch": {
            "command": "prime train run eval/configs/extension3_prime/prime_train.example.toml",
            "requires_prime_credits": True,
            "allowed_only_after": [
                "local no-credit readiness report has passed",
                "Prime CLI is authenticated",
                "8-H100 Prime capacity is available or reserved",
                "baseline eval artifact exists",
                "user explicitly approves using Prime credits",
            ],
        },
    }


def build_metric_to_lift_schema() -> dict[str, Any]:
    return {
        "schema_version": "extension3.metric_to_lift.v1",
        "unit": "cohort",
        "required_rows": [
            {
                "cohort_id": "c00_baseline_retry",
                "split": "train",
                "cheap_metrics_before_training": {
                    "mean_best_reward": 0.0,
                    "mean_best_f1": 0.0,
                    "mean_reward_variance": 0.0,
                    "trajectory_entropy": 0.0,
                    "mean_steps_to_threshold": 0.0,
                    "mean_memory_selectivity": 0.0,
                    "mean_movement_selectivity": 0.0,
                    "cost_proxy_model_calls": 0,
                },
                "training": {
                    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "algorithm": "GRPO",
                    "adapter_artifact": "",
                    "train_wall_clock_minutes": 0.0,
                    "estimated_cost_usd": 0.0,
                },
                "eval": {
                    "baseline_reward": 0.0,
                    "trained_reward": 0.0,
                    "lift": 0.0,
                    "heldout_pass_rate": 0.0,
                },
            }
        ],
        "analysis_outputs": {
            "predictor_columns": [
                "mean_reward_variance",
                "trajectory_entropy",
                "mean_steps_to_threshold",
                "mean_memory_selectivity",
                "mean_movement_selectivity",
            ],
            "target_column": "lift",
            "goodness_of_fit": ["r2", "rmse", "spearman_rank"],
            "pareto_axes": ["metric_cost_proxy", "predictive_power"],
        },
    }


def build_no_credit_smoke_plan() -> dict[str, Any]:
    return {
        "schema_version": "extension3.no_credit_smoke.v1",
        "purpose": "Validate repo setup, configs, and environment generation before spending Prime training credits.",
        "commands": [
            "python -m eval.agent_loop_prime --output-dir eval/artifacts/prime_readiness --smoke-docs 60 --task-count 3",
            "python -m eval.agent_loop --n-docs 60 --task-count 3 --max-steps 5",
            "prime --help",
            "prime train checkpoints <existing-run-id>",
        ],
        "pass_criteria": {
            "paid_training_launched": False,
            "mean_truth_gain_min": 0.1,
            "pass_rate_min": 0.8,
            "required_artifacts": [
                "cohort_manifest.example.json",
                "prime_train.example.toml",
                "metric_to_lift_schema.example.json",
                "reward_contract.example.json",
                "NO_CREDIT_RUNBOOK.md",
            ],
        },
    }


def write_prime_readiness_artifacts(output_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, Path]:
    packet = build_prime_readiness_packet()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cohorts": output_dir / "cohort_manifest.example.json",
        "train_config": output_dir / "prime_train.example.toml",
        "lift_schema": output_dir / "metric_to_lift_schema.example.json",
        "reward_contract": output_dir / "reward_contract.example.json",
        "runbook": output_dir / "NO_CREDIT_RUNBOOK.md",
    }
    paths["cohorts"].write_text(_json(packet["cohort_manifest"]))
    paths["train_config"].write_text(_training_toml(packet["prime_training_config"]))
    paths["lift_schema"].write_text(_json(packet["metric_to_lift_schema"]))
    paths["reward_contract"].write_text(_json(packet["reward_contract"]))
    paths["runbook"].write_text(_runbook_markdown(packet))
    return paths


def run_no_credit_readiness_check(
    output_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    smoke_docs: int = 60,
    task_count: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifacts = write_prime_readiness_artifacts(output_dir)
    payload = asyncio.run(
        run_agent_loop_experiment(
            n_docs=smoke_docs,
            task_count=task_count,
            max_steps=5,
            beam_width=5,
            threshold=0.5,
        )
    )
    truth_gains = [float(episode["metrics"]["truth_gain"]) for episode in payload["episodes"]]
    mean_truth_gain = sum(truth_gains) / len(truth_gains) if truth_gains else 0.0
    prime_cli = _prime_cli_check()
    local_passed = mean_truth_gain >= 0.1 and payload["dataset_metrics"]["pass_rate"] >= 0.8
    blocking_items = []
    if not local_passed:
        blocking_items.append("local no-credit agent-loop smoke failed")
    if not prime_cli["passed"]:
        blocking_items.append("prime CLI is not available or did not respond to --help")
    blocking_items.append("8-H100 Prime capacity or reservation has not been recorded")
    blocking_items.append("heldout baseline eval artifact has not been recorded")
    blocking_items.append("explicit user approval for Prime credit use has not been recorded")
    report = {
        "schema_version": "extension3.no_credit_readiness_report.v1",
        "generated_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "paid_training_launched": False,
        "ready_for_prime_training_launch": False,
        "blocking_items": blocking_items,
        "smoke_docs": smoke_docs,
        "task_count": task_count,
        "checks": {
            "local_agent_loop_smoke": {
                "passed": local_passed,
                "mean_truth_gain": mean_truth_gain,
                "pass_rate": payload["dataset_metrics"]["pass_rate"],
                "mean_best_f1": payload["dataset_metrics"]["mean_best_f1"],
            },
            "prime_cli": prime_cli,
            "training_gate": {
                "passed": True,
                "reason": "This smoke path never invokes prime train run.",
            },
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "next_allowed_action": "Authenticate Prime, confirm 8-H100 capacity, then only launch Prime training after explicit credit approval.",
    }
    report_path = output_dir / "no_credit_readiness_report.json"
    report_md_path = output_dir / "no_credit_readiness_report.md"
    report["artifacts"]["readiness_report"] = str(report_path)
    report["artifacts"]["readiness_report_md"] = str(report_md_path)
    report_path.write_text(_json(report))
    report_md_path.write_text(_readiness_markdown(report) + "\n")
    return report


def _cohort(cohort_id: str, split: str, baseline: dict[str, Any], **changes: Any) -> dict[str, Any]:
    cohort = dict(baseline)
    cohort.update(changes)
    cohort["cohort_id"] = cohort_id
    cohort["split"] = split
    cohort["baseline_cohort_id"] = "c00_baseline_retry"
    cohort["description"] = f"{cohort_id}: vary {cohort['single_changed_variable']} against baseline."
    return cohort


def _prime_cli_check() -> dict[str, Any]:
    prime_path = shutil.which("prime")
    if not prime_path:
        return {
            "passed": False,
            "available": False,
            "reason": "prime CLI not found; install/authenticate before running Prime eval or training.",
        }
    try:
        completed = subprocess.run(
            [prime_path, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "passed": False,
            "available": True,
            "path": prime_path,
            "reason": str(exc),
        }
    return {
        "passed": completed.returncode == 0,
        "available": True,
        "path": prime_path,
        "returncode": completed.returncode,
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _training_toml(config: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Example Prime Hosted Training config for Extension 3.",
            "# Do not run this file until the no-credit readiness checks pass and the user approves Prime credit use.",
            "",
            "[run]",
            f'project = "{config["run"]["project"]}"',
            f'name = "{config["run"]["name"]}"',
            f'purpose = "{config["run"]["purpose"]}"',
            "",
            "[model]",
            f'base = "{config["model"]["candidate_base"]}"',
            f'adapter = "{config["model"]["adapter"]}"',
            "",
            "[hardware]",
            f'target = "{config["hardware"]["target"]}"',
            f'gpu_type = "{config["hardware"]["gpu_type"]}"',
            f'gpu_count = {config["hardware"]["gpu_count"]}',
            "",
            "[algorithm]",
            f'name = "{config["algorithm"]["name"]}"',
            f'rollouts_per_example = {config["algorithm"]["rollouts_per_example"]}',
            f'max_turns = {config["algorithm"]["max_turns"]}',
            f'temperature = {config["algorithm"]["temperature"]}',
            f'top_p = {config["algorithm"]["top_p"]}',
            "",
            "[checkpoints]",
            f'interval = {config["checkpointing"]["checkpoint_interval"]}',
            f'keep_cloud = {str(config["checkpointing"]["keep_cloud_checkpoints"]).lower()}',
            "",
            "[adapters]",
            f'interval = {config["checkpointing"]["adapter_interval"]}',
            f'keep_last = {config["checkpointing"]["adapter_keep_last"]}',
            "",
            "[val]",
            f'interval = {config["checkpointing"]["validation_interval"]}',
            "",
            "[resume]",
            'checkpoint_id = ""',
            f'checkpoint_list_command = "{config["checkpointing"]["checkpoint_list_command"]}"',
            "",
            "[environment]",
            f'id = "{config["environment"]["id"]}"',
            f'cohort_manifest = "{config["environment"]["cohort_manifest"]}"',
            f'train_split = "{config["environment"]["train_split"]}"',
            f'heldout_split = "{config["environment"]["heldout_split"]}"',
            "",
            "[launch_gate]",
            f'requires_prime_credits = {str(config["launch"]["requires_prime_credits"]).lower()}',
            f'command = "{config["launch"]["command"]}"',
            'allowed_only_after = ["local no-credit readiness report has passed", "Prime CLI is authenticated", "8-H100 Prime capacity is available or reserved", "baseline eval artifact exists", "user explicitly approves using Prime credits"]',
            "",
        ]
    )


def _runbook_markdown(packet: dict[str, Any]) -> str:
    smoke = packet["no_credit_smoke"]
    training = packet["prime_training_config"]
    lines = [
        "# Extension 3 Prime No-Credit Runbook",
        "",
        "This runbook validates setup before any Prime training credits are used.",
        "",
        "## No-Credit Smoke",
        "",
    ]
    for idx, command in enumerate(smoke["commands"], start=1):
        label = "Optional if resuming. Run:" if "checkpoints <existing-run-id>" in command else "Run:"
        lines.extend([f"{idx}. {label}", "", "```bash", command, "```", ""])
    lines.extend(
        [
            "Pass criteria:",
            "",
            f"- Mean truth gain >= `{smoke['pass_criteria']['mean_truth_gain_min']}`.",
            f"- Pass rate >= `{smoke['pass_criteria']['pass_rate_min']}`.",
            "- No command in this section may start with `prime train run`.",
            "",
            "## Paid Training Gate",
            "",
            "Only after the smoke passes and the user explicitly approves credit use:",
            "",
            "```bash",
            training["launch"]["command"],
            "```",
            "",
            "Checkpoint settings:",
            "",
            f"- Target hardware: `{training['hardware']['target']}`.",
            f"- Full checkpoints every `{training['checkpointing']['checkpoint_interval']}` steps with `keep_cloud = true`.",
            f"- Adapter uploads every `{training['checkpointing']['adapter_interval']}` steps; keep last `{training['checkpointing']['adapter_keep_last']}` adapters.",
            f"- To resume, run `{training['checkpointing']['checkpoint_list_command']}` and set top-level `checkpoint_id`.",
            "",
            "Stop immediately if baseline eval artifacts are missing, the heldout split is empty, no checkpoint appears after the first interval, or the first pilot run fails to improve heldout reward.",
            "",
        ]
    )
    return "\n".join(lines)


def _readiness_markdown(report: dict[str, Any]) -> str:
    smoke = report["checks"]["local_agent_loop_smoke"]
    prime_cli = report["checks"]["prime_cli"]
    return "\n".join(
        [
            "# Extension 3 No-Credit Readiness Report",
            "",
            f"- paid_training_launched: `{report['paid_training_launched']}`",
            f"- ready_for_prime_training_launch: `{report['ready_for_prime_training_launch']}`",
            f"- smoke_docs: `{report['smoke_docs']}`",
            f"- task_count: `{report['task_count']}`",
            f"- local_smoke_passed: `{smoke['passed']}`",
            f"- mean_truth_gain: `{smoke['mean_truth_gain']:.6f}`",
            f"- pass_rate: `{smoke['pass_rate']:.6f}`",
            f"- prime_cli_available: `{prime_cli.get('available', False)}`",
            "",
            "## Blocking Items",
            "",
            *[f"- {item}" for item in report["blocking_items"]],
            "",
            "## Artifacts",
            "",
            "| artifact | path |",
            "|---|---|",
            *[f"| {key} | `{path}` |" for key, path in report["artifacts"].items()],
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Extension 3 Prime readiness templates and no-credit smoke.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--smoke-docs", type=int, default=60)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--templates-only", action="store_true")
    args = parser.parse_args(argv)

    if args.templates_only:
        paths = write_prime_readiness_artifacts(args.output_dir)
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))
        return

    report = run_no_credit_readiness_check(args.output_dir, smoke_docs=args.smoke_docs, task_count=args.task_count)
    print(json.dumps(report["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
