from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_DIR = Path("eval/configs/extension3_modal")
DEFAULT_ARTIFACT_DIR = Path("eval/artifacts/modal_checkpoints")
SCHEMA_VERSION = "extension3.modal_checkpoint.v1"
MODAL_H100_USD_PER_SEC = 0.001097


def build_modal_checkpoint_packet(
    *,
    budget_usd: float = 18.0,
    gpu_count: int = 6,
    h100_usd_per_sec: float = MODAL_H100_USD_PER_SEC,
    reserve_fraction: float = 0.20,
) -> dict[str, Any]:
    if budget_usd <= 0:
        raise ValueError("budget_usd must be positive")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if h100_usd_per_sec <= 0:
        raise ValueError("h100_usd_per_sec must be positive")
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0.0, 1.0)")

    six_gpu_usd_per_sec = h100_usd_per_sec * gpu_count
    max_wall_seconds = budget_usd / six_gpu_usd_per_sec
    planned_training_seconds = math.floor(max_wall_seconds * (1.0 - reserve_fraction))
    hard_timeout_seconds = math.floor(min(max_wall_seconds * 0.90, planned_training_seconds + 240))
    planned_training_gpu_cost = planned_training_seconds * six_gpu_usd_per_sec
    save_every_seconds = 180
    expected_checkpoints = max(1, planned_training_seconds // save_every_seconds)
    return {
        "schema_version": SCHEMA_VERSION,
        "modal": {
            "gpu_request": f"H100:{gpu_count}",
            "gpu_count": gpu_count,
            "volume_name": "extension3-agent-loop-checkpoints",
            "volume_mount": "/checkpoints",
            "modal_function_timeout_seconds": hard_timeout_seconds,
            "paid_gpu_launch_allowed": False,
        },
        "budget": {
            "budget_usd": budget_usd,
            "h100_usd_per_sec": h100_usd_per_sec,
            "gpu_count": gpu_count,
            "gpu_usd_per_wall_second": six_gpu_usd_per_sec,
            "max_wall_seconds_at_full_budget": math.floor(max_wall_seconds),
            "planned_training_seconds": planned_training_seconds,
            "planned_training_minutes": round(planned_training_seconds / 60.0, 2),
            "planned_training_gpu_cost_usd": round(planned_training_gpu_cost, 2),
            "reserve_fraction": reserve_fraction,
            "reserve_usd": round(budget_usd - planned_training_gpu_cost, 2),
            "notes": "GPU-only estimate; CPU, memory, image build, storage, and egress can add cost.",
        },
        "checkpoint_policy": {
            "save_every_seconds": save_every_seconds,
            "save_every_optimizer_steps": 25,
            "save_first_checkpoint_after_steps": 5,
            "expected_checkpoints": expected_checkpoints,
            "keep_last": 4,
            "keep_best": True,
            "write_latest_pointer": True,
            "resume_from_latest": True,
            "atomic_write": "write checkpoint.tmp then rename to checkpoint.json and update latest.json last",
            "must_save_fields": [
                "model_or_adapter_state",
                "optimizer_state",
                "scheduler_state",
                "rng_state",
                "global_step",
                "cohort_id",
                "reward_window",
                "git_commit",
                "config_hash",
            ],
        },
        "run_plan": {
            "stage_0_no_credit": "Run local checkpoint dry-run and Modal CPU smoke only.",
            "stage_1_paid_micro": "Optional paid 6-H100 run capped at first checkpoint and <= $2 GPU estimate.",
            "stage_2_paid_budget": "Full $18 budget-capped run after stage 1 produces a resumable checkpoint.",
            "stop_conditions": [
                "estimated spend reaches 80% of budget",
                "no checkpoint has been written within the first 6 minutes",
                "reward is NaN or constant for two checkpoint windows",
                "selected bytes trend toward full-corpus selection",
                "latest checkpoint cannot be loaded in a resume dry run",
            ],
        },
    }


def write_modal_checkpoint_artifacts(
    output_dir: Path = DEFAULT_CONFIG_DIR,
    *,
    budget_usd: float = 18.0,
    gpu_count: int = 6,
) -> dict[str, Path]:
    packet = build_modal_checkpoint_packet(budget_usd=budget_usd, gpu_count=gpu_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "policy": output_dir / "modal_checkpoint_policy.example.json",
        "train_config": output_dir / "modal_6h100_checkpoint_train.example.toml",
        "runbook": output_dir / "MODAL_CHECKPOINT_RUNBOOK.md",
    }
    paths["policy"].write_text(_json(packet))
    paths["train_config"].write_text(_modal_train_toml(packet))
    paths["runbook"].write_text(_runbook_markdown(packet))
    return paths


def run_local_checkpoint_dry_run(
    output_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    total_steps: int = 12,
    save_every_steps: int = 3,
    budget_usd: float = 18.0,
    gpu_count: int = 6,
) -> dict[str, Any]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if save_every_steps <= 0:
        raise ValueError("save_every_steps must be positive")

    started = time.perf_counter()
    artifacts = write_modal_checkpoint_artifacts(output_dir, budget_usd=budget_usd, gpu_count=gpu_count)
    checkpoint_dir = output_dir / "dry_run_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_written: list[str] = []
    latest_path: Path | None = None
    for step in range(1, total_steps + 1):
        if step % save_every_steps != 0:
            continue
        ckpt_path = checkpoint_dir / f"checkpoint_step_{step:06d}.json"
        tmp_path = ckpt_path.with_suffix(".tmp")
        payload = {
            "schema_version": "extension3.local_checkpoint_dry_run.v1",
            "step": step,
            "cohort_id": "dry_run",
            "reward_window": [0.0, 0.1, 0.2],
            "paid_modal_gpu_launched": False,
        }
        tmp_path.write_text(_json(payload))
        tmp_path.replace(ckpt_path)
        latest_path = ckpt_path
        latest_pointer = {
            "latest_checkpoint": str(ckpt_path),
            "step": step,
            "paid_modal_gpu_launched": False,
        }
        (checkpoint_dir / "latest.json").write_text(_json(latest_pointer))
        checkpoints_written.append(str(ckpt_path))

    passed = bool(checkpoints_written and latest_path and latest_path.exists())
    report = {
        "schema_version": "extension3.modal_checkpoint_dry_run_report.v1",
        "paid_modal_gpu_launched": False,
        "passed": passed,
        "generated_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "total_steps": total_steps,
        "save_every_steps": save_every_steps,
        "latest_step": json.loads((checkpoint_dir / "latest.json").read_text())["step"] if passed else None,
        "latest_checkpoint_path": str(latest_path) if latest_path else None,
        "checkpoints_written": checkpoints_written,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    report_path = output_dir / "local_checkpoint_dry_run_report.json"
    report_md_path = output_dir / "local_checkpoint_dry_run_report.md"
    report["artifacts"]["dry_run_report"] = str(report_path)
    report["artifacts"]["dry_run_report_md"] = str(report_md_path)
    report_path.write_text(_json(report))
    report_md_path.write_text(_dry_run_markdown(report) + "\n")
    return report


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _modal_train_toml(packet: dict[str, Any]) -> str:
    modal = packet["modal"]
    budget = packet["budget"]
    checkpoint = packet["checkpoint_policy"]
    return "\n".join(
        [
            "# Example Modal 6-H100 checkpointed post-training config for Extension 3.",
            "# Do not launch GPU training until the local dry-run passes and the user approves spend.",
            "",
            "[modal]",
            f'gpu_request = "{modal["gpu_request"]}"',
            f'volume_name = "{modal["volume_name"]}"',
            f'volume_mount = "{modal["volume_mount"]}"',
            f'timeout_seconds = {modal["modal_function_timeout_seconds"]}',
            "paid_gpu_launch_allowed = false",
            "",
            "[budget]",
            f'budget_usd = {budget["budget_usd"]}',
            f'h100_usd_per_sec = {budget["h100_usd_per_sec"]}',
            f'gpu_count = {budget["gpu_count"]}',
            f'max_wall_seconds_at_full_budget = {budget["max_wall_seconds_at_full_budget"]}',
            f'planned_training_seconds = {budget["planned_training_seconds"]}',
            f'planned_training_gpu_cost_usd = {budget["planned_training_gpu_cost_usd"]}',
            f'reserve_usd = {budget["reserve_usd"]}',
            "",
            "[checkpoint]",
            f'save_every_seconds = {checkpoint["save_every_seconds"]}',
            f'save_every_optimizer_steps = {checkpoint["save_every_optimizer_steps"]}',
            f'save_first_checkpoint_after_steps = {checkpoint["save_first_checkpoint_after_steps"]}',
            f'keep_last = {checkpoint["keep_last"]}',
            f'keep_best = {str(checkpoint["keep_best"]).lower()}',
            f'resume_from_latest = {str(checkpoint["resume_from_latest"]).lower()}',
            f'write_latest_pointer = {str(checkpoint["write_latest_pointer"]).lower()}',
            "",
            "[training]",
            'base_model = "Qwen/Qwen2.5-1.5B-Instruct"',
            'method = "LoRA-GRPO-pilot"',
            'cohort_manifest = "eval/configs/extension3_prime/cohort_manifest.example.json"',
            'reward_contract = "eval/configs/extension3_prime/reward_contract.example.json"',
            'launch_command = "modal run inference/modal_app.py::extension3_checkpointed_posttrain --config eval/configs/extension3_modal/modal_6h100_checkpoint_train.example.toml"',
            "",
        ]
    )


def _runbook_markdown(packet: dict[str, Any]) -> str:
    modal = packet["modal"]
    budget = packet["budget"]
    checkpoint = packet["checkpoint_policy"]
    run_plan = packet["run_plan"]
    lines = [
        "# Extension 3 Modal Checkpoint Runbook",
        "",
        "This packet is for the Modal post-training path when only an $18 budget is available.",
        "It is intentionally checkpoint-heavy and budget-capped. It does not launch GPU training by itself.",
        "",
        "## Budget",
        "",
        f"- GPU request: `{modal['gpu_request']}`.",
        f"- H100 price assumption: `${budget['h100_usd_per_sec']}` per GPU-second.",
        f"- Full-budget wall time: `{budget['max_wall_seconds_at_full_budget']}` seconds.",
        f"- Planned training window: `{budget['planned_training_seconds']}` seconds / `{budget['planned_training_minutes']}` minutes.",
        f"- Planned GPU-only cost: `${budget['planned_training_gpu_cost_usd']}`.",
        f"- Reserve: `${budget['reserve_usd']}` for CPU, memory, startup, checkpoint writes, eval, and pricing drift.",
        "",
        "## Checkpoint Policy",
        "",
        f"- Save every `{checkpoint['save_every_seconds']}` seconds or `{checkpoint['save_every_optimizer_steps']}` optimizer steps, whichever comes first.",
        f"- Save the first checkpoint after `{checkpoint['save_first_checkpoint_after_steps']}` optimizer steps.",
        f"- Keep last `{checkpoint['keep_last']}` checkpoints plus best checkpoint.",
        "- Update `latest.json` only after checkpoint state is fully written.",
        "- Resume from `latest.json` by default.",
        "",
        "## No-Credit Dry Run",
        "",
        "```bash",
        "python -m eval.agent_loop_modal_checkpoints --output-dir eval/artifacts/modal_checkpoints --total-steps 12 --save-every-steps 3",
        "```",
        "",
        "This writes fake checkpoint metadata locally and proves the resume pointer path without using GPUs.",
        "",
        "## Paid Modal Stages",
        "",
        f"1. Stage 0: {run_plan['stage_0_no_credit']}",
        f"2. Stage 1: {run_plan['stage_1_paid_micro']}",
        f"3. Stage 2: {run_plan['stage_2_paid_budget']}",
        "",
        "## Stop Conditions",
        "",
    ]
    lines.extend(f"- {item}" for item in run_plan["stop_conditions"])
    lines.extend(
        [
            "",
            "## Future Paid Launch Shape",
            "",
            "```bash",
            "modal run inference/modal_app.py::extension3_checkpointed_posttrain \\",
            "  --config eval/configs/extension3_modal/modal_6h100_checkpoint_train.example.toml",
            "```",
            "",
            "The launch entrypoint is intentionally not implemented yet. Add it only after the dry run passes, the Modal account has budget, and the user approves the paid run.",
            "",
        ]
    )
    return "\n".join(lines)


def _dry_run_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Extension 3 Modal Checkpoint Dry Run",
            "",
            f"- paid_modal_gpu_launched: `{report['paid_modal_gpu_launched']}`",
            f"- passed: `{report['passed']}`",
            f"- total_steps: `{report['total_steps']}`",
            f"- save_every_steps: `{report['save_every_steps']}`",
            f"- latest_step: `{report['latest_step']}`",
            f"- latest_checkpoint_path: `{report['latest_checkpoint_path']}`",
            f"- checkpoints_written: `{len(report['checkpoints_written'])}`",
            "",
            "## Artifacts",
            "",
            "| artifact | path |",
            "|---|---|",
            *[f"| {key} | `{path}` |" for key, path in report["artifacts"].items()],
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Modal 6-H100 checkpoint policy and local dry-run artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--budget-usd", type=float, default=18.0)
    parser.add_argument("--gpu-count", type=int, default=6)
    parser.add_argument("--total-steps", type=int, default=12)
    parser.add_argument("--save-every-steps", type=int, default=3)
    parser.add_argument("--templates-only", action="store_true")
    args = parser.parse_args(argv)

    if args.templates_only:
        paths = write_modal_checkpoint_artifacts(args.output_dir, budget_usd=args.budget_usd, gpu_count=args.gpu_count)
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))
        return

    report = run_local_checkpoint_dry_run(
        args.output_dir,
        total_steps=args.total_steps,
        save_every_steps=args.save_every_steps,
        budget_usd=args.budget_usd,
        gpu_count=args.gpu_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
