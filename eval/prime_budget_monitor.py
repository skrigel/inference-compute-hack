from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass


class BudgetMonitorError(RuntimeError):
    pass


@dataclass(frozen=True)
class MonitorConfig:
    run_id: str
    target_cost: float
    hard_limit: float
    poll_seconds: float
    stop: bool


def extract_total_cost(usage_output: str) -> float:
    total_lines = [line for line in usage_output.splitlines() if "Total" in line]
    if not total_lines:
        raise BudgetMonitorError("could not find Total line in Prime usage output")
    matches = re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", total_lines[-1])
    if not matches:
        raise BudgetMonitorError("could not find dollar amount in Prime usage Total line")
    return float(matches[-1])


def budget_decision(total_cost: float, *, target_cost: float, hard_limit: float) -> str:
    if total_cost >= hard_limit:
        return "stop_hard_limit"
    if total_cost >= target_cost:
        return "stop_target"
    return "continue"


def monitor(config: MonitorConfig) -> None:
    while True:
        usage = _run(["prime", "--plain", "train", "usage", config.run_id])
        total_cost = extract_total_cost(usage)
        decision = budget_decision(total_cost, target_cost=config.target_cost, hard_limit=config.hard_limit)
        print(f"run_id={config.run_id} total_cost=${total_cost:.4f} decision={decision}", flush=True)
        if decision != "continue":
            if config.stop:
                _run(["prime", "--plain", "train", "stop", config.run_id])
                print(f"stopped run {config.run_id} at ${total_cost:.4f} ({decision})", flush=True)
            return
        time.sleep(config.poll_seconds)


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise BudgetMonitorError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr.strip()}"
        )
    return completed.stdout


def _parse_args() -> MonitorConfig:
    parser = argparse.ArgumentParser(description="Monitor a Prime Hosted Training run and stop it at a dollar budget.")
    parser.add_argument("run_id")
    parser.add_argument("--target-cost", type=float, default=62.0)
    parser.add_argument("--hard-limit", type=float, default=65.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--no-stop", action="store_true", help="report when over budget but do not stop the run")
    args = parser.parse_args()
    if args.target_cost <= 0:
        raise SystemExit("--target-cost must be positive")
    if args.hard_limit < args.target_cost:
        raise SystemExit("--hard-limit must be greater than or equal to --target-cost")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    return MonitorConfig(
        run_id=args.run_id,
        target_cost=args.target_cost,
        hard_limit=args.hard_limit,
        poll_seconds=args.poll_seconds,
        stop=not args.no_stop,
    )


def main() -> None:
    monitor(_parse_args())


if __name__ == "__main__":
    main()
