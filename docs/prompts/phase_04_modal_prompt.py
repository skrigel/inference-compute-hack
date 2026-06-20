#!/usr/bin/env python3
"""Emit the Phase 04 prompt with Modal-shell-specific instructions.

Usage from a Modal shell or repo checkout:

    python docs/prompts/phase_04_modal_prompt.py
    python docs/prompts/phase_04_modal_prompt.py --output /tmp/phase04_prompt.md
    python docs/prompts/phase_04_modal_prompt.py --raw

The source of truth remains `phase-04-gpu-cluster-agent.md`; this script wraps it
with Modal-specific context so the agent does not accidentally rebuild the SSH
cluster path while a Modal implementation already exists in `inference/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROMPT_MD = HERE / "phase-04-gpu-cluster-agent.md"


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def modal_wrapper(include_metadata: bool = True) -> str:
    branch = git_value("branch", "--show-current") or "unknown"
    commit = git_value("rev-parse", "--short", "HEAD") or "unknown"
    status = git_value("status", "--short", "--branch") or "unknown"
    parts = [
        textwrap.dedent(
            """\
            # Modal Shell Wrapper For Phase 04

            You are an autonomous coding agent running in a Modal shell or Modal-backed
            repo checkout. The full Phase 04 prompt below was originally written for an
            SSH GPU cluster. Interpret cluster/SSH instructions through the existing
            Modal implementation instead of starting a parallel serving path.
            """
        )
    ]

    if include_metadata:
        parts.append(
            f"""## Local Repo Metadata

- branch: `{branch}`
- commit: `{commit}`

```text
{status}
```"""
        )

    parts.append(
        textwrap.dedent(
            """\
            ## Modal-Specific Starting Point

            First inspect these files:

            - `inference/modal_app.py`
            - `inference/modal_client.py`
            - `inference/config.py`
            - `docs/prompts/phase-04-gpu-cluster-agent.md`

            The remote already has a Modal implementation:

            - `inference/modal_app.py` defines the Modal app `grep-for-meaning-scorer`
              and vLLM engine settings.
            - `inference/modal_client.py` implements the backend `ScorerClient` path.
            - `SCORER_BACKEND=modal` is the Modal backend switch in `inference/config.py`.

            Do not duplicate this with a separate SSH-only `VLLMScorer` unless the
            Modal path proves insufficient. Prefer improving the Modal path and keeping
            `SCORER_BACKEND=mock` green.

            ## Modal Commands To Try First

            ```bash
            git fetch --prune origin
            git switch main
            git pull --ff-only
            python -m pip install -r backend/requirements.txt
            python -m pip install -r performance/requirements.txt
            python -m pip install -r eval/requirements.txt
            python -m pip install modal
            modal token new  # only if MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are absent
            modal run inference/modal_app.py::test
            modal deploy inference/modal_app.py
            SCORER_BACKEND=modal python -m eval.bench --smoke
            ```

            ## Modal vLLM Metrics And Memory Work

            Fold the requested vLLM settings into `inference/modal_app.py`, not just a
            standalone shell script:

            - Ensure prefix caching remains enabled.
            - Add/enable the vLLM engine equivalent of `--enable-mfu-metrics` if the
              installed vLLM Python API supports it. If unsupported, document the exact
              installed vLLM version and the missing option.
            - Parameterize `gpu_memory_utilization` via an environment variable such as
              `GPU_MEMORY_UTILIZATION`, defaulting to the current chosen value in the app
              after checking the installed vLLM docs/version.
            - Record `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`,
              `vllm:num_requests_waiting`, and MFU metrics if Modal exposes vLLM
              Prometheus metrics. If the in-process `vllm.LLM` path does not expose
              `/metrics`, add a Modal method or artifact that records equivalent engine
              stats, and document the limitation.
            - Run a small GPU memory sweep, for example `0.80 0.85 0.90 0.92 0.95`,
              recording OOM/no-OOM, p50/p95, KV-cache usage, queue saturation, and MFU.

            Required artifact:

            - `eval/artifacts/phase04_gpu_memory_sweep.json`

            ## Continue With The Full Phase 04 Prompt

            The full source prompt follows. Preserve its quality-gate-first rule and
            verification requirements.

            ---

            """
        )
    )
    return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n\n"


def build_prompt(raw: bool = False, include_metadata: bool = True) -> str:
    if not PROMPT_MD.exists():
        raise FileNotFoundError(f"Missing prompt markdown: {PROMPT_MD}")
    prompt = PROMPT_MD.read_text()
    if raw:
        return prompt
    return modal_wrapper(include_metadata=include_metadata) + prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Modal-ready Phase 04 agent prompt.")
    parser.add_argument("--output", type=Path, help="write prompt to this file instead of stdout")
    parser.add_argument("--raw", action="store_true", help="print only the markdown source prompt")
    parser.add_argument("--no-metadata", action="store_true", help="omit local branch/commit/status")
    args = parser.parse_args(argv)

    prompt = build_prompt(raw=args.raw, include_metadata=not args.no_metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prompt)
        print(f"wrote {args.output}", file=sys.stderr)
        return 0

    sys.stdout.write(prompt)
    if not prompt.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
