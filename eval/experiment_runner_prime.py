#!/usr/bin/env python
"""
Prime Intellect experiment runner for GPU memory optimization experiments.

Runs experiments on Prime Compute pods with H100 GPUs instead of Modal.

Usage:
    python -m eval.experiment_runner_prime --list              # List experiments
    python -m eval.experiment_runner_prime EXP-FP8-001         # Run single experiment
    python -m eval.experiment_runner_prime --all               # Run all experiments
    python -m eval.experiment_runner_prime --all --dry-run     # Preview without running
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.experiment_runner import EXPERIMENTS
from eval.weave_ops import DEFAULT_WEAVE_PROJECT, init_weave, weave_op


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
EXPERIMENT_ROOT = ARTIFACT_DIR / "experiment_results"
PRIME_RESULTS_ROOT = ARTIFACT_DIR / "prime_experiment_results"

# Default Prime pod configuration for experiments
DEFAULT_POD_CONFIG = {
    "gpu_type": "H100_80GB",
    "gpu_count": 1,
    "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    "disk_size": 100,
}

# vLLM setup script to run on the pod
VLLM_SETUP_SCRIPT = """#!/bin/bash
set -e

echo "=== Setting up vLLM environment ==="

# Install vLLM and dependencies
pip install -q vllm==0.8.5 transformers>=4.51.1 huggingface-hub>=0.24.0 httpx>=0.27.0

# Download model
echo "=== Downloading model ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct-AWQ')"

echo "=== Setup complete ==="
"""

# Benchmark script template
BENCHMARK_SCRIPT_TEMPLATE = """#!/usr/bin/env python
import asyncio
import json
import math
import subprocess
import sys
import threading
import time

import httpx

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
H100_SXM_BF16_FLOPS_PER_GPU = 989.5e12

def _benchmark_prompt(i: int, dataset_mode: str = "dynamic") -> str:
    if dataset_mode == "static":
        chunk = "service batch shared-static: retry logic, GPU queue saturation"
    else:
        chunk = f"service batch dynamic-{{i}}: retry logic, GPU queue saturation, fresh version {{i}}"
    predicate = "GPU queue saturation and throughput metrics"
    return f"Chunk: {{chunk}}\\nPredicate: {{predicate}}\\nRelevant? Answer Yes or No:"

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def _read_gpu_sample() -> dict:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2, check=False,
    )
    if proc.returncode != 0:
        return {{"error": proc.stderr.strip()}}
    line = proc.stdout.strip().splitlines()[0]
    values = [part.strip() for part in line.split(",")]
    if len(values) < 6:
        return {{"error": f"unexpected nvidia-smi output: {{line}}"}}
    def _parse(v):
        try: return float(v)
        except: return 0.0
    return {{
        "gpu_utilization_pct": _parse(values[0]),
        "gpu_memory_utilization_pct": _parse(values[1]),
        "gpu_memory_used_mb": _parse(values[2]),
        "gpu_memory_total_mb": _parse(values[3]),
        "gpu_power_w": _parse(values[4]),
        "gpu_power_limit_w": _parse(values[5]),
    }}

async def run_benchmark(
    num_requests: int = 128,
    concurrency: int = 32,
    gpu_memory_utilization: float = {gpu_memory_utilization},
    max_num_batched_tokens: int = {max_num_batched_tokens},
    dataset_mode: str = "dynamic",
    env_vars: dict = None,
) -> dict:
    env_vars = env_vars or {{}}
    port = 8000
    host = "127.0.0.1"
    base_url = f"http://{{host}}:{{port}}"
    api_url = f"{{base_url}}/v1"

    # Build vLLM server command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", host, "--port", str(port),
        "--model", MODEL_NAME,
        "--served-model-name", "tier1-filter",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "4096",
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--quantization", "awq_marlin",
    ]

    # Apply experiment env vars
    env = dict(os.environ)
    env.update(env_vars)

    # Handle KV cache dtype from env
    kv_dtype = env.get("KV_CACHE_DTYPE", "auto")
    if kv_dtype != "auto":
        cmd.extend(["--kv-cache-dtype", kv_dtype])

    logs = []
    gpu_samples = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    def _read_logs():
        for line in proc.stdout:
            if len(logs) < 240:
                logs.append(line.rstrip())
            print(f"[vllm] {{line}}", end="")

    reader = threading.Thread(target=_read_logs, daemon=True)
    reader.start()

    async def _wait_ready(client):
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited: {{logs[-40:]}}")
            try:
                response = await client.get(f"{{api_url}}/models")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("vLLM server did not become ready")

    async def _post_completion(client, idx):
        payload = {{
            "model": "tier1-filter",
            "prompt": _benchmark_prompt(idx, dataset_mode),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 5,
        }}
        started = time.perf_counter()
        response = await client.post(f"{{api_url}}/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        return {{"latency_ms": latency_ms}}

    try:
        limits = httpx.Limits(max_connections=max(concurrency * 2, 8), max_keepalive_connections=max(concurrency, 4))
        async with httpx.AsyncClient(timeout=90.0, limits=limits) as client:
            await _wait_ready(client)
            await _post_completion(client, -1)  # warmup

            semaphore = asyncio.Semaphore(max(1, concurrency))
            gpu_stop = threading.Event()
            gpu_started = time.perf_counter()

            def _sample_gpu():
                while not gpu_stop.is_set():
                    sample = _read_gpu_sample()
                    sample["elapsed_s"] = time.perf_counter() - gpu_started
                    gpu_samples.append(sample)
                    gpu_stop.wait(0.25)

            async def _bounded(idx):
                async with semaphore:
                    return await _post_completion(client, idx)

            sampler = threading.Thread(target=_sample_gpu, daemon=True)
            sampler.start()
            started = time.perf_counter()
            try:
                results = await asyncio.gather(*(_bounded(i) for i in range(num_requests)))
                elapsed_s = time.perf_counter() - started
            finally:
                gpu_stop.set()
                sampler.join(timeout=2)

        latencies = [r["latency_ms"] for r in results]
        numeric_samples = [s for s in gpu_samples if "error" not in s]

        return {{
            "exp_id": "{exp_id}",
            "scenario": "prime_benchmark",
            "num_requests": num_requests,
            "concurrency": concurrency,
            "dataset_mode": dataset_mode,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_batched_tokens": max_num_batched_tokens,
            "env_vars": env_vars,
            "elapsed_s": elapsed_s,
            "requests_per_s": num_requests / elapsed_s,
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "latency_ms_p99": _percentile(latencies, 0.99),
            "gpu_utilization_pct_mean": sum(s.get("gpu_utilization_pct", 0) for s in numeric_samples) / max(len(numeric_samples), 1),
            "gpu_memory_used_mb_max": max((s.get("gpu_memory_used_mb", 0) for s in numeric_samples), default=0),
            "gpu_power_w_mean": sum(s.get("gpu_power_w", 0) for s in numeric_samples) / max(len(numeric_samples), 1),
            "gpu_sample_count": len(numeric_samples),
            "logs_tail": logs[-40:],
        }}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

if __name__ == "__main__":
    env_vars = {env_vars_json}
    result = asyncio.run(run_benchmark(env_vars=env_vars))
    print("\\n=== BENCHMARK RESULT ===")
    print(json.dumps(result, indent=2))
"""


@dataclass
class PodInfo:
    """Information about a Prime pod."""
    pod_id: str
    name: str
    status: str
    ip_address: str | None = None
    ssh_command: str | None = None


def _run_prime(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a Prime CLI command."""
    cmd = ["prime", "--plain"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _run_prime_json(args: list[str]) -> dict | list:
    """Run a Prime CLI command and parse JSON output."""
    result = _run_prime(args + ["--output", "json"], check=False)
    if result.returncode != 0:
        # Try without --output json for commands that don't support it
        result = _run_prime(args, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Prime command failed: {result.stderr}")
        # Try to parse as JSON anyway
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"raw_output": result.stdout}
    return json.loads(result.stdout)


def list_pods() -> list[PodInfo]:
    """List all running Prime pods."""
    result = _run_prime(["pods", "list", "--output", "json"], check=False)
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
        pods = data.get("pods", []) if isinstance(data, dict) else data
        return [
            PodInfo(
                pod_id=pod.get("id", ""),
                name=pod.get("name", ""),
                status=pod.get("status", ""),
                ip_address=pod.get("ip_address"),
                ssh_command=pod.get("ssh_command"),
            )
            for pod in pods
        ]
    except (json.JSONDecodeError, KeyError):
        return []


def get_available_resource_id(
    gpu_type: str = "H100_80GB",
    gpu_count: int = 1,
) -> str:
    """Get an available resource ID from Prime availability list."""
    result = _run_prime(
        ["availability", "list", "--gpu-type", gpu_type, "--gpu-count", str(gpu_count), "--output", "json"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get availability: {result.stderr}")

    try:
        data = json.loads(result.stdout)
        resources = data.get("gpu_resources", [])
        # Find first available resource
        for resource in resources:
            if resource.get("stock_status") == "Available":
                resource_id = resource.get("id")
                if resource_id:
                    print(f"Found available resource: {resource_id} ({resource.get('gpu_type')}, "
                          f"{resource.get('gpu_count')} GPU(s), {resource.get('price_per_hour')}/hr, "
                          f"{resource.get('provider')} in {resource.get('location')})")
                    return resource_id
        raise RuntimeError(f"No available {gpu_count}x {gpu_type} resources found")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to parse availability response: {e}")


def create_pod(
    name: str,
    gpu_type: str = DEFAULT_POD_CONFIG["gpu_type"],
    gpu_count: int = DEFAULT_POD_CONFIG["gpu_count"],
    image: str = DEFAULT_POD_CONFIG["image"],
    disk_size: int = DEFAULT_POD_CONFIG["disk_size"],
    env_vars: dict[str, str] | None = None,
) -> PodInfo:
    """Create a new Prime pod for experiments."""
    # Get available resource ID first
    resource_id = get_available_resource_id(gpu_type=gpu_type, gpu_count=gpu_count)

    cmd = [
        "pods", "create",
        "--id", resource_id,
        "--image", image,
        "--disk-size", str(disk_size),
        "--name", name,
        "-y",  # Skip confirmation
    ]

    if env_vars:
        for key, value in env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

    print(f"Creating pod: {name} (resource {resource_id})")
    result = _run_prime(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create pod: {result.stderr}\nstdout: {result.stdout}")

    # Parse pod ID from output
    output = result.stdout + result.stderr
    pod_id = None

    # Try to parse JSON response first
    try:
        data = json.loads(result.stdout)
        pod_id = data.get("id") or data.get("pod_id") or data.get("pod", {}).get("id")
    except (json.JSONDecodeError, AttributeError):
        pass

    if not pod_id:
        # Look for pod ID in text output
        for line in output.splitlines():
            line_lower = line.lower()
            if "pod" in line_lower and ("created" in line_lower or "id" in line_lower):
                parts = line.split()
                for part in parts:
                    # Pod IDs are typically 6+ alphanumeric chars
                    clean_part = part.strip("'\"()[]{}:,")
                    if len(clean_part) >= 6 and clean_part.replace("-", "").replace("_", "").isalnum():
                        pod_id = clean_part
                        break
            if pod_id:
                break

    if not pod_id:
        # List pods to find the one we just created
        print("Looking up pod ID from list...")
        time.sleep(3)
        pods = list_pods()
        for pod in pods:
            if pod.name == name:
                pod_id = pod.pod_id
                break

    if not pod_id:
        raise RuntimeError(f"Could not determine pod ID from output: {output}")

    print(f"Pod created with ID: {pod_id}")
    return PodInfo(pod_id=pod_id, name=name, status="creating")


def wait_for_pod_ready(pod_id: str, timeout: int = 600) -> PodInfo:
    """Wait for a pod to be ready and return its info."""
    start = time.time()
    while time.time() - start < timeout:
        result = _run_prime(["pods", "status", pod_id], check=False)
        if result.returncode == 0:
            output = result.stdout.lower()
            if "running" in output or "ready" in output:
                # Get pod info
                pods = list_pods()
                for pod in pods:
                    if pod.pod_id == pod_id:
                        return pod
                # Return basic info if not in list
                return PodInfo(pod_id=pod_id, name="", status="running")
        time.sleep(10)
        print(f"Waiting for pod {pod_id}... ({int(time.time() - start)}s)")

    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def terminate_pod(pod_id: str) -> None:
    """Terminate a Prime pod."""
    print(f"Terminating pod: {pod_id}")
    result = _run_prime(["pods", "terminate", pod_id, "-y"], check=False)
    if result.returncode != 0:
        print(f"Warning: Failed to terminate pod {pod_id}: {result.stderr}")


def run_on_pod(pod_id: str, command: str, timeout: int = 1800) -> str:
    """Run a command on a Prime pod via SSH."""
    # Use prime pods ssh to run the command
    cmd = ["prime", "--plain", "pods", "ssh", pod_id, "--", "bash", "-c", command]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed on pod: {result.stderr}\nStdout: {result.stdout}")
    return result.stdout


def upload_to_pod(pod_id: str, local_path: str, remote_path: str) -> None:
    """Upload a file to a Prime pod."""
    # Use prime pods ssh with scp-like functionality or cat through pipe
    with open(local_path, "r") as f:
        content = f.read()

    # Escape content for shell
    escaped = content.replace("'", "'\"'\"'")
    run_on_pod(pod_id, f"cat > {remote_path} << 'SCRIPT_EOF'\n{content}\nSCRIPT_EOF")


def _ensure_experiment_dir(exp_id: str) -> Path:
    """Create experiment artifact directory."""
    exp_dir = PRIME_RESULTS_ROOT / exp_id
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
        "backend": "prime",
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (exp_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def generate_benchmark_script(
    exp_id: str,
    env_vars: dict[str, str],
    gpu_memory_utilization: float = 0.92,
    max_num_batched_tokens: int = 8192,
) -> str:
    """Generate the benchmark script for a specific experiment."""
    return BENCHMARK_SCRIPT_TEMPLATE.format(
        exp_id=exp_id,
        env_vars_json=json.dumps(env_vars),
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
    )


@weave_op(name="eval.experiment.run_prime")
def run_experiment_prime(
    exp_id: str,
    *,
    repetitions: int = 5,
    gpu_count: int = 1,
    dry_run: bool = False,
    keep_pod: bool = False,
) -> dict[str, Any]:
    """Run a single experiment on Prime Intellect compute."""
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_id}. Available: {list(EXPERIMENTS.keys())}")

    config = EXPERIMENTS[exp_id]
    exp_dir = _ensure_experiment_dir(exp_id)
    _write_config(exp_dir, exp_id, config)

    # Parse modal_args to extract max_num_batched_tokens
    max_num_batched_tokens = 8192
    for i, arg in enumerate(config.get("modal_args", [])):
        if arg == "--max-num-batched-tokens" and i + 1 < len(config.get("modal_args", [])):
            max_num_batched_tokens = int(config["modal_args"][i + 1])

    result = {
        "exp_id": exp_id,
        "config": config,
        "backend": "prime",
        "gpu_count": gpu_count,
        "repetitions": repetitions,
        "dry_run": dry_run,
        "runs": [],
    }

    if dry_run:
        print(f"[DRY RUN] Would run experiment {exp_id}: {config['name']}")
        print(f"[DRY RUN] Env vars: {config['env_vars']}")
        print(f"[DRY RUN] Max batched tokens: {max_num_batched_tokens}")
        print(f"[DRY RUN] GPU count: {gpu_count}")
        print(f"[DRY RUN] Repetitions: {repetitions}")
        result["status"] = "dry_run"
        return result

    pod_id = None
    try:
        # Create pod
        pod_name = f"exp-{exp_id.lower()}-{int(time.time())}"
        pod = create_pod(
            name=pod_name,
            gpu_count=gpu_count,
            env_vars=config["env_vars"],
        )
        pod_id = pod.pod_id
        print(f"Created pod: {pod_id}")

        # Wait for pod to be ready
        pod = wait_for_pod_ready(pod_id)
        print(f"Pod ready: {pod_id}")

        # Setup vLLM
        print("Setting up vLLM...")
        run_on_pod(pod_id, VLLM_SETUP_SCRIPT, timeout=900)

        # Generate and upload benchmark script
        benchmark_script = generate_benchmark_script(
            exp_id=exp_id,
            env_vars=config["env_vars"],
            max_num_batched_tokens=max_num_batched_tokens,
        )

        # Upload via echo
        escaped_script = benchmark_script.replace("'", "'\"'\"'")
        run_on_pod(pod_id, f"cat > /tmp/benchmark.py << 'EOF'\n{benchmark_script}\nEOF")

        # Run benchmark multiple times
        for rep in range(repetitions):
            print(f"\n=== Run {rep + 1}/{repetitions} ===")
            try:
                output = run_on_pod(
                    pod_id,
                    "cd /tmp && python benchmark.py",
                    timeout=1200,
                )

                # Parse result from output
                if "=== BENCHMARK RESULT ===" in output:
                    json_start = output.index("=== BENCHMARK RESULT ===") + len("=== BENCHMARK RESULT ===")
                    json_str = output[json_start:].strip()
                    run_result = json.loads(json_str)
                    run_result["run_index"] = rep
                    result["runs"].append(run_result)

                    # Save individual run
                    run_file = exp_dir / "runs" / f"run_{rep:03d}.json"
                    run_file.write_text(json.dumps(run_result, indent=2))

                    print(f"  requests/s: {run_result.get('requests_per_s', 'N/A'):.2f}")
                    print(f"  p50 latency: {run_result.get('latency_ms_p50', 'N/A'):.2f}ms")
                    print(f"  GPU util: {run_result.get('gpu_utilization_pct_mean', 'N/A'):.1f}%")
                else:
                    print(f"Warning: Could not parse result from run {rep}")
                    result["runs"].append({"run_index": rep, "error": "parse_failed", "output": output[-2000:]})

            except Exception as e:
                print(f"Error in run {rep}: {e}")
                result["runs"].append({"run_index": rep, "error": str(e)})

        # Aggregate results
        successful_runs = [r for r in result["runs"] if "error" not in r]
        if successful_runs:
            result["aggregated"] = {
                "run_count": len(successful_runs),
                "requests_per_s_mean": sum(r["requests_per_s"] for r in successful_runs) / len(successful_runs),
                "latency_ms_p50_mean": sum(r["latency_ms_p50"] for r in successful_runs) / len(successful_runs),
                "latency_ms_p95_mean": sum(r["latency_ms_p95"] for r in successful_runs) / len(successful_runs),
                "gpu_utilization_pct_mean": sum(r["gpu_utilization_pct_mean"] for r in successful_runs) / len(successful_runs),
            }
            result["status"] = "success"
        else:
            result["status"] = "failed"

        # Save aggregated result
        (exp_dir / "aggregated.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"Experiment failed: {e}")

    finally:
        if pod_id and not keep_pod:
            terminate_pod(pod_id)
        elif pod_id and keep_pod:
            print(f"Keeping pod for debugging: {pod_id}")

    return result


def run_all_experiments_prime(
    *,
    repetitions: int = 5,
    gpu_count: int = 1,
    dry_run: bool = False,
    experiments: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all experiments on Prime Intellect compute."""
    results: dict[str, dict[str, Any]] = {}
    exp_ids = experiments or list(EXPERIMENTS.keys())

    print(f"\n{'='*60}")
    print(f"Running {len(exp_ids)} experiments on Prime Intellect")
    print(f"Repetitions: {repetitions}")
    print(f"GPU count: {gpu_count}")
    print(f"{'='*60}\n")

    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n[{i}/{len(exp_ids)}] Starting {exp_id}: {EXPERIMENTS[exp_id]['name']}")
        print("-" * 40)

        start_time = time.time()
        result = run_experiment_prime(
            exp_id,
            repetitions=repetitions,
            gpu_count=gpu_count,
            dry_run=dry_run,
        )
        elapsed = time.time() - start_time

        result["elapsed_seconds"] = elapsed
        results[exp_id] = result

        status = result.get("status", "unknown")
        print(f"[{i}/{len(exp_ids)}] {exp_id}: {status} ({elapsed:.1f}s)")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GPU optimization experiments on Prime Intellect")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID to run (e.g., EXP-FP8-001)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per experiment (5 recommended for statistical confidence)")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs per pod")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--keep-pod", action="store_true", help="Keep pod after experiment for debugging")
    parser.add_argument("--weave", action="store_true", help="Enable Weave tracing")
    parser.add_argument("--weave-project", default=DEFAULT_WEAVE_PROJECT)
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run")
    args = parser.parse_args(argv)

    if args.list:
        print("Available experiments:")
        for exp_id, config in EXPERIMENTS.items():
            print(f"  {exp_id}: {config['name']}")
            print(f"    Hypothesis: {config['hypothesis']}")
            print(f"    Env vars: {config['env_vars']}")
            print()
        return

    if args.weave:
        init_weave(args.weave_project)

    if args.all:
        results = run_all_experiments_prime(
            repetitions=args.repetitions,
            gpu_count=args.gpu_count,
            dry_run=args.dry_run,
            experiments=args.experiments,
        )

        # Summary
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

    elif args.exp_id:
        result = run_experiment_prime(
            args.exp_id,
            repetitions=args.repetitions,
            gpu_count=args.gpu_count,
            dry_run=args.dry_run,
            keep_pod=args.keep_pod,
        )
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.error("Specify an experiment ID or --list or --all")


if __name__ == "__main__":
    main()
