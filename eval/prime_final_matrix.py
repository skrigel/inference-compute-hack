from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from eval.experiment_runner_prime import create_pod, run_on_pod, terminate_pod, wait_for_pod_ready
from eval.rag_compare import DEFAULT_QUERY, _measure_rag_size


ARTIFACT_DIR = Path("eval/artifacts")
DEFAULT_OUTPUT_JSON = ARTIFACT_DIR / "prime_final_h100_rag_matrix.json"
DEFAULT_OUTPUT_MD = ARTIFACT_DIR / "prime_final_h100_rag_matrix.md"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
VLLM_VERSION = "0.6.6.post1"


REMOTE_MATRIX_SCRIPT = r'''
import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import httpx

MODEL_NAME = "__MODEL_NAME__"
VLLM_VERSION = "__VLLM_VERSION__"


def prompt(i, dataset_mode):
    if dataset_mode == "static":
        chunk = "service batch shared-static: retry logic, GPU queue saturation"
    else:
        chunk = f"service batch dynamic-{i}: retry logic, GPU queue saturation, fresh version {i}"
    predicate = "GPU queue saturation and throughput metrics"
    return f"Chunk: {chunk}\nPredicate: {predicate}\nRelevant? Answer Yes or No:"


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def gpu_samples():
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or f"nvidia-smi exited {proc.returncode}"}]
    rows = []
    for line in proc.stdout.strip().splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) < 7:
            continue
        def parse(value):
            try:
                return float(value)
            except ValueError:
                return 0.0
        rows.append(
            {
                "index": int(parse(values[0])),
                "gpu_utilization_pct": parse(values[1]),
                "gpu_memory_utilization_pct": parse(values[2]),
                "gpu_memory_used_mb": parse(values[3]),
                "gpu_memory_total_mb": parse(values[4]),
                "gpu_power_w": parse(values[5]),
                "gpu_power_limit_w": parse(values[6]),
            }
        )
    return rows


def summarize_samples(samples, active_count):
    active = [
        item
        for snapshot in samples
        for item in snapshot
        if "error" not in item and int(item.get("index", -1)) < active_count
    ]
    if not active:
        return {
            "gpu_sample_count": 0,
            "gpu_utilization_pct_mean": None,
            "gpu_utilization_pct_max": None,
            "gpu_memory_used_mb_max": None,
            "gpu_power_w_mean": None,
            "gpu_power_w_max": None,
        }
    def vals(key):
        return [float(item[key]) for item in active if item.get(key) is not None]
    def mean(key):
        value = vals(key)
        return sum(value) / len(value) if value else None
    def maxv(key):
        value = vals(key)
        return max(value) if value else None
    return {
        "gpu_sample_count": len(active),
        "gpu_utilization_pct_mean": mean("gpu_utilization_pct"),
        "gpu_utilization_pct_max": maxv("gpu_utilization_pct"),
        "gpu_memory_used_mb_max": maxv("gpu_memory_used_mb"),
        "gpu_power_w_mean": mean("gpu_power_w"),
        "gpu_power_w_max": maxv("gpu_power_w"),
    }


def start_server(gpu_index, port, max_num_batched_tokens):
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        MODEL_NAME,
        "--served-model-name",
        "tier1-filter",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.92",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "256",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--quantization",
        "awq_marlin",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    logs = []
    def read_logs():
        assert proc.stdout is not None
        for line in proc.stdout:
            if len(logs) < 80:
                logs.append(line.rstrip())
            print(f"[vllm:{gpu_index}] {line}", end="")
    threading.Thread(target=read_logs, daemon=True).start()
    return proc, logs, cmd


def wait_cuda_ready(gpu_count):
    """Wait for CUDA runtime init, not just nvidia-smi/NVML visibility."""
    probe = r"""
import sys
import torch

expected = int(sys.argv[1])
count = torch.cuda.device_count()
if count < expected:
    raise SystemExit(f"expected at least {expected} CUDA devices, saw {count}")
for idx in range(expected):
    torch.cuda.set_device(idx)
    tensor = torch.empty((1,), device="cuda")
    tensor.fill_(1)
    torch.cuda.synchronize()
print(f"cuda-ready devices={count}")
"""
    deadline = time.time() + 600
    last_output = ""
    while time.time() < deadline:
        proc = subprocess.run(
            [sys.executable, "-c", probe, str(gpu_count)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        last_output = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            print(last_output)
            return
        print(f"CUDA not ready yet: {last_output}")
        time.sleep(10)
    raise TimeoutError(f"CUDA did not become ready: {last_output}")


async def wait_ready(port, proc, logs):
    deadline = time.time() + 900
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server on port {port} exited: {logs[-40:]}")
            try:
                response = await client.get(f"http://127.0.0.1:{port}/v1/models")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"server on port {port} did not become ready")


async def post_completion(client, port, idx, dataset_mode):
    payload = {
        "model": "tier1-filter",
        "prompt": prompt(idx, dataset_mode),
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 5,
    }
    started = time.perf_counter()
    response = await client.post(f"http://127.0.0.1:{port}/v1/completions", json=payload)
    latency_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    usage = response.json().get("usage") or {}
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or 0.0),
        "total_tokens": float(usage.get("total_tokens") or 0.0),
    }


async def run_load(ports, scenario):
    concurrency = scenario["concurrency"]
    requests_per_replica = scenario["num_requests"]
    samples = []
    stop = threading.Event()

    def sample_loop():
        while not stop.is_set():
            samples.append(gpu_samples())
            stop.wait(0.25)

    async def run_port(port, replica_index):
        limits = httpx.Limits(max_connections=max(concurrency * 2, 8), max_keepalive_connections=max(concurrency, 4))
        async with httpx.AsyncClient(timeout=90.0, limits=limits) as client:
            semaphore = asyncio.Semaphore(max(1, concurrency))
            async def bounded(i):
                async with semaphore:
                    return await post_completion(
                        client,
                        port,
                        replica_index * requests_per_replica + i,
                        scenario["dataset_mode"],
                    )
            return await asyncio.gather(*(bounded(i) for i in range(requests_per_replica)))

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        nested = await asyncio.gather(*(run_port(port, idx) for idx, port in enumerate(ports)))
        elapsed_s = time.perf_counter() - started
    finally:
        stop.set()
        sampler.join(timeout=2)
    results = [item for group in nested for item in group]
    latencies = [item["latency_ms"] for item in results]
    total_requests = len(results)
    prompt_tokens = sum(item["prompt_tokens"] for item in results)
    total_tokens = sum(item["total_tokens"] for item in results)
    return {
        "requests": total_requests,
        "elapsed_s": elapsed_s,
        "requests_per_s": total_requests / elapsed_s,
        "prompt_tokens_per_s": prompt_tokens / elapsed_s,
        "total_tokens_per_s": total_tokens / elapsed_s,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
        "latency_ms_max": max(latencies) if latencies else 0.0,
        **summarize_samples(samples, len(ports)),
    }


async def main():
    gpu_count = int(os.environ["FINAL_GPU_COUNT"])
    max_num_batched_tokens = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384"))
    active_counts = [int(part) for part in os.environ.get("ACTIVE_COUNTS", "1").split(",") if part]
    base_port = 8000
    servers = []
    try:
        wait_cuda_ready(gpu_count)
        for gpu_index in range(gpu_count):
            proc, logs, cmd = start_server(gpu_index, base_port + gpu_index, max_num_batched_tokens)
            servers.append((proc, logs, cmd))
            await wait_ready(base_port + gpu_index, proc, logs)
            await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=30.0) as client:
            await asyncio.gather(*(post_completion(client, base_port + idx, -1, "static") for idx in range(gpu_count)))

        scenarios = [
            {"name": "single_user_static", "dataset_mode": "static", "num_requests": 32, "concurrency": 1},
            {"name": "multi_user_static", "dataset_mode": "static", "num_requests": 96, "concurrency": 32},
            {"name": "single_user_dynamic", "dataset_mode": "dynamic", "num_requests": 32, "concurrency": 1},
            {"name": "multi_user_dynamic", "dataset_mode": "dynamic", "num_requests": 96, "concurrency": 32},
        ]
        matrix = {}
        for active_count in active_counts:
            ports = [base_port + idx for idx in range(active_count)]
            matrix[str(active_count)] = {}
            for scenario in scenarios:
                print(f"=== {active_count} GPU(s): {scenario['name']} ===")
                matrix[str(active_count)][scenario["name"]] = await run_load(ports, scenario)
        print("=== FINAL MATRIX RESULT ===")
        print(json.dumps({
            "gpu_count": gpu_count,
            "active_counts": active_counts,
            "model": MODEL_NAME,
            "vllm_version": VLLM_VERSION,
            "max_num_batched_tokens": max_num_batched_tokens,
            "matrix": matrix,
        }, indent=2, sort_keys=True))
    finally:
        for proc, _, _ in servers:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()


asyncio.run(main())
'''


def _setup_script() -> str:
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        python -m pip install -q 'vllm=={VLLM_VERSION}' 'transformers>=4.45,<5' 'huggingface-hub>=0.24.0' 'httpx>=0.27.0'
        python - <<'PY'
        from huggingface_hub import snapshot_download
        snapshot_download('{MODEL_NAME}')
        PY
        """
    )


def _remote_script() -> str:
    return REMOTE_MATRIX_SCRIPT.replace("__MODEL_NAME__", MODEL_NAME).replace("__VLLM_VERSION__", VLLM_VERSION)


def _parse_marked_json(output: str, marker: str) -> dict[str, Any]:
    if marker not in output:
        raise RuntimeError(f"Could not parse final matrix output:\n{output[-4000:]}")
    tail = output.split(marker, 1)[1].lstrip()
    payload, _ = json.JSONDecoder().raw_decode(tail)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object after {marker!r}, got {type(payload).__name__}")
    return payload


def _run_prime_matrix(
    *,
    gpu_count: int,
    active_counts: list[int],
    keep_pod: bool,
    pod_id: str | None = None,
) -> dict[str, Any]:
    created_pod = pod_id is None
    try:
        if pod_id is None:
            pod = create_pod(name=f"final-matrix-{gpu_count}gpu-{int(time.time())}", gpu_count=gpu_count)
            pod_id = pod.pod_id
        assert pod_id is not None
        ready = wait_for_pod_ready(pod_id, timeout=900)
        print(f"Prime pod ready: {ready.pod_id}")
        run_on_pod(pod_id, _setup_script(), timeout=1800)
        script = _remote_script()
        run_on_pod(pod_id, f"cat > /tmp/prime_final_matrix.py <<'PY'\n{script}\nPY", timeout=120)
        command = (
            f"cd /tmp && FINAL_GPU_COUNT={gpu_count} "
            f"ACTIVE_COUNTS={','.join(str(item) for item in active_counts)} "
            "MAX_NUM_BATCHED_TOKENS=16384 python /tmp/prime_final_matrix.py"
        )
        output = run_on_pod(pod_id, command, timeout=3600)
        marker = "=== FINAL MATRIX RESULT ==="
        payload = _parse_marked_json(output, marker)
        payload["prime_pod"] = {"pod_id": pod_id, "gpu_count": gpu_count, "created_by_runner": created_pod}
        return payload
    finally:
        if pod_id and created_pod and not keep_pod:
            terminate_pod(pod_id)


def _projection(matrix: dict[str, Any], *, project_to: int) -> dict[str, Any] | None:
    if "1" not in matrix or "2" not in matrix or project_to <= 2:
        return None
    projected = {}
    for scenario, two_row in matrix["2"].items():
        one_row = matrix["1"][scenario]
        efficiency = two_row["requests_per_s"] / max(2.0 * one_row["requests_per_s"], 1e-9)
        conservative_efficiency = min(efficiency, 1.0)
        projected[scenario] = {
            "source": "projected_8x_from_actual_1x_2x",
            "actual_1x_requests_per_s": one_row["requests_per_s"],
            "actual_2x_requests_per_s": two_row["requests_per_s"],
            "observed_2x_efficiency": efficiency,
            "projected_requests_per_s_linear_from_2x": two_row["requests_per_s"] * (project_to / 2.0),
            "projected_requests_per_s_efficiency_capped": one_row["requests_per_s"] * project_to * conservative_efficiency,
            "projected_latency_ms_p50": two_row["latency_ms_p50"],
            "projected_latency_ms_p95": two_row["latency_ms_p95"],
        }
    return projected


def _with_rag(payload: dict[str, Any], *, rag_sizes: list[int], rag_runs: int, project_to: int) -> dict[str, Any]:
    rag_rows = [_measure_rag_size(n_docs=size, query=DEFAULT_QUERY, top_k=5, runs=rag_runs) for size in rag_sizes]
    payload["rag_reference"] = {
        "query": DEFAULT_QUERY,
        "top_k": 5,
        "runs": rag_runs,
        "rows": rag_rows,
    }
    payload["projected_8x"] = _projection(payload["matrix"], project_to=project_to)
    payload["selected_optimization"] = {
        "setting": "max_num_batched_tokens=16384",
        "source": "eval/artifacts/experiment_results/EXP-MBT-002/run_001.json",
        "reason": "Sasha's Prime result showed the only global throughput-positive vLLM config among pushed experiments.",
        "rejected_for_final_throughput": [
            "EXP-FP8-001: slower with AWQ",
            "EXP-SCHED-001: better tail consistency but lower throughput",
            "EXP-LENBIN-001: better tail latency but lower throughput",
        ],
    }
    payload["sasha_result_index"] = {
        "summary": "eval/artifacts/experiment_results/PRIME_BENCHMARK_SUMMARY.md",
        "raw_results": {
            "baseline": "eval/artifacts/experiment_results/EXP-MBT-001/run_001.json",
            "best_throughput": "eval/artifacts/experiment_results/EXP-MBT-002/run_001.json",
            "fp8_do_not_use_with_awq": "eval/artifacts/experiment_results/EXP-FP8-001/run_001.json",
            "best_latency_consistency": "eval/artifacts/experiment_results/EXP-SCHED-001/run_001.json",
            "better_tail_latencies": "eval/artifacts/experiment_results/EXP-LENBIN-001/run_001.json",
        },
        "teammate_tldr": [
            "Use max_num_batched_tokens=16384: slight throughput win.",
            "Skip FP8 KV cache with AWQ models: it is about 7% slower in Sasha's run.",
            "For latency-sensitive endpoints, consider 15ms batch accumulation (SCHED-001 pattern).",
            "EXP-BATCH-001 and EXP-OVERLAP-001/002 still need application-layer testing.",
        ],
    }
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Prime Final H100/RAG Matrix",
        "",
        f"- model: `{payload['model']}`",
        f"- vLLM: `{payload['vllm_version']}`",
        f"- selected optimization: `{payload['selected_optimization']['setting']}`",
        f"- source summary: `{payload['sasha_result_index']['summary']}`",
        "",
        "## Actual Prime H100 Results",
        "",
        "| H100s | scenario | req/s | p50 ms | p95 ms | GPU util mean/max | power mean/max W | memory max MB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for count, scenarios in payload["matrix"].items():
        for name, row in scenarios.items():
            lines.append(
                f"| {count} | {name} | {row['requests_per_s']:.3f} | {row['latency_ms_p50']:.3f} | "
                f"{row['latency_ms_p95']:.3f} | "
                f"{(row.get('gpu_utilization_pct_mean') or 0.0):.1f}/{(row.get('gpu_utilization_pct_max') or 0.0):.1f} | "
                f"{(row.get('gpu_power_w_mean') or 0.0):.1f}/{(row.get('gpu_power_w_max') or 0.0):.1f} | "
                f"{(row.get('gpu_memory_used_mb_max') or 0.0):.1f} |"
            )
    if payload.get("projected_8x"):
        lines.extend(
            [
                "",
                "## Projected 8xH100 From Actual 1x/2x",
                "",
                "These rows are projections, not measured 8xH100 results.",
                "",
                "| scenario | observed 2x efficiency | projected 8x req/s linear | projected 8x req/s efficiency-capped | p50 ms assumption | p95 ms assumption |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in payload["projected_8x"].items():
            lines.append(
                f"| {name} | {row['observed_2x_efficiency']:.3f} | "
                f"{row['projected_requests_per_s_linear_from_2x']:.3f} | "
                f"{row['projected_requests_per_s_efficiency_capped']:.3f} | "
                f"{row['projected_latency_ms_p50']:.3f} | {row['projected_latency_ms_p95']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## RAG Reference",
            "",
            "| docs | retrieve p50 ms | fresh-file total ms | retrieve qps |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rag_reference"]["rows"]:
        lines.append(
            f"| {row['n_docs']} | {row['retrieve_ms_p50']:.3f} | "
            f"{row['fresh_file_total_ms']:.3f} | {row['single_process_retrieve_qps_p50']:.3f} |"
        )
    lines.extend(["", "## Teammate TL;DR", ""])
    lines.extend(f"- {item}" for item in payload["sasha_result_index"]["teammate_tldr"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final Prime H100/RAG matrix.")
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--active-counts", default="1,2")
    parser.add_argument("--project-to", type=int, default=8)
    parser.add_argument("--rag-sizes", default="7,100,1000,10000,25000,100000")
    parser.add_argument("--rag-runs", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--pod-id", help="Reuse an existing Prime pod instead of creating one.")
    parser.add_argument("--keep-pod", action="store_true")
    args = parser.parse_args()

    active_counts = [int(part.strip()) for part in args.active_counts.split(",") if part.strip()]
    rag_sizes = [int(part.strip()) for part in args.rag_sizes.split(",") if part.strip()]
    payload = _run_prime_matrix(
        gpu_count=args.gpu_count,
        active_counts=active_counts,
        keep_pod=args.keep_pod,
        pod_id=args.pod_id,
    )
    payload = _with_rag(payload, rag_sizes=rag_sizes, rag_runs=args.rag_runs, project_to=args.project_to)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(_markdown(payload))
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
