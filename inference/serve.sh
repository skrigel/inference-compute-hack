#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8001}"
N_REPLICAS="${N_REPLICAS:-6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

pids=()
urls=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

for i in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((BASE_PORT + i))
  urls+=("http://${HOST}:${port}/v1")
  CUDA_VISIBLE_DEVICES="$i" python -m vllm.entrypoints.openai.api_server \
    --host "$HOST" \
    --port "$port" \
    --model "$MODEL" \
    --served-model-name tier1-filter \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-mfu-metrics \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    $EXTRA_VLLM_ARGS &
  pids+=("$!")
done

printf 'export VLLM_REPLICAS=%s\n' "$(IFS=,; echo "${urls[*]}")" | tee .vllm_replicas.env
wait
