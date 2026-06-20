#!/usr/bin/env bash
# Phase 03 cut-line preflight — run before walking on stage.
# Exercises the full irreducible loop end-to-end, arms the canned-SSE fallback,
# and prints a single GO / NO-GO line. Honors $PYTHON (default: python3) and
# $SCORER_BACKEND (default: mock).
set -euo pipefail

PYTHON="${PYTHON:-python3}"
export SCORER_BACKEND="${SCORER_BACKEND:-mock}"
cd "$(dirname "$0")/.."

echo "── cut-line preflight ──────────────────────────────────────────"
echo "python=$($PYTHON --version 2>&1)  scorer_backend=$SCORER_BACKEND"
echo

# 1) Drive the irreducible loop (ingest → query → click-NOT → AND → threshold → fresh-file).
echo "[1/2] running cut-line loop (eval.cut_line)…"
if "$PYTHON" -m eval.cut_line --figure; then
  loop_ok=1
else
  loop_ok=0
fi
echo

# 2) Arm the canned-SSE replay fallback.
echo "[2/2] recording canned SSE fixtures (replay fallback)…"
if "$PYTHON" -m scripts.replay_sse record >/dev/null; then
  replay_ok=1
else
  replay_ok=0
fi

echo
echo "── result ──────────────────────────────────────────────────────"
if [[ "$loop_ok" == "1" && "$replay_ok" == "1" ]]; then
  echo "GO ✓  loop is green and the canned-SSE fallback is armed."
  echo "      trace:  eval/artifacts/cut_line_trace.json"
  echo "      figure: eval/artifacts/area_under_loop.png"
  echo "      replay: python -m scripts.replay_sse serve --port 8090  (point VITE_API_BASE at it)"
  exit 0
fi
echo "NO-GO ✗  loop_ok=$loop_ok replay_ok=$replay_ok — fix before demoing."
exit 1
