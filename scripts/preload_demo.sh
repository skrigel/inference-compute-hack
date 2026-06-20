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

# 2) Arm the canned-SSE replay fallback (records query + refine + fresh fixtures).
echo "[2/3] recording canned SSE fixtures (replay fallback)…"
if "$PYTHON" -m scripts.replay_sse record >/dev/null; then
  replay_ok=1
else
  replay_ok=0
fi
echo

# 3) Verify every demo-lock artifact a beat depends on exists.
echo "[3/3] verifying demo-lock artifacts…"
artifacts_ok=1
for f in \
  eval/artifacts/cut_line_trace.json \
  eval/artifacts/area_under_loop.png \
  eval/artifacts/cut_line_query.sse \
  eval/artifacts/cut_line_refine.sse \
  eval/artifacts/cut_line_fresh.sse \
  eval/SLIDE.md \
  DEMO.md; do
  if [[ -s "$f" ]]; then echo "  ok   $f"; else echo "  MISS $f"; artifacts_ok=0; fi
done

echo
echo "── result ──────────────────────────────────────────────────────"
if [[ "$loop_ok" == "1" && "$replay_ok" == "1" && "$artifacts_ok" == "1" ]]; then
  echo "GO ✓  loop green · replay armed · demo-lock artifacts present."
  echo "      script:   DEMO.md  (spoken beats, ≤90s budget, operator commands)"
  echo "      eval slide: eval/SLIDE.md  (frozen figures + honest labels)"
  echo "      beats: 1 query · 2 click-NOT · 3 AND · 4 threshold(client) · 5 fresh-file · 6 perf-close"
  echo "      replay:   python -m scripts.replay_sse serve --port 8090  (point VITE_API_BASE at it)"
  exit 0
fi
echo "NO-GO ✗  loop_ok=$loop_ok replay_ok=$replay_ok artifacts_ok=$artifacts_ok — fix before demoing."
exit 1
