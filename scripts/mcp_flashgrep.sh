#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$ROOT/.venv"
  else
    python3 -m venv "$ROOT/.venv"
  fi
fi

if ! "$PY" -c "import fastapi, mcp, pydantic, datasets" >/dev/null 2>&1; then
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -r "$ROOT/backend/requirements.txt" "datasets>=2.20.0" >/dev/null
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m backend.mcp_server
