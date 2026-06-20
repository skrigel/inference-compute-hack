"""Canned SSE replay — the bottom rung of the demo fallback ladder.

`record` captures real `/query` and `/refine` SSE from the (mock or live) backend
into byte-for-byte fixtures. `serve` replays them on the same endpoints so the
frontend can point `VITE_API_BASE` at the replay server and not know the
difference. This is the "never nothing on stage" guarantee: if the live path
stalls, flip to the replay server and the UI stays real and interactive.

    python -m scripts.replay_sse record               # capture fixtures from mock backend
    python -m scripts.replay_sse serve --port 8090    # serve them

The recorded frames are emitted verbatim, so the replay output is identical to
what the backend produced when recorded.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.schemas import IngestRequest, QueryRequest, RefineRequest

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "eval" / "artifacts"
QUERY_FIXTURE = "cut_line_query.sse"
REFINE_FIXTURE = "cut_line_refine.sse"

DEMO_QUERY = "every place we retry a network call without backoff"


def load_frames(path: Path) -> list[str]:
    """Return the raw `data: ...` frames (without the trailing blank line)."""
    text = path.read_text().strip()
    return [frame for frame in text.split("\n\n") if frame.strip()]


def _reset_mock_backend():
    import backend.main as main
    from backend.cache import ScoreCache
    from backend.state import BackendState
    from inference.mock_scorer import MockScorer

    main.state = BackendState()
    main.cache = ScoreCache()
    main.scorer = MockScorer()
    main._clause_seq = itertools.count(1)
    return main


def record(out_dir: Path = ARTIFACT_DIR) -> dict[str, Path]:
    """Capture /query and /refine SSE from the in-process mock backend."""
    from fastapi.testclient import TestClient

    main = _reset_mock_backend()
    client = TestClient(main.app)
    client.post("/ingest", json={"corpus_id": "demo"})

    query_text = client.post("/query", json={"predicate": DEMO_QUERY, "threshold": 0.5}).text
    refine_text = client.post("/refine", json={"utterance": "only in the networking layer"}).text

    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / QUERY_FIXTURE
    refine_path = out_dir / REFINE_FIXTURE
    query_path.write_text(query_text)
    refine_path.write_text(refine_text)
    return {"query": query_path, "refine": refine_path}


def build_replay_app(fixtures_dir: Path = ARTIFACT_DIR) -> FastAPI:
    app = FastAPI(title="Grep-for-Meaning SSE replay", version="0.3.0")
    # The replay server is pointed at DIRECTLY via VITE_API_BASE (cross-origin
    # from the Vite dev server), so it must send permissive CORS headers or the
    # browser blocks the fallback. SSE needs no special CORS beyond this.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def _stream(path: Path) -> StreamingResponse:
        frames = load_frames(path)

        async def gen() -> AsyncIterator[str]:
            for frame in frames:
                yield frame + "\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ready": True, "scorer": "replay", "warmed": True}

    @app.post("/ingest")
    async def ingest(request: IngestRequest) -> dict:
        # Canned ingest: the replay corpus is whatever the fixtures encode. Body
        # accepted (and echoed) to conform to the frozen /ingest interface.
        return {"corpus_id": request.corpus_id, "n_chunks": 7, "facets": {}, "warm_eta_s": 0.0}

    @app.post("/query")
    async def query(request: QueryRequest) -> StreamingResponse:
        return _stream(fixtures_dir / QUERY_FIXTURE)

    @app.post("/refine")
    async def refine(request: RefineRequest) -> StreamingResponse:
        return _stream(fixtures_dir / REFINE_FIXTURE)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Canned SSE replay (fallback ladder)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record", help="capture fixtures from the mock backend")
    serve = sub.add_parser("serve", help="serve recorded fixtures")
    serve.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if args.cmd == "record":
        paths = record()
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return

    import uvicorn

    uvicorn.run(build_replay_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
