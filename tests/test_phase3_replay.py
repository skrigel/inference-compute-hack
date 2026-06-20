"""Phase 03 fallback-ladder test: record canned SSE, then prove the replay
server serves it byte-for-byte so the frontend can fail over to it."""
import json
import tempfile
import unittest
from pathlib import Path


class ReplayLadderTests(unittest.TestCase):
    def tearDown(self):
        import itertools

        import backend.main as main
        from backend.cache import ScoreCache
        from backend.state import BackendState
        from inference.mock_scorer import MockScorer

        main.state = BackendState()
        main.cache = ScoreCache()
        main.scorer = MockScorer()
        main._clause_seq = itertools.count(1)

    def test_recorded_fixtures_replay_identically(self):
        from fastapi.testclient import TestClient

        from scripts.replay_sse import build_replay_app, load_frames, record

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = record(out)
            recorded_query = load_frames(paths["query"])
            recorded_refine = load_frames(paths["refine"])

            # The recordings themselves must be contract-shaped.
            self.assertEqual(json.loads(recorded_query[0].removeprefix("data: "))["type"], "result")
            self.assertEqual(json.loads(recorded_refine[0].removeprefix("data: "))["type"], "chip")

            client = TestClient(build_replay_app(out))
            self.assertEqual(client.get("/healthz").json()["scorer"], "replay")

            served_query = client.post("/query", json={"predicate": "x"}).text.strip().split("\n\n")
            served_refine = client.post("/refine", json={"utterance": "y"}).text.strip().split("\n\n")

            self.assertEqual([f for f in served_query if f.strip()], recorded_query)
            self.assertEqual([f for f in served_refine if f.strip()], recorded_refine)

            # Replay refine is chip-first and ends with done — the frozen order.
            served_types = [json.loads(f.removeprefix("data: "))["type"] for f in served_refine if f.strip()]
            self.assertEqual(served_types[0], "chip")
            self.assertEqual(served_types[-1], "done")


if __name__ == "__main__":
    unittest.main()
