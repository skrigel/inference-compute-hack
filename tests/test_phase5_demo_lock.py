"""Phase 05 demo-lock guard: every demo beat has a contract-shaped canned twin,
and the replay server serves them (incl. the fresh-file toggle for beat 5)."""
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _types(frames: list[str]) -> list[str]:
    return [json.loads(f.removeprefix("data: "))["type"] for f in frames if f.strip()]


class DemoLockTests(unittest.TestCase):
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

    def test_locked_docs_exist(self):
        self.assertTrue((REPO / "DEMO.md").exists())
        slide = (REPO / "eval" / "SLIDE.md").read_text()
        # The deck must carry honest labels and the pending gate, not bare numbers.
        self.assertIn("measured-mock", slide)
        self.assertIn("pending", slide.lower())

    def test_every_beat_has_a_contract_shaped_canned_twin(self):
        from scripts.replay_sse import load_frames, record

        with tempfile.TemporaryDirectory() as tmp:
            paths = record(Path(tmp))
            query = load_frames(paths["query"])
            refine = load_frames(paths["refine"])
            fresh = load_frames(paths["fresh"])

            # Beat 1: opening query streams result … aggregate … done.
            self.assertEqual(_types(query)[0], "result")
            self.assertIn("aggregate", _types(query))
            self.assertEqual(_types(query)[-1], "done")
            # Beats 2-3: refine is chip-first then done.
            self.assertEqual(_types(refine)[0], "chip")
            self.assertEqual(_types(refine)[-1], "done")
            # Beat 5: the fresh-file query surfaces the dropped document.
            fresh_titles = [
                json.loads(f.removeprefix("data: "))["meta"]["title"]
                for f in fresh
                if f.strip() and json.loads(f.removeprefix("data: "))["type"] == "result"
            ]
            self.assertIn("fresh_incident.py", fresh_titles)

    def test_replay_server_serves_each_beat_and_arms_the_fresh_toggle(self):
        from fastapi.testclient import TestClient

        from scripts.replay_sse import build_replay_app, record

        with tempfile.TemporaryDirectory() as tmp:
            record(Path(tmp))
            client = TestClient(build_replay_app(Path(tmp)))

            self.assertEqual(client.get("/healthz").json()["scorer"], "replay")

            # Before any fresh drop, /query replays the opening query (no fresh chunk).
            opening = client.post("/query", json={"predicate": "x", "threshold": 0.5}).text
            self.assertNotIn("fresh_incident.py", opening)

            # Drop a fresh file → the next /query replays the fresh fixture.
            client.post("/ingest", json={"corpus_id": "demo", "documents": [{"title": "f.py", "text": "t"}]})
            after = client.post("/query", json={"predicate": "x", "threshold": 0.5}).text
            self.assertIn("fresh_incident.py", after)

            refine = client.post("/refine", json={"utterance": "y"}).text
            self.assertEqual(_types([f for f in refine.strip().split("\n\n") if f.strip()])[0], "chip")


if __name__ == "__main__":
    unittest.main()
