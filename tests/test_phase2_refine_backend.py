import itertools
import json
import unittest

from data.schema import Chunk
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient


def sse_events(response) -> list[dict]:
    events = []
    for frame in response.text.strip().split("\n\n"):
        if not frame:
            continue
        assert frame.startswith("data: "), frame
        events.append(json.loads(frame.removeprefix("data: ")))
    return events


class ScriptedScorer(ScorerClient):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        return PrefixState(corpus_id, len(chunks), True, self.model_id())

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        self.batch_sizes.append(len(items))
        results = []
        for item in items:
            text = item.chunk_text.lower()
            predicate = item.predicate.lower()
            score = 0.12
            if "retry" in predicate and ("retry" in text or "retries" in text):
                score = 0.9
            if "networking" in predicate and ("network" in text or "http" in text):
                score = 0.86
            if "sentinel" in predicate and "sentinel" in text:
                score = 0.97
            results.append(
                ScoreResult(
                    chunk_id=item.chunk_id,
                    score=score,
                    p_yes=score,
                    p_no=1.0 - score,
                    tier=tier,
                )
            )
        return results

    async def health(self) -> dict:
        return {"ready": True, "backend": "scripted"}

    def model_id(self) -> str:
        return "scripted-scorer"


def reset_backend(scorer: ScriptedScorer | None = None):
    import backend.main as main
    from backend.cache import ScoreCache
    from backend.state import BackendState

    main.state = BackendState()
    main.cache = ScoreCache()
    main.scorer = scorer or ScriptedScorer()
    main._clause_seq = itertools.count(1)
    return main


class Phase2RefineBackendTests(unittest.TestCase):
    def tearDown(self):
        import backend.main as main
        from backend.cache import ScoreCache
        from backend.state import BackendState
        from inference.mock_scorer import MockScorer

        main.state = BackendState()
        main.cache = ScoreCache()
        main.scorer = MockScorer()
        main._clause_seq = itertools.count(1)

    def test_refine_stream_starts_with_chip_and_scores_only_current_survivors(self):
        from fastapi.testclient import TestClient

        scorer = ScriptedScorer()
        main = reset_backend(scorer)
        client = TestClient(main.app)

        client.post("/ingest", json={"corpus_id": "demo"})
        query_response = client.post("/query", json={"predicate": "retry", "threshold": 0.5})
        self.assertEqual(query_response.status_code, 200)
        self.assertEqual(scorer.batch_sizes, [5])

        refine_response = client.post("/refine", json={"utterance": "only networking layer"})

        self.assertEqual(refine_response.status_code, 200)
        events = sse_events(refine_response)
        self.assertEqual([event["type"] for event in events], ["chip", "diff", "aggregate", "done"])
        self.assertEqual(events[0]["operation"], "require")
        self.assertEqual(events[0]["chip"]["op"], "require")
        self.assertEqual(events[1]["refine_ms"], events[0]["refine_ms"])
        self.assertEqual(scorer.batch_sizes, [5, 2])

    def test_click_drop_and_clause_delete_are_zero_inference_cache_recomputes(self):
        from fastapi.testclient import TestClient

        scorer = ScriptedScorer()
        main = reset_backend(scorer)
        client = TestClient(main.app)

        client.post("/ingest", json={"corpus_id": "demo"})
        query_events = sse_events(client.post("/query", json={"predicate": "retry", "threshold": 0.5}))
        target = next(event["chunk_id"] for event in query_events if event["type"] == "result" and event["score"] >= 0.5)
        scorer.batch_sizes.clear()

        refine_events = sse_events(client.post("/refine", json={"click": {"chunk_id": target, "sign": "-"}}))

        self.assertEqual(refine_events[0]["type"], "chip")
        self.assertEqual(refine_events[0]["operation"], "exclude")
        self.assertIn(target, refine_events[1]["removed"])
        self.assertEqual(scorer.batch_sizes, [])

        clause_id = refine_events[0]["chip"]["clause_id"]
        delete_response = client.delete(f"/clause/{clause_id}")

        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json()["removed"], True)
        self.assertEqual(scorer.batch_sizes, [])
        results = client.get("/results", params={"threshold": 0.5}).json()["items"]
        self.assertIn(target, [item["chunk_id"] for item in results])

    def test_fresh_document_ingest_appends_queryable_chunk(self):
        from fastapi.testclient import TestClient

        main = reset_backend()
        client = TestClient(main.app)

        response = client.post(
            "/ingest",
            json={
                "corpus_id": "demo",
                "documents": [
                    {
                        "title": "fresh_retry.py",
                        "text": "fresh sentinel retry example without backoff",
                        "type": "code",
                        "category": "python",
                        "path": "fresh_retry.py",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["n_chunks"], 6)
        events = sse_events(client.post("/query", json={"predicate": "sentinel", "threshold": 0.5}))
        titles = [event["meta"]["title"] for event in events if event["type"] == "result"]
        self.assertIn("fresh_retry.py", titles)


if __name__ == "__main__":
    unittest.main()
