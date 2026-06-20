import json
import os
import unittest


class BackendSchemaContractTests(unittest.TestCase):
    def test_schema_constants_match_shared_contract(self):
        from backend.schemas import HIST_BINS, RefineOp

        self.assertEqual(HIST_BINS, 20)
        self.assertEqual({op.value for op in RefineOp}, {"require", "exclude", "include", "refocus", "brush"})

    def test_result_event_serializes_wire_shape(self):
        from backend.schemas import ChunkWireMeta, ResultEvent

        event = ResultEvent(
            chunk_id="chunk-1",
            score=0.91,
            meta=ChunkWireMeta(
                type="code",
                title="urllib3/connectionpool.py",
                category="python",
                year=2023,
                path="src/urllib3/connectionpool.py",
                lang="python",
                repo="urllib3",
            ),
            rank=0,
            rationale=None,
        )

        payload = event.model_dump()

        self.assertEqual(payload["type"], "result")
        self.assertEqual(payload["chunk_id"], "chunk-1")
        self.assertEqual(payload["score"], 0.91)
        self.assertEqual(payload["rank"], 0)
        self.assertIsNone(payload["rationale"])
        self.assertEqual(payload["meta"]["type"], "code")


class BackendApiContractTests(unittest.TestCase):
    def setUp(self):
        os.environ["SCORER_BACKEND"] = "mock"

    def test_healthz_reports_ready_mock_scorer(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        client = TestClient(app)
        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ready"], True)
        self.assertEqual(response.json()["scorer"], "mock")

    def test_ingest_demo_returns_corpus_summary(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        client = TestClient(app)
        response = client.post("/ingest", json={"corpus_id": "demo"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["corpus_id"], "demo")
        self.assertGreater(payload["n_chunks"], 0)
        self.assertIn("type", payload["facets"])
        self.assertIn("category", payload["facets"])
        self.assertIn("year", payload["facets"])
        self.assertIn("warm_eta_s", payload)

    def test_query_stream_contains_result_aggregate_done_events(self):
        from fastapi.testclient import TestClient

        from backend.main import app

        client = TestClient(app)
        client.post("/ingest", json={"corpus_id": "demo"})

        response = client.post(
            "/query",
            json={"predicate": "retry network call without backoff", "threshold": 0.5},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")

        events = []
        for frame in response.text.strip().split("\n\n"):
            self.assertTrue(frame.startswith("data: "))
            events.append(json.loads(frame.removeprefix("data: ")))

        self.assertTrue({"result", "aggregate", "done"}.issubset({event["type"] for event in events}))


if __name__ == "__main__":
    unittest.main()
