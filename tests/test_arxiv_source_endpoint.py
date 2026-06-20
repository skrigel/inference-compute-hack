import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.schemas import FreshDocument


class ArxivSourceEndpointTest(unittest.TestCase):
    def test_arxiv_source_fetches_and_appends_documents(self):
        main.state.chunks = []
        main.cache.clear()
        docs = [
            FreshDocument(
                title="Agentic Query Refinement",
                text="A paper about agentic retrieval ranking metrics.",
                type="paper",
                category="cs.IR",
                year=2026,
                path="https://arxiv.org/abs/2601.00001",
                repo="arxiv",
            )
        ]

        with patch("backend.main.fetch_arxiv_documents", return_value=docs) as fetch:
            client = TestClient(main.app)
            response = client.post("/source/arxiv", json={"query": "retrieval ranking metrics", "count": 1})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["corpus_id"], "arxiv")
        self.assertGreaterEqual(payload["n_chunks"], 1)
        self.assertEqual(payload["facets"]["type"][0]["key"], "paper")
        fetch.assert_called_once_with("retrieval ranking metrics", max_results=1)


if __name__ == "__main__":
    unittest.main()
