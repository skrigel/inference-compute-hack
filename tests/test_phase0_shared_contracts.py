import asyncio
import os
import unittest


class SharedContractTests(unittest.TestCase):
    def test_chunk_id_is_stable_and_depends_on_text(self):
        from data.schema import chunk_id_of

        first = chunk_id_of("doc-1", 0, "same text")
        second = chunk_id_of("doc-1", 0, "same text")
        changed = chunk_id_of("doc-1", 0, "different text")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 16)

    def test_mock_scorer_preserves_order_and_score_shape(self):
        from inference.mock_scorer import MockScorer
        from inference.scorer import ScoreRequest

        async def run():
            scorer = MockScorer()
            items = [
                ScoreRequest("c1", "retry without backoff in networking layer", "retry without backoff"),
                ScoreRequest("c2", "unrelated abstract about ranking", "retry without backoff"),
                ScoreRequest("c3", "network retry with exponential backoff", "retry without backoff"),
            ]
            return await scorer.score_batch(items)

        results = asyncio.run(run())

        self.assertEqual([r.chunk_id for r in results], ["c1", "c2", "c3"])
        for result in results:
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
            self.assertGreaterEqual(result.p_yes, 0.0)
            self.assertGreaterEqual(result.p_no, 0.0)
            self.assertEqual(result.tier, 1)
            self.assertFalse(result.from_cache)

    def test_make_scorer_defaults_to_mock_and_rejects_unknown_backend(self):
        from inference.config import make_scorer
        from inference.mock_scorer import MockScorer

        old_value = os.environ.get("SCORER_BACKEND")
        try:
            os.environ.pop("SCORER_BACKEND", None)
            self.assertIsInstance(make_scorer(), MockScorer)

            os.environ["SCORER_BACKEND"] = "not-real"
            with self.assertRaises(ValueError):
                make_scorer()
        finally:
            if old_value is None:
                os.environ.pop("SCORER_BACKEND", None)
            else:
                os.environ["SCORER_BACKEND"] = old_value


if __name__ == "__main__":
    unittest.main()
