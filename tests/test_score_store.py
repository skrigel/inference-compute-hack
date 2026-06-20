"""Tests for the persistent SQLite score cache (backend/score_store.py) and its
read-through wiring into query_stream — a repeated query fetches stored scores
instead of re-scanning on the scorer."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient

from backend.cache import ScoreCache
from backend.score_store import ScoreStore
from backend.streaming import query_stream


def _chunk(doc: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id_of(doc, 0, text),
        doc_id=doc,
        type="code",
        title=doc,
        text=text,
        meta=ChunkMeta(category=None, year=2024, path=doc, lang="py", repo="r", source=None),
    )


class CountingScorer(ScorerClient):
    """Scores everything 0.8 and counts how many chunks it was asked to score."""

    def __init__(self) -> None:
        self.scored = 0

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        return PrefixState(corpus_id, len(chunks), True, self.model_id())

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        self.scored += len(items)
        return [ScoreResult(i.chunk_id, 0.8, 0.8, 0.2, tier=tier) for i in items]

    async def health(self) -> dict:
        return {"ready": True, "backend": "counting"}

    def model_id(self) -> str:
        return "counting"


class ScoreStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ScoreStore(str(Path(self.tmp.name) / "t.db"))

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_put_get_roundtrip_marks_from_cache(self):
        r = ScoreResult("c1", 0.42, 0.42, 0.58, tier=1)
        self.store.put_scores("demo", "is python?", "m", [r])
        got = self.store.get_scores("demo", "is python?", "m", ["c1"])
        self.assertIn("c1", got)
        self.assertAlmostEqual(got["c1"].score, 0.42)
        self.assertTrue(got["c1"].from_cache)

    def test_isolation_by_collection_predicate_model(self):
        self.store.put_scores("demo", "p", "m", [ScoreResult("c1", 0.9, 0.9, 0.1)])
        self.assertEqual(self.store.get_scores("browsecomp", "p", "m", ["c1"]), {})  # other collection
        self.assertEqual(self.store.get_scores("demo", "other", "m", ["c1"]), {})    # other predicate
        self.assertEqual(self.store.get_scores("demo", "p", "m2", ["c1"]), {})       # other model

    def test_query_stream_second_scan_fetches_instead_of_rescoring(self):
        chunks = [_chunk("d1", "retry with backoff"), _chunk("d2", "unrelated text")]
        scorer = CountingScorer()

        async def scan() -> int:
            # Fresh in-memory cache each run (simulates a new process/session);
            # the SQLite store persists across them.
            cache = ScoreCache()
            async for _ in query_stream(
                scorer, chunks, "about retries", clause_id="q1", threshold=0.5,
                cache=cache, store=self.store, collection="demo", scorer_tag="counting",
            ):
                pass
            return scorer.scored

        first = asyncio.run(scan())
        second = asyncio.run(scan())
        self.assertEqual(first, 2)         # cold: both chunks scored
        self.assertEqual(second, 2)        # warm: scorer NOT called again (still 2 total)
        self.assertEqual(self.store.stats()["n_scores"], 2)


if __name__ == "__main__":
    unittest.main()
