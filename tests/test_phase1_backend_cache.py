import asyncio
import unittest

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.scorer import PrefixState, ScoreRequest, ScoreResult, ScorerClient


def chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id_of(doc_id, 0, text),
        doc_id=doc_id,
        type="code",
        title=doc_id,
        text=text,
        meta=ChunkMeta("python", 2024, f"{doc_id}.py", "python", "demo", "synthetic"),
    )


class CountingScorer(ScorerClient):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def warm(self, corpus_id: str, chunks: list[Chunk]) -> PrefixState:
        return PrefixState(corpus_id, len(chunks), True, self.model_id())

    async def score_batch(self, items: list[ScoreRequest], *, tier: int = 1) -> list[ScoreResult]:
        self.batch_sizes.append(len(items))
        return [
            ScoreResult(
                chunk_id=item.chunk_id,
                score=0.9 if "retry" in item.chunk_text else 0.1,
                p_yes=0.9 if "retry" in item.chunk_text else 0.1,
                p_no=0.1 if "retry" in item.chunk_text else 0.9,
                tier=tier,
            )
            for item in items
        ]

    async def health(self) -> dict:
        return {"ready": True, "backend": "counting"}

    def model_id(self) -> str:
        return "counting-scorer"


async def collect_stream(*, scorer: CountingScorer, cache, chunks: list[Chunk], clause_id: str) -> list:
    from backend.streaming import query_stream

    events = []
    async for event in query_stream(
        scorer,
        chunks,
        "retry",
        clause_id=clause_id,
        threshold=0.5,
        cache=cache,
        batch_size=2,
    ):
        events.append(event)
    return events


class Phase1BackendCacheTests(unittest.TestCase):
    def test_cold_scan_records_cache_misses_not_fresh_write_hits(self):
        from backend.cache import ScoreCache

        cache = ScoreCache()
        scorer = CountingScorer()
        chunks = [chunk("a", "retry without backoff"), chunk("b", "ranking metrics")]

        events = asyncio.run(collect_stream(scorer=scorer, cache=cache, chunks=chunks, clause_id="q1"))

        self.assertEqual(scorer.batch_sizes, [2])
        self.assertEqual(cache.stats()["n_entries"], 2)
        self.assertEqual(cache.hit_rate(), 0.0)
        self.assertEqual(events[-1].matched, 1)

    def test_warm_same_clause_serves_all_candidates_from_cache(self):
        from backend.cache import ScoreCache

        cache = ScoreCache()
        scorer = CountingScorer()
        chunks = [chunk("a", "retry without backoff"), chunk("b", "ranking metrics")]

        asyncio.run(collect_stream(scorer=scorer, cache=cache, chunks=chunks, clause_id="q1"))
        scorer.batch_sizes.clear()
        asyncio.run(collect_stream(scorer=scorer, cache=cache, chunks=chunks, clause_id="q1"))

        self.assertEqual(scorer.batch_sizes, [])
        self.assertEqual(cache.hit_rate(), 0.5)


if __name__ == "__main__":
    unittest.main()
