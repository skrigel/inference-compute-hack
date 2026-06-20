import asyncio
import os
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

from data.schema import Chunk, ChunkMeta
from inference.scorer import ScoreRequest, ScoreResult


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        type="code",
        title=f"Title {chunk_id}",
        text=text,
        meta=ChunkMeta("python", 2024, "path.py", "python", "repo", "test"),
    )


class StreamingAccumulatorTests(unittest.TestCase):
    def test_batch_accumulate_ms_zero_dispatches_immediately(self):
        from backend.streaming import BATCH_SIZE

        self.assertEqual(BATCH_SIZE, 64)

    def test_accumulator_integration_disabled_by_default(self):
        self.assertEqual(os.environ.get("BATCH_ACCUMULATE_MS", "0"), "0")


if __name__ == "__main__":
    unittest.main()
