import unittest


class ChunkerOverlapTests(unittest.TestCase):
    def test_no_overlap_returns_contiguous_chunks(self):
        from data.chunker import chunk_with_overlap

        text = "word " * 100  # 100 words
        chunks = chunk_with_overlap(text, chunk_size=50, overlap_ratio=0.0)

        self.assertEqual(len(chunks), 2)
        self.assertNotIn(chunks[0][-10:], chunks[1][:10])

    def test_overlap_creates_overlapping_chunks(self):
        from data.chunker import chunk_with_overlap

        text = "word " * 100
        chunks = chunk_with_overlap(text, chunk_size=50, overlap_ratio=0.2)

        self.assertGreater(len(chunks), 2)
        overlap_words = int(50 * 0.2)
        first_end = chunks[0].split()[-overlap_words:]
        second_start = chunks[1].split()[:overlap_words]
        self.assertEqual(first_end, second_start)

    def test_overlap_ratio_increases_chunk_count(self):
        from data.chunker import chunk_with_overlap

        text = "word " * 1000
        chunks_0 = chunk_with_overlap(text, chunk_size=100, overlap_ratio=0.0)
        chunks_10 = chunk_with_overlap(text, chunk_size=100, overlap_ratio=0.1)
        chunks_20 = chunk_with_overlap(text, chunk_size=100, overlap_ratio=0.2)

        self.assertLess(len(chunks_0), len(chunks_10))
        self.assertLess(len(chunks_10), len(chunks_20))

    def test_small_text_returns_single_chunk(self):
        from data.chunker import chunk_with_overlap

        text = "small text"
        chunks = chunk_with_overlap(text, chunk_size=100, overlap_ratio=0.2)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].strip(), text)

    def test_env_var_controls_default_overlap(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"CHUNK_OVERLAP_RATIO": "0.15"}):
            from importlib import reload
            import data.chunker as chunker_module
            reload(chunker_module)

            self.assertEqual(chunker_module.CHUNK_OVERLAP_RATIO, 0.15)


if __name__ == "__main__":
    unittest.main()
