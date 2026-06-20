import time
import unittest

from inference.scorer import ScoreRequest


class BatchAccumulatorTests(unittest.TestCase):
    def test_immediate_dispatch_when_disabled(self):
        from backend.batch_accumulator import BatchAccumulator

        acc = BatchAccumulator(max_wait_ms=0, max_batch_size=64)
        req = ScoreRequest("c1", "text", "predicate")
        result = acc.add(req)

        self.assertEqual(result, [req])
        self.assertEqual(acc.pending(), [])

    def test_accumulates_until_max_batch_size(self):
        from backend.batch_accumulator import BatchAccumulator

        acc = BatchAccumulator(max_wait_ms=1000, max_batch_size=2)
        req1 = ScoreRequest("c1", "text1", "predicate")
        req2 = ScoreRequest("c2", "text2", "predicate")

        result1 = acc.add(req1)
        self.assertEqual(result1, [])
        self.assertEqual(acc.pending(), [req1])

        result2 = acc.add(req2)
        self.assertEqual(result2, [req1, req2])
        self.assertEqual(acc.pending(), [])

    def test_flush_returns_pending_requests(self):
        from backend.batch_accumulator import BatchAccumulator

        acc = BatchAccumulator(max_wait_ms=1000, max_batch_size=64)
        req = ScoreRequest("c1", "text", "predicate")
        acc.add(req)

        flushed = acc.flush()
        self.assertEqual(flushed, [req])
        self.assertEqual(acc.pending(), [])

    def test_time_elapsed_triggers_dispatch(self):
        from backend.batch_accumulator import BatchAccumulator

        acc = BatchAccumulator(max_wait_ms=10, max_batch_size=64)
        req = ScoreRequest("c1", "text", "predicate")
        acc.add(req)

        time.sleep(0.015)
        self.assertTrue(acc.should_flush())
        flushed = acc.flush()
        self.assertEqual(flushed, [req])


if __name__ == "__main__":
    unittest.main()
