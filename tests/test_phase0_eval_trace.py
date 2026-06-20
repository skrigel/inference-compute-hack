import json
import subprocess
import sys
import unittest
from unittest.mock import patch


REQUIRED_TRACE_FIELDS = {
    "run_id",
    "commit",
    "corpus_id",
    "model_id",
    "scorer_backend",
    "turn",
    "operation",
    "threshold",
    "n_chunks_total",
    "candidate_count",
    "chunks_scored",
    "chunks_served_from_cache",
    "survivor_count",
    "rho",
    "elapsed_ms",
    "model_ms",
    "queue_ms",
    "ttft_ms",
    "cache_hit_rate",
    "gpu_cache_usage_perc",
    "warm_state",
    "latency_kind",
    "quality_slice",
}


class EvalTraceTests(unittest.TestCase):
    def test_eval_bench_smoke_emits_required_trace_fields(self):
        completed = subprocess.run(
            [sys.executable, "-m", "eval.bench", "--smoke"],
            check=True,
            capture_output=True,
            text=True,
        )

        trace = json.loads(completed.stdout)

        self.assertTrue(REQUIRED_TRACE_FIELDS.issubset(trace.keys()))
        self.assertEqual(trace["scorer_backend"], "mock")
        self.assertEqual(trace["operation"], "query")
        self.assertEqual(trace["ttft_ms"], 0.0)
        self.assertEqual(trace["gpu_cache_usage_perc"], 0.0)
        self.assertIsNone(trace["quality_slice"])
        self.assertGreater(trace["n_chunks_total"], 0)
        self.assertGreater(trace["chunks_scored"], 0)
        self.assertGreaterEqual(trace["chunks_served_from_cache"], 0)
        self.assertGreaterEqual(trace["cache_hit_rate"], 0.0)
        self.assertLessEqual(trace["cache_hit_rate"], 1.0)

    def test_eval_bench_backend_label_follows_environment(self):
        from eval.bench import _active_backend

        with patch.dict("os.environ", {"SCORER_BACKEND": "modal"}):
            self.assertEqual(_active_backend(), "modal")


if __name__ == "__main__":
    unittest.main()
