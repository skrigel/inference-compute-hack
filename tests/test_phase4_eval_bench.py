import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class Phase4EvalBenchTests(unittest.TestCase):
    def test_quality_gate_writes_artifacts(self):
        from eval import bench

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"SCORER_BACKEND": "mock"}):
            artifact_dir = Path(tmp)
            payload = asyncio.run(bench.run_quality_gate("mock", artifact_dir=artifact_dir))

            self.assertEqual(payload["scorer_backend"], "mock")
            self.assertGreaterEqual(payload["quality"]["f1"], 0.7)
            self.assertTrue((artifact_dir / "phase04_quality_gate.json").exists())
            self.assertTrue((artifact_dir / "phase04_quality_gate.md").exists())

    def test_freeze_writes_contract_trace_jsonl(self):
        from eval import bench

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"SCORER_BACKEND": "mock"}):
            artifact_dir = Path(tmp)
            payload = asyncio.run(bench.run_freeze("mock", artifact_dir=artifact_dir))

            trace_rows = [
                json.loads(line)
                for line in (artifact_dir / "phase04_vllm_trace.jsonl").read_text().splitlines()
                if line.strip()
            ]

            self.assertEqual(payload["scorer_backend"], "mock")
            self.assertGreater(payload["scoped_cumulative_chunks"], 0)
            self.assertGreater(payload["full_cumulative_chunks"], payload["scoped_cumulative_chunks"])
            self.assertGreaterEqual(len(trace_rows), 3)
            for row in trace_rows:
                self.assertIn("run_id", row)
                self.assertIn("rho", row)
                self.assertIn("gpu_cache_usage_perc", row)
                self.assertEqual(row["scorer_backend"], "mock")
                self.assertLessEqual(row["rho"], 1.0)
                self.assertLessEqual(row["survivor_count"], row["candidate_count"])
                self.assertLessEqual(row["candidate_count"], row["n_chunks_total"])
            self.assertTrue((artifact_dir / "phase04_metrics.json").exists())
            self.assertTrue((artifact_dir / "phase04_environment.md").exists())


if __name__ == "__main__":
    unittest.main()
