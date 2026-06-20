import json
import tempfile
import unittest
from pathlib import Path


class RagCompareTests(unittest.TestCase):
    def test_rag_compare_writes_json_and_markdown(self):
        from eval.rag_compare import run_rag_vs_6xh100

        reference = {
            "run_id": "modal-openai-server-test",
            "replicas": 6,
            "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
            "vllm_version": "0.22.1",
            "gpu_memory_utilization": 0.92,
            "aggregate_client": {
                "requests_per_s": 100.0,
                "total_tokens_per_s": 1000.0,
                "latency_ms_p50_mean": 10.0,
                "latency_ms_p95_max": 20.0,
            },
            "aggregate_server": {
                "derived_mfu_bf16_peak_mean": 0.1,
                "estimated_tflops_per_gpu_mean": 99.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            modal_artifact = tmp_path / "modal.json"
            output_json = tmp_path / "rag.json"
            output_md = tmp_path / "rag.md"
            modal_artifact.write_text(json.dumps(reference))

            payload = run_rag_vs_6xh100(
                modal_artifact=modal_artifact,
                output_json=output_json,
                output_md=output_md,
                sizes=[7],
                runs=2,
            )

            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())
            self.assertEqual(payload["six_h100_reference"]["replicas"], 6)
            self.assertEqual(len(payload["rag_rows"]), 1)
            row = payload["rag_rows"][0]
            self.assertEqual(row["n_docs"], 7)
            self.assertGreater(row["retrieve_ms_p50"], 0)
            self.assertGreater(row["single_process_retrieve_qps_p50"], 0)
            self.assertIn("RAG backend", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
