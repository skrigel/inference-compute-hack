import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

# The full RAG-vs-6xH100 comparison loads the BrowseComp corpus, which pulls the
# optional HuggingFace `datasets` dependency. Skip it gracefully when absent.
_HAS_DATASETS = importlib.util.find_spec("datasets") is not None


class RagCompareTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_DATASETS, "requires optional 'datasets' dependency")
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
            self.assertIn("quality_comparison", payload)
            self.assertIn("rag", payload["quality_comparison"])
            self.assertIn("recall", payload["quality_comparison"]["rag"]["quality"])
            self.assertIn("optimization_findings", payload)
            self.assertGreaterEqual(len(payload["optimization_findings"]), 1)
            self.assertIn("RAG backend", output_md.read_text())

    def test_rag_quality_metrics_use_gold_predicates(self):
        from eval.rag_compare import evaluate_rag_quality

        quality = evaluate_rag_quality(top_k=5)

        self.assertEqual(quality["corpus_size"], 7)
        self.assertEqual(quality["top_k"], 5)
        self.assertGreaterEqual(quality["quality"]["recall"], 0.0)
        self.assertLessEqual(quality["quality"]["recall"], 1.0)
        self.assertEqual(len(quality["rows"]), 3)
        for row in quality["rows"]:
            self.assertIn("predicate", row)
            self.assertIn("recall_at_k", row)


if __name__ == "__main__":
    unittest.main()
