import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class StandardBenchmarkTests(unittest.TestCase):
    def test_standard_benchmark_writes_comparison_artifacts(self):
        from eval import standard_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            candidate = tmp_path / "candidate.json"
            baseline.write_text(json.dumps(self._matrix(qps=100.0, p50=10.0)))
            candidate.write_text(json.dumps(self._matrix(qps=120.0, p50=9.0)))

            with (
                patch.object(standard_benchmark, "EXPERIMENT_ROOT", tmp_path / "experiment_results"),
                patch.object(standard_benchmark, "SUMMARY_ROOT", tmp_path / "experiment_summaries"),
            ):
                result = standard_benchmark.run_standard_benchmark(
                    opt_id="OPT-TEST",
                    name="test optimization",
                    baseline_artifacts=[baseline],
                    candidate_artifacts=[candidate],
                    dataset_sizes=[7],
                    skip_rag=True,
                    command="python -m eval.standard_benchmark --opt-id OPT-TEST",
                )

            paths = result["paths"]
            self.assertTrue((tmp_path / "experiment_results" / "OPT-TEST" / "config.json").exists())
            self.assertTrue((tmp_path / "experiment_results" / "OPT-TEST" / "aggregated.json").exists())
            self.assertTrue((tmp_path / "experiment_results" / "OPT-TEST" / "ledger_entry.md").exists())
            self.assertTrue((tmp_path / "experiment_summaries" / "OPT-TEST_summary.md").exists())
            self.assertIn("aggregated", paths)

            comparisons = result["aggregated"]["comparisons"]
            qps = next(row for row in comparisons if row["metric"] == "requests_per_s")
            p50 = next(row for row in comparisons if row["metric"] == "latency_ms_p50")
            self.assertEqual(qps["verdict"], "improved")
            self.assertEqual(p50["verdict"], "improved")

    def test_parse_sizes_preserves_standard_and_adds_extreme(self):
        from eval.standard_benchmark import _parse_sizes

        self.assertEqual(_parse_sizes([100, 7, 100], include_extreme=False), [7, 100])
        self.assertIn(250_000, _parse_sizes([7], include_extreme=True))

    def _matrix(self, *, qps: float, p50: float) -> dict:
        return {
            "run_id": "matrix-test",
            "model": "model",
            "vllm_version": "0.22.1",
            "gpu_counts": [1],
            "prompt_variant": "compact",
            "gpu_memory_utilization": 0.92,
            "max_num_batched_tokens": 8192,
            "scenarios": [
                {
                    "name": "single_user_static",
                    "dataset_mode": "static",
                    "num_requests": 32,
                    "concurrency": 1,
                    "rag_latency_metric": "retrieve_ms_p50",
                }
            ],
            "h100_results": {
                "single_user_static": {
                    "1": {
                        "aggregate_client": {
                            "requests_per_s": qps,
                            "prompt_tokens_per_s": qps * 10,
                            "total_tokens_per_s": qps * 11,
                            "latency_ms_p50_mean": p50,
                            "latency_ms_p95_max": p50 * 2,
                            "latency_ms_p99_max": p50 * 3,
                        },
                        "aggregate_server": {
                            "derived_mfu_bf16_peak_mean": 0.01,
                            "estimated_tflops_per_gpu_mean": 10.0,
                            "gpu_utilization_pct_mean": 20.0,
                            "gpu_utilization_pct_max": 30.0,
                            "gpu_memory_used_mb_max": 70_000.0,
                            "gpu_memory_utilization_pct_max": 90.0,
                            "gpu_power_w_mean": 120.0,
                            "gpu_power_w_max": 150.0,
                            "kv_cache_usage_perc_max": 0.0,
                            "server_queue_avg_ms_mean": 1.0,
                            "server_prefill_avg_ms_mean": 8.0,
                            "server_ttft_avg_ms_mean": 10.0,
                        },
                    }
                }
            },
            "rag_reference": {"backend": "numpy-fallback", "query": "q", "rows": []},
            "comparisons": [],
            "refinement_overlap": [],
        }


if __name__ == "__main__":
    unittest.main()
