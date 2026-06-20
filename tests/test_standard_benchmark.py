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

    def test_warmup_exclusion_marks_and_excludes_first_matrix_artifact(self):
        from eval import standard_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_paths = []
            candidate_paths = []
            for idx, qps in enumerate([10.0, 100.0, 102.0], start=1):
                path = tmp_path / f"baseline_{idx}.json"
                path.write_text(json.dumps(self._matrix(qps=qps, p50=10.0)))
                baseline_paths.append(path)
            for idx, qps in enumerate([5.0, 120.0, 122.0], start=1):
                path = tmp_path / f"candidate_{idx}.json"
                path.write_text(json.dumps(self._matrix(qps=qps, p50=9.0)))
                candidate_paths.append(path)

            with (
                patch.object(standard_benchmark, "EXPERIMENT_ROOT", tmp_path / "experiment_results"),
                patch.object(standard_benchmark, "SUMMARY_ROOT", tmp_path / "experiment_summaries"),
            ):
                result = standard_benchmark.run_standard_benchmark(
                    opt_id="OPT-WARMUP",
                    name="warmup exclusion",
                    baseline_artifacts=baseline_paths,
                    candidate_artifacts=candidate_paths,
                    dataset_sizes=[7],
                    skip_rag=True,
                    warmup_excluded=True,
                )

            baseline = result["aggregated"]["baseline"]
            candidate = result["aggregated"]["candidate"]
            self.assertEqual(baseline["artifact_count"], 3)
            self.assertEqual(baseline["run_count"], 2)
            self.assertEqual(baseline["warmup_excluded_count"], 1)
            self.assertEqual(candidate["run_count"], 2)

            workload = "single_user_static|gpu=1|concurrency=1|requests=32"
            self.assertAlmostEqual(
                baseline["by_workload"][workload]["metrics"]["requests_per_s"]["mean"],
                101.0,
            )
            self.assertAlmostEqual(
                candidate["by_workload"][workload]["metrics"]["requests_per_s"]["mean"],
                121.0,
            )

            warmup_run = json.loads(
                (tmp_path / "experiment_results" / "OPT-WARMUP" / "runs" / "baseline_matrix_run_001.json").read_text()
            )
            self.assertTrue(warmup_run["is_warmup"])

    def test_modal_matrix_runs_generate_multiple_candidate_artifacts(self):
        from eval import standard_benchmark

        captured = {}

        def fake_run_modal_matrix(args, artifact_prefix):
            path = Path(f"/tmp/{artifact_prefix.replace('/', '_')}.json")
            path.write_text(json.dumps(self._matrix(qps=100.0, p50=10.0)))
            return path

        def fake_run_standard_benchmark(**kwargs):
            captured.update(kwargs)
            return {"paths": {"config": "config.json"}}

        with (
            patch.object(standard_benchmark, "run_modal_matrix", side_effect=fake_run_modal_matrix) as run_modal,
            patch.object(standard_benchmark, "run_standard_benchmark", side_effect=fake_run_standard_benchmark),
        ):
            standard_benchmark.main(
                [
                    "--opt-id",
                    "OPT-MATRIX",
                    "--name",
                    "matrix repetitions",
                    "--run-modal",
                    "--matrix-runs",
                    "3",
                    "--skip-rag",
                ]
            )

        self.assertEqual(run_modal.call_count, 3)
        self.assertEqual(len(captured["candidate_artifacts"]), 3)
        self.assertTrue(str(captured["candidate_artifacts"][0]).endswith("run_001.json"))
        self.assertTrue(str(captured["candidate_artifacts"][2]).endswith("run_003.json"))

    def test_summary_surfaces_positive_results_without_hiding_regressions(self):
        from eval import standard_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline = tmp_path / "baseline.json"
            candidate = tmp_path / "candidate.json"
            baseline.write_text(json.dumps(self._matrix(qps=100.0, p50=10.0)))
            candidate.write_text(json.dumps(self._matrix(qps=120.0, p50=20.0)))

            with (
                patch.object(standard_benchmark, "EXPERIMENT_ROOT", tmp_path / "experiment_results"),
                patch.object(standard_benchmark, "SUMMARY_ROOT", tmp_path / "experiment_summaries"),
            ):
                standard_benchmark.run_standard_benchmark(
                    opt_id="OPT-WINS",
                    name="wins and regressions",
                    baseline_artifacts=[baseline],
                    candidate_artifacts=[candidate],
                    dataset_sizes=[7],
                    skip_rag=True,
                )

            summary = (tmp_path / "experiment_summaries" / "OPT-WINS_summary.md").read_text()
            self.assertIn("## Positive Results", summary)
            self.assertIn("requests_per_s", summary)
            self.assertIn("## Regressions And Tradeoffs", summary)
            self.assertIn("latency_ms_p50", summary)

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
