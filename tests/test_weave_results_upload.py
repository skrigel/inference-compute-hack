import unittest


class WeaveResultsUploadTests(unittest.TestCase):
    def test_build_upload_bundle_flattens_phase04_matrix(self):
        from eval.upload_weave_results import build_upload_bundle

        matrix = {
            "run_id": "matrix-1",
            "model": "model",
            "vllm_version": "0.22.1",
            "gpu_counts": [1, 6],
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
                    "1": self._h100_result(100.0, 10.0),
                    "6": self._h100_result(540.0, 12.0),
                }
            },
            "rag_reference": {
                "rows": [
                    {
                        "backend": "numpy-fallback",
                        "n_docs": 1000,
                        "retrieve_ms_p50": 6.0,
                        "fresh_file_total_ms": 40.0,
                        "single_process_retrieve_qps_p50": 166.0,
                        "index_total_ms": 34.0,
                    }
                ]
            },
            "comparisons": [
                {
                    "scenario": "single_user_static",
                    "h100_replicas": 6,
                    "rag_n_docs": 1000,
                    "rag_metric": "retrieve_ms_p50",
                    "h100_latency_ms_p50": 12.0,
                    "rag_latency_ms": 6.0,
                    "rag_latency_over_h100_p50": 0.5,
                    "h100_requests_per_s": 540.0,
                    "rag_single_process_qps": 166.0,
                    "h100_qps_over_rag_single_process_qps": 3.25,
                }
            ],
        }

        bundle = build_upload_bundle(
            {
                "matrix": matrix,
                "quality": {"run_id": "quality-1", "quality": {"precision": 1.0, "recall": 0.9, "f1": 0.95}},
                "rag_vs_6xh100": {"run_id": "rag-1"},
                "artifact_paths": {"matrix": "matrix.json"},
            }
        )

        self.assertEqual(bundle["summary"]["matrix_run_id"], "matrix-1")
        self.assertEqual(bundle["summary"]["h100_row_count"], 2)
        self.assertEqual(bundle["summary"]["rag_row_count"], 1)
        self.assertEqual(bundle["summary"]["quality_f1"], 0.95)
        self.assertEqual(bundle["h100_rows"][0]["dataset_mode"], "static")
        self.assertEqual(bundle["h100_rows"][1]["gpu_count"], 6)
        self.assertEqual(bundle["comparison_rows"][0]["h100_qps_over_rag_single_process_qps"], 3.25)

    def _h100_result(self, qps: float, p50_ms: float) -> dict:
        return {
            "aggregate_client": {
                "requests_per_s": qps,
                "prompt_tokens_per_s": qps * 10,
                "total_tokens_per_s": qps * 11,
                "latency_ms_p50_mean": p50_ms,
                "latency_ms_p95_max": p50_ms * 2,
                "latency_ms_p99_max": p50_ms * 3,
            },
            "aggregate_server": {
                "derived_mfu_bf16_peak_mean": 0.01,
                "estimated_tflops_per_gpu_mean": 10.0,
                "gpu_utilization_pct_mean": 15.0,
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


if __name__ == "__main__":
    unittest.main()
