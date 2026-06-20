import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx


def completion_response(index: int, top_logprobs: dict[str, float]) -> dict:
    return {
        "choices": [
            {
                "index": index,
                "text": " Yes",
                "logprobs": {
                    "tokens": [" Yes"],
                    "token_logprobs": [-0.1],
                    "top_logprobs": [top_logprobs],
                },
            }
        ]
    }


class VLLMScorerTests(unittest.TestCase):
    def test_make_scorer_returns_vllm_scorer(self):
        from inference.config import make_scorer
        from inference.vllm_scorer import VLLMScorer

        with patch.dict(os.environ, {"SCORER_BACKEND": "vllm", "VLLM_REPLICAS": "http://r1/v1"}):
            self.assertIsInstance(make_scorer(), VLLMScorer)

    def test_yes_no_logprobs_normalize_score(self):
        from inference.scorer import ScoreRequest
        from inference.vllm_scorer import VLLMScorer

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=completion_response(0, {" Yes": -0.1, " No": -2.3}),
            )

        scorer = VLLMScorer(["http://r1/v1"], transport=httpx.MockTransport(handler))
        result = asyncio.run(scorer.score_batch([ScoreRequest("c1", "retry", "retry")]))[0]

        self.assertEqual(result.chunk_id, "c1")
        self.assertGreater(result.score, 0.8)
        self.assertAlmostEqual(result.score, result.p_yes / (result.p_yes + result.p_no))

    def test_response_order_matches_request_order(self):
        from inference.scorer import ScoreRequest
        from inference.vllm_scorer import VLLMScorer

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            prompt = body["prompt"]
            if "second chunk" in prompt:
                return httpx.Response(200, json=completion_response(0, {" Yes": -2.0, " No": -0.1}))
            return httpx.Response(200, json=completion_response(0, {" Yes": -0.1, " No": -2.0}))

        scorer = VLLMScorer(["http://r1/v1"], transport=httpx.MockTransport(handler))
        results = asyncio.run(
            scorer.score_batch(
                [
                    ScoreRequest("first", "first chunk", "predicate"),
                    ScoreRequest("second", "second chunk", "predicate"),
                ]
            )
        )

        self.assertEqual([result.chunk_id for result in results], ["first", "second"])
        self.assertGreater(results[0].score, results[1].score)

    def test_missing_yes_no_logprobs_raises_clear_error(self):
        from inference.scorer import ScoreRequest
        from inference.vllm_scorer import VLLMScorer, VLLMScoringError

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=completion_response(0, {" Maybe": -0.1}))

        scorer = VLLMScorer(["http://r1/v1"], transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(VLLMScoringError, "Yes/No logprobs"):
            asyncio.run(scorer.score_batch([ScoreRequest("c1", "retry", "retry")]))

    def test_multiple_replicas_are_round_robined(self):
        from inference.scorer import ScoreRequest
        from inference.vllm_scorer import VLLMScorer

        seen_hosts = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_hosts.append(request.url.host)
            return httpx.Response(200, json=completion_response(0, {" Yes": -0.1, " No": -2.0}))

        scorer = VLLMScorer(
            ["http://r1/v1", "http://r2/v1"],
            transport=httpx.MockTransport(handler),
        )
        asyncio.run(
            scorer.score_batch(
                [
                    ScoreRequest("c1", "chunk 1", "predicate"),
                    ScoreRequest("c2", "chunk 2", "predicate"),
                    ScoreRequest("c3", "chunk 3", "predicate"),
                ]
            )
        )

        self.assertEqual(seen_hosts, ["r1", "r2", "r1"])

    def test_chunk_sticky_routing_keeps_same_chunk_on_same_replica(self):
        from inference.scorer import ScoreRequest
        from inference.vllm_scorer import VLLMScorer

        seen_hosts = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_hosts.append(request.url.host)
            return httpx.Response(200, json=completion_response(0, {" Yes": -0.1, " No": -2.0}))

        scorer = VLLMScorer(
            ["http://r1/v1", "http://r2/v1"],
            routing_mode="chunk_sticky",
            transport=httpx.MockTransport(handler),
        )
        asyncio.run(
            scorer.score_batch(
                [
                    ScoreRequest("same-chunk", "chunk 1", "predicate 1"),
                    ScoreRequest("same-chunk", "chunk 1", "predicate 2"),
                ]
            )
        )

        self.assertEqual(len(seen_hosts), 2)
        self.assertEqual(seen_hosts[0], seen_hosts[1])

    def test_health_reports_routing_and_priority_settings(self):
        from inference.vllm_scorer import VLLMScorer

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": []})
            return httpx.Response(200, json=completion_response(0, {" Yes": -0.1, " No": -2.0}))

        scorer = VLLMScorer(
            ["http://r1/v1"],
            max_concurrency=8,
            priority_reserved=2,
            routing_mode="chunk_sticky",
            transport=httpx.MockTransport(handler),
        )
        health = asyncio.run(scorer.health())

        self.assertEqual(health["routing_mode"], "chunk_sticky")
        self.assertEqual(health["max_concurrency"], 8)
        self.assertEqual(health["priority_reserved"], 2)

    def test_prometheus_parser_keeps_mfu_latency_and_queue_metrics(self):
        from inference.vllm_scorer import parse_vllm_prometheus_metrics

        text = "\n".join(
            [
                "# HELP vllm:estimated_flops_per_gpu_total Estimated FLOPs",
                'vllm:estimated_flops_per_gpu_total{model_name="tier1-filter"} 9.895e14',
                'vllm:estimated_read_bytes_per_gpu_total{model_name="tier1-filter"} 1000',
                'vllm:estimated_write_bytes_per_gpu_total{model_name="tier1-filter"} 500',
                'vllm:prompt_tokens_total{model_name="tier1-filter"} 128',
                'vllm:generation_tokens_total{model_name="tier1-filter"} 8',
                'vllm:request_success_total{finished_reason="stop",model_name="tier1-filter"} 4',
                'vllm:request_success_total{finished_reason="length",model_name="tier1-filter"} 0',
                'vllm:kv_cache_usage_perc{model_name="tier1-filter"} 0.42',
                'vllm:num_requests_running{model_name="tier1-filter"} 8',
                'vllm:num_requests_waiting{model_name="tier1-filter"} 2',
                'vllm:time_to_first_token_seconds_sum{model_name="tier1-filter"} 1.5',
                'vllm:time_to_first_token_seconds_count{model_name="tier1-filter"} 10',
                'vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="tier1-filter"} 8',
            ]
        )

        metrics = parse_vllm_prometheus_metrics(text)

        self.assertEqual(metrics["vllm:estimated_flops_per_gpu_total"], 9.895e14)
        self.assertEqual(metrics["vllm:prompt_tokens_total"], 128)
        self.assertEqual(metrics["vllm:generation_tokens_total"], 8)
        self.assertEqual(metrics["vllm:request_success_total"], 4)
        self.assertEqual(metrics["vllm:kv_cache_usage_perc"], 0.42)
        self.assertEqual(metrics["vllm:num_requests_running"], 8)
        self.assertEqual(metrics["vllm:num_requests_waiting"], 2)
        self.assertEqual(metrics["vllm:time_to_first_token_seconds_sum"], 1.5)
        self.assertNotIn("vllm:time_to_first_token_seconds_bucket", metrics)

    def test_metric_delta_derives_mfu_from_flop_counter(self):
        from inference.vllm_scorer import H100_SXM_BF16_FLOPS_PER_GPU, summarize_vllm_metric_delta

        before = {
            "vllm:estimated_flops_per_gpu_total": 100.0,
            "vllm:prompt_tokens_total": 10.0,
            "vllm:generation_tokens_total": 1.0,
            "vllm:request_success_total": 1.0,
        }
        after = {
            "vllm:estimated_flops_per_gpu_total": H100_SXM_BF16_FLOPS_PER_GPU + 100.0,
            "vllm:prompt_tokens_total": 110.0,
            "vllm:generation_tokens_total": 11.0,
            "vllm:request_success_total": 11.0,
            "vllm:e2e_request_latency_seconds_sum": 1.0,
            "vllm:e2e_request_latency_seconds_count": 10.0,
            "vllm:time_to_first_token_seconds_sum": 0.5,
            "vllm:time_to_first_token_seconds_count": 10.0,
        }

        summary = summarize_vllm_metric_delta(before, after, 1.0)

        self.assertAlmostEqual(summary["derived_mfu_bf16_peak"], 1.0)
        self.assertAlmostEqual(summary["estimated_tflops_per_gpu"], H100_SXM_BF16_FLOPS_PER_GPU / 1e12)
        self.assertEqual(summary["prompt_tokens_per_s"], 100.0)
        self.assertEqual(summary["generation_tokens_per_s"], 10.0)
        self.assertEqual(summary["requests_per_s"], 10.0)
        self.assertEqual(summary["server_e2e_latency_avg_ms"], 100.0)
        self.assertEqual(summary["server_ttft_avg_ms"], 50.0)

    def test_metric_delta_uses_histogram_count_when_success_counter_is_absent(self):
        from inference.vllm_scorer import summarize_vllm_metric_delta

        summary = summarize_vllm_metric_delta(
            {"vllm:e2e_request_latency_seconds_count": 1.0, "vllm:request_success_total": 0.0},
            {"vllm:e2e_request_latency_seconds_count": 65.0, "vllm:request_success_total": 0.0},
            0.5,
        )

        self.assertEqual(summary["request_success_delta"], 0.0)
        self.assertEqual(summary["request_count_delta"], 64.0)
        self.assertEqual(summary["requests_per_s"], 128.0)

    def test_modal_benchmark_static_dataset_reuses_prompt_prefix(self):
        from inference.modal_app import _benchmark_prompt

        static_first = _benchmark_prompt(0, variant="compact", dataset_mode="static")
        static_second = _benchmark_prompt(1, variant="compact", dataset_mode="static")
        dynamic_first = _benchmark_prompt(0, variant="compact", dataset_mode="dynamic")
        dynamic_second = _benchmark_prompt(1, variant="compact", dataset_mode="dynamic")

        self.assertEqual(static_first, static_second)
        self.assertNotEqual(dynamic_first, dynamic_second)
        self.assertIn("shared-static", static_first)
        self.assertIn("dynamic-0", dynamic_first)

    def test_modal_benchmark_summarizes_gpu_samples(self):
        from inference.modal_app import _summarize_gpu_samples

        summary = _summarize_gpu_samples(
            [
                {
                    "gpu_utilization_pct": 10.0,
                    "gpu_memory_used_mb": 20_000.0,
                    "gpu_memory_total_mb": 80_000.0,
                    "gpu_power_w": 250.0,
                    "gpu_power_limit_w": 700.0,
                },
                {
                    "gpu_utilization_pct": 90.0,
                    "gpu_memory_used_mb": 40_000.0,
                    "gpu_memory_total_mb": 80_000.0,
                    "gpu_power_w": 500.0,
                    "gpu_power_limit_w": 700.0,
                },
                {"error": "sample failed"},
            ]
        )

        self.assertEqual(summary["gpu_sample_count"], 2.0)
        self.assertEqual(summary["gpu_utilization_pct_mean"], 50.0)
        self.assertEqual(summary["gpu_utilization_pct_max"], 90.0)
        self.assertEqual(summary["gpu_memory_used_mb_max"], 40_000.0)
        self.assertEqual(summary["gpu_memory_utilization_pct_max"], 50.0)
        self.assertAlmostEqual(summary["gpu_power_utilization_pct_mean"], (250.0 / 700.0 * 100.0 + 500.0 / 700.0 * 100.0) / 2)

    def test_h100_rag_matrix_markdown_includes_gpu_scaling(self):
        from inference.modal_app import _h100_rag_matrix_markdown

        def result(qps: float, p50_ms: float) -> dict:
            return {
                "aggregate_client": {
                    "requests_per_s": qps,
                    "latency_ms_p50_mean": p50_ms,
                    "latency_ms_p95_max": p50_ms * 2,
                },
                "aggregate_server": {
                    "derived_mfu_bf16_peak_mean": 0.01,
                    "gpu_utilization_pct_mean": 10.0,
                    "gpu_utilization_pct_max": 20.0,
                    "gpu_power_w_mean": 100.0,
                    "gpu_power_w_max": 120.0,
                    "gpu_memory_used_mb_max": 70_000.0,
                },
            }

        markdown = _h100_rag_matrix_markdown(
            {
                "run_id": "run-1",
                "model": "model",
                "vllm_version": "vllm",
                "prompt_variant": "compact",
                "gpu_memory_utilization": 0.92,
                "gpu_counts": [1, 6],
                "h100_results": {
                    "single_user_static": {
                        "1": result(100.0, 10.0),
                        "6": result(540.0, 12.0),
                    }
                },
                "rag_reference": {
                    "rows": [
                        {
                            "n_docs": 1000,
                            "retrieve_ms_p50": 5.0,
                            "fresh_file_total_ms": 50.0,
                            "single_process_retrieve_qps_p50": 200.0,
                        }
                    ]
                },
                "comparisons": [
                    {
                        "scenario": "single_user_static",
                        "h100_replicas": 6,
                        "rag_n_docs": 1000,
                        "rag_latency_over_h100_p50": 4.0,
                    }
                ],
                "refinement_overlap": [],
            }
        )

        self.assertIn("## 1 vs 6 H100 Scaling", markdown)
        self.assertIn("| single_user_static | 100.000 | 540.000 | 5.400x | 10.000 | 12.000 | 1.200x |", markdown)

    def test_modal_benchmark_parses_csv_ints(self):
        from inference.modal_app import _parse_int_csv

        self.assertEqual(_parse_int_csv("1,6, 100000"), [1, 6, 100000])


if __name__ == "__main__":
    unittest.main()
