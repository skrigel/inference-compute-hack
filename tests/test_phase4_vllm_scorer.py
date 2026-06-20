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


if __name__ == "__main__":
    unittest.main()
