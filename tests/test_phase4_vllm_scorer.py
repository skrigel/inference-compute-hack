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


if __name__ == "__main__":
    unittest.main()
