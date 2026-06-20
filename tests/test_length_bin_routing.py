import asyncio
import json
import unittest

import httpx


def completion_response(top_logprobs: dict[str, float]) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "text": " Yes",
                "logprobs": {
                    "tokens": [" Yes"],
                    "token_logprobs": [-0.1],
                    "top_logprobs": [top_logprobs],
                },
            }
        ]
    }


class LengthBinRoutingTests(unittest.TestCase):
    def test_estimate_tokens_approximates_word_count(self):
        from inference.vllm_scorer import _estimate_tokens

        short_text = "Hello world"
        long_text = " ".join(["word"] * 1000)

        self.assertLess(_estimate_tokens(short_text), 50)
        self.assertGreater(_estimate_tokens(long_text), 500)

    def test_length_bin_categorizes_correctly(self):
        from inference.vllm_scorer import _length_bin

        self.assertEqual(_length_bin(100), "short")
        self.assertEqual(_length_bin(512), "medium")
        self.assertEqual(_length_bin(1000), "medium")
        self.assertEqual(_length_bin(2048), "long")
        self.assertEqual(_length_bin(3000), "long")

    def test_length_bin_routing_groups_similar_lengths(self):
        from inference.scorer import ScoreRequest
        from inference.vllm_scorer import VLLMScorer

        seen_hosts: list[tuple[str, int]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            prompt_len = len(body["prompt"])
            seen_hosts.append((request.url.host, prompt_len))
            return httpx.Response(200, json=completion_response({" Yes": -0.1, " No": -2.0}))

        scorer = VLLMScorer(
            ["http://short/v1", "http://medium/v1", "http://long/v1"],
            routing_mode="length_bin",
            transport=httpx.MockTransport(handler),
        )

        short_text = "short text"
        long_text = " ".join(["word"] * 600)

        asyncio.run(
            scorer.score_batch(
                [
                    ScoreRequest("c1", short_text, "predicate"),
                    ScoreRequest("c2", short_text, "predicate"),
                    ScoreRequest("c3", long_text, "predicate"),
                ]
            )
        )

        short_hosts = [host for host, _ in seen_hosts[:2]]
        self.assertEqual(short_hosts[0], short_hosts[1])


if __name__ == "__main__":
    unittest.main()
