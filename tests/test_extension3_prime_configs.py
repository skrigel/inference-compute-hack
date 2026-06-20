import tomllib
import unittest
from pathlib import Path


class Extension3PrimeConfigTests(unittest.TestCase):
    def test_full_budget_config_uses_large_model_and_large_retrieval_data(self):
        config_path = Path("eval/configs/extension3_prime/prime_train.full.toml")
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)

        self.assertEqual(config["model"], "Qwen/Qwen3.5-35B-A3B")
        self.assertGreaterEqual(config["max_steps"], 1000)
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["rollouts_per_example"], 8)
        self.assertEqual(config["sampling"]["max_tokens"], 256)
        self.assertFalse(config["sampling"]["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(config["eval"]["sampling"]["max_tokens"], 256)
        self.assertEqual(config["eval"]["sampling"]["temperature"], 0.0)
        self.assertFalse(config["eval"]["sampling"]["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(config["checkpoints"]["interval"], 25)
        self.assertGreaterEqual(config["checkpoints"]["keep_cloud"], 8)
        self.assertEqual(config["env"][0]["args"]["max_examples"], -1)
        self.assertEqual(config["env"][0]["args"]["passages_per_prompt"], 16)
        self.assertEqual(config["eval"]["env"][0]["args"]["max_examples"], -1)
        self.assertEqual(config["eval"]["env"][0]["args"]["passages_per_prompt"], 16)
        self.assertGreaterEqual(config["eval"]["env"][0]["num_examples"], 72)


if __name__ == "__main__":
    unittest.main()
