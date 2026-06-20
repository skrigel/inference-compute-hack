import json
import tempfile
import tomllib
import unittest
from pathlib import Path


class Extension3ModalCheckpointTests(unittest.TestCase):
    def test_budget_policy_caps_six_h100_training_under_18_dollars(self):
        from eval.agent_loop_modal_checkpoints import build_modal_checkpoint_packet

        packet = build_modal_checkpoint_packet(budget_usd=18.0, gpu_count=6)
        budget = packet["budget"]
        checkpoint = packet["checkpoint_policy"]

        self.assertEqual(packet["modal"]["gpu_request"], "H100:6")
        self.assertEqual(budget["gpu_count"], 6)
        self.assertLessEqual(budget["planned_training_gpu_cost_usd"], 18.0)
        self.assertLess(budget["planned_training_gpu_cost_usd"], budget["budget_usd"])
        self.assertGreaterEqual(budget["reserve_usd"], 3.0)
        self.assertEqual(checkpoint["save_every_seconds"], 180)
        self.assertGreaterEqual(checkpoint["expected_checkpoints"], 10)
        self.assertLessEqual(checkpoint["keep_last"], 4)

    def test_write_modal_checkpoint_artifacts_outputs_parseable_files(self):
        from eval.agent_loop_modal_checkpoints import write_modal_checkpoint_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            paths = write_modal_checkpoint_artifacts(output_dir)

            self.assertEqual(set(paths), {"policy", "train_config", "runbook"})
            policy = json.loads(paths["policy"].read_text())
            config = tomllib.loads(paths["train_config"].read_text())

            self.assertEqual(policy["schema_version"], "extension3.modal_checkpoint.v1")
            self.assertEqual(config["modal"]["gpu_request"], "H100:6")
            self.assertTrue(config["checkpoint"]["resume_from_latest"])
            self.assertEqual(config["checkpoint"]["save_every_seconds"], 180)

    def test_custom_budget_is_consistent_across_generated_files(self):
        from eval.agent_loop_modal_checkpoints import write_modal_checkpoint_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_modal_checkpoint_artifacts(Path(tmp), budget_usd=12.0, gpu_count=6)
            policy = json.loads(paths["policy"].read_text())
            config = tomllib.loads(paths["train_config"].read_text())

            self.assertEqual(policy["budget"]["budget_usd"], 12.0)
            self.assertEqual(config["budget"]["budget_usd"], 12.0)
            self.assertEqual(policy["budget"]["planned_training_seconds"], config["budget"]["planned_training_seconds"])

    def test_local_checkpoint_dry_run_writes_resume_pointer_without_gpu(self):
        from eval.agent_loop_modal_checkpoints import run_local_checkpoint_dry_run

        with tempfile.TemporaryDirectory() as tmp:
            report = run_local_checkpoint_dry_run(Path(tmp), total_steps=7, save_every_steps=2)

            self.assertFalse(report["paid_modal_gpu_launched"])
            self.assertTrue(report["passed"])
            self.assertEqual(report["latest_step"], 6)
            self.assertEqual(len(report["checkpoints_written"]), 3)
            latest_path = Path(report["latest_checkpoint_path"])
            self.assertTrue(latest_path.exists())
            latest = json.loads(latest_path.read_text())
            self.assertEqual(latest["step"], 6)


if __name__ == "__main__":
    unittest.main()
