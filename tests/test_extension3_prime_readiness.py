import json
import tempfile
import tomllib
import unittest
from pathlib import Path


class Extension3PrimeReadinessTests(unittest.TestCase):
    def test_prime_readiness_packet_has_training_gates_and_cohorts(self):
        from eval.agent_loop_prime import build_prime_readiness_packet

        packet = build_prime_readiness_packet()

        self.assertEqual(packet["schema_version"], "extension3.prime_readiness.v1")
        self.assertTrue(packet["credit_policy"]["no_credit_smoke_required"])
        self.assertGreaterEqual(len(packet["cohort_manifest"]["cohorts"]), 8)
        self.assertIn("prime train run", packet["prime_training_config"]["launch"]["command"])
        self.assertTrue(packet["prime_training_config"]["launch"]["requires_prime_credits"])
        self.assertEqual(packet["prime_training_config"]["hardware"]["gpu_count"], 8)
        self.assertEqual(packet["prime_training_config"]["hardware"]["gpu_type"], "H100")
        self.assertLessEqual(packet["prime_training_config"]["checkpointing"]["checkpoint_interval"], 25)
        self.assertTrue(packet["prime_training_config"]["checkpointing"]["keep_cloud_checkpoints"])
        for command in packet["no_credit_smoke"]["commands"]:
            self.assertNotIn("prime train run", command)
        for cohort in packet["cohort_manifest"]["cohorts"]:
            self.assertIn(cohort["split"], {"train", "heldout"})
            self.assertIn("single_changed_variable", cohort)
            self.assertGreater(cohort["n_docs"], 0)
            self.assertGreater(cohort["max_steps"], 0)

    def test_write_prime_readiness_artifacts_outputs_parseable_files(self):
        from eval.agent_loop_prime import write_prime_readiness_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            paths = write_prime_readiness_artifacts(output_dir)

            self.assertEqual(set(paths), {"cohorts", "train_config", "lift_schema", "reward_contract", "runbook"})
            cohorts = json.loads(paths["cohorts"].read_text())
            lift_schema = json.loads(paths["lift_schema"].read_text())
            train_config = tomllib.loads(paths["train_config"].read_text())

            self.assertGreaterEqual(len(cohorts["cohorts"]), 8)
            self.assertEqual(train_config["run"]["project"], "inference-compute-hack")
            self.assertEqual(train_config["hardware"]["gpu_count"], 8)
            self.assertEqual(train_config["hardware"]["gpu_type"], "H100")
            self.assertEqual(train_config["checkpoints"]["interval"], 25)
            self.assertTrue(train_config["checkpoints"]["keep_cloud"])
            self.assertEqual(train_config["adapters"]["keep_last"], 4)
            self.assertEqual(train_config["resume"]["checkpoint_id"], "")
            self.assertEqual(train_config["environment"]["id"], "extension3-agent-loop")
            self.assertEqual(lift_schema["schema_version"], "extension3.metric_to_lift.v1")

    def test_no_credit_readiness_smoke_writes_report_without_training_launch(self):
        from eval.agent_loop_prime import run_no_credit_readiness_check

        with tempfile.TemporaryDirectory() as tmp:
            report = run_no_credit_readiness_check(Path(tmp), smoke_docs=60, task_count=3)

            self.assertFalse(report["paid_training_launched"])
            self.assertTrue(report["checks"]["local_agent_loop_smoke"]["passed"])
            self.assertGreater(report["checks"]["local_agent_loop_smoke"]["mean_truth_gain"], 0.0)
            self.assertIn("prime_cli", report["checks"])
            self.assertTrue(Path(report["artifacts"]["cohorts"]).exists())
            self.assertTrue(Path(report["artifacts"]["runbook"]).exists())


if __name__ == "__main__":
    unittest.main()
