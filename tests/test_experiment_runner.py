import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ExperimentRunnerTests(unittest.TestCase):
    def test_experiment_config_loads_from_spec(self):
        from eval.experiment_runner import EXPERIMENTS

        self.assertIn("EXP-FP8-001", EXPERIMENTS)
        self.assertIn("EXP-BATCH-001", EXPERIMENTS)
        self.assertIn("EXP-MBT-001", EXPERIMENTS)
        self.assertIn("EXP-SCHED-001", EXPERIMENTS)
        self.assertIn("EXP-LENBIN-001", EXPERIMENTS)
        self.assertIn("EXP-OVERLAP-001", EXPERIMENTS)

    def test_experiment_config_has_required_fields(self):
        from eval.experiment_runner import EXPERIMENTS

        required_fields = {"name", "env_vars", "hypothesis", "success_criteria"}
        for exp_id, config in EXPERIMENTS.items():
            for field in required_fields:
                self.assertIn(field, config, f"{exp_id} missing {field}")

    def test_build_env_merges_with_baseline(self):
        from eval.experiment_runner import _build_env

        baseline = {"A": "1", "B": "2"}
        overrides = {"B": "3", "C": "4"}

        result = _build_env(baseline, overrides)

        self.assertEqual(result["A"], "1")
        self.assertEqual(result["B"], "3")
        self.assertEqual(result["C"], "4")

    def test_run_experiment_creates_artifact_directory(self):
        from eval.experiment_runner import _ensure_experiment_dir

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            path = _ensure_experiment_dir("EXP-TEST-001")
            mock_mkdir.assert_called()
            self.assertIn("EXP-TEST-001", str(path))

    def test_repetitions_drive_h100_matrix_runs_and_warmup_exclusion(self):
        from eval import experiment_runner

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(experiment_runner, "EXPERIMENT_ROOT", Path(tmp) / "experiment_results"):
                result = experiment_runner.run_experiment(
                    "EXP-BATCH-001",
                    repetitions=5,
                    gpu_counts="1,6",
                    dataset_sizes=[7],
                    dry_run=True,
                )

        self.assertIn("--matrix-runs 5", result["command"])
        self.assertIn("--rag-runs 5", result["command"])
        self.assertIn("--warmup-excluded", result["command"])


if __name__ == "__main__":
    unittest.main()
