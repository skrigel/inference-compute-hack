import asyncio
import unittest


class Extension3AgentLoopTests(unittest.TestCase):
    def test_dynamic_corpus_generates_large_labeled_task(self):
        from eval.agent_loop import generate_dynamic_query_task

        task = generate_dynamic_query_task(task_id="task-1", n_docs=250, target_topic="retry_backoff")

        self.assertEqual(len(task.chunks), 250)
        self.assertGreaterEqual(len(task.positive_chunk_ids), 5)
        self.assertIn("retry", task.initial_query)
        self.assertGreater(task.memory_bytes_total, 0)
        self.assertLess(task.positive_bytes, task.memory_bytes_total)

    def test_agent_loop_improves_reward_over_initial_query(self):
        from eval.agent_loop import generate_dynamic_query_task, run_query_refinement_episode
        from inference.mock_scorer import MockScorer

        task = generate_dynamic_query_task(task_id="task-1", n_docs=120, target_topic="retry_backoff")
        episode = asyncio.run(
            run_query_refinement_episode(
                task,
                scorer=MockScorer(),
                threshold=0.5,
                max_steps=4,
                beam_width=4,
            )
        )

        self.assertGreater(episode.best_reward, episode.steps[0].reward)
        self.assertGreater(episode.metrics["truth_gain"], 0.0)
        self.assertGreaterEqual(episode.best_quality["recall"], 0.8)
        self.assertGreaterEqual(episode.metrics["branching_factor"], 2)
        self.assertLess(episode.metrics["movement_selectivity"], 1.0)

    def test_experiment_reports_task_and_dataset_metrics(self):
        from eval.agent_loop import run_agent_loop_experiment

        payload = asyncio.run(
            run_agent_loop_experiment(
                n_docs=140,
                task_count=3,
                max_steps=4,
                beam_width=4,
                threshold=0.5,
                human_turn_ms=30_000.0,
            )
        )

        self.assertEqual(payload["task_count"], 3)
        self.assertEqual(len(payload["episodes"]), 3)
        metrics = payload["dataset_metrics"]
        self.assertIn("mean_best_reward", metrics)
        self.assertIn("trajectory_entropy", metrics)
        self.assertIn("cost_quality_frontier", metrics)
        self.assertGreater(metrics["agent_vs_human_speedup_estimate"], 1.0)
        for episode in payload["episodes"]:
            self.assertIn("memory_selectivity", episode["metrics"])
            self.assertIn("movement_selectivity", episode["metrics"])
            self.assertIn("truth_gain", episode["metrics"])

    def test_agent_loop_markdown_report_contains_three_axis_metrics(self):
        from eval.agent_loop import _markdown_report, run_agent_loop_experiment

        payload = asyncio.run(run_agent_loop_experiment(n_docs=60, task_count=1, max_steps=3, beam_width=3))
        report = _markdown_report(payload)

        self.assertIn("mean_memory_selectivity", report)
        self.assertIn("mean_movement_selectivity", report)
        self.assertIn("mean_best_f1", report)


if __name__ == "__main__":
    unittest.main()
