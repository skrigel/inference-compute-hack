import unittest


class EvalTraceModelTests(unittest.TestCase):
    def test_operation_aliases_normalize_to_contract_vocabulary(self):
        from eval.trace import normalize_operation

        self.assertEqual(normalize_operation("and"), "require")
        self.assertEqual(normalize_operation("not"), "exclude")
        self.assertEqual(normalize_operation("or"), "include")
        self.assertEqual(normalize_operation("rewrite"), "refocus")
        self.assertEqual(normalize_operation("threshold"), "brush")
        self.assertEqual(normalize_operation("query"), "query")

    def test_turn_trace_serializes_eval_plan_fields(self):
        from eval.trace import QualityMetrics, TurnTrace

        trace = TurnTrace(
            run_id="run-1",
            commit="abc123",
            corpus_id="demo",
            model_id="mock",
            scorer_backend="mock",
            turn=2,
            operation="and",
            threshold=0.5,
            n_chunks_total=100,
            candidate_count=40,
            chunks_scored=20,
            chunks_served_from_cache=20,
            survivor_count=10,
            elapsed_ms=12.5,
            model_ms=8.0,
            queue_ms=1.0,
            ttft_ms=7.5,
            cache_hit_rate=0.5,
            gpu_cache_usage_perc=0.0,
            quality_slice=QualityMetrics(precision=0.8, recall=0.6, f1=0.6857, auc=None, ece=None),
        )

        payload = trace.to_dict()

        self.assertEqual(payload["operation"], "require")
        self.assertEqual(payload["rho"], 0.25)
        self.assertEqual(payload["ttft_ms"], 7.5)
        self.assertEqual(payload["gpu_cache_usage_perc"], 0.0)
        self.assertEqual(payload["quality_slice"]["precision"], 0.8)

    def test_turn_trace_zero_candidate_count_has_zero_rho(self):
        from eval.trace import TurnTrace

        trace = TurnTrace(
            run_id="run-1",
            commit="abc123",
            corpus_id="demo",
            model_id="mock",
            scorer_backend="mock",
            turn=1,
            operation="threshold",
            threshold=0.7,
            n_chunks_total=100,
            candidate_count=0,
            chunks_scored=0,
            chunks_served_from_cache=100,
            survivor_count=0,
            elapsed_ms=1.0,
            model_ms=0.0,
            queue_ms=0.0,
            ttft_ms=0.0,
            cache_hit_rate=1.0,
            gpu_cache_usage_perc=0.0,
            quality_slice=None,
        )

        payload = trace.to_dict()

        self.assertEqual(payload["operation"], "brush")
        self.assertEqual(payload["rho"], 0.0)
        self.assertEqual(payload["chunks_scored"], 0)


if __name__ == "__main__":
    unittest.main()
