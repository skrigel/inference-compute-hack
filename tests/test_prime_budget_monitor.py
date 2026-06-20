import unittest


class PrimeBudgetMonitorTests(unittest.TestCase):
    def test_extract_total_cost_from_plain_usage_output(self):
        from eval.prime_budget_monitor import extract_total_cost

        usage = """
        Run Usage — abc [RUNNING]
        Bucket              Tokens              Cost
        Training            528.00K             $0.08
        Inference (input)   467.91K             $0.02
        Inference (output)  76.27K              $0.01
        [bold]Total[/bold]  [bold]1.07M[/bold]  [bold]$0.11[/bold]
        """

        self.assertEqual(extract_total_cost(usage), 0.11)

    def test_budget_decision_uses_target_and_hard_limit(self):
        from eval.prime_budget_monitor import budget_decision

        self.assertEqual(budget_decision(59.99, target_cost=62.0, hard_limit=65.0), "continue")
        self.assertEqual(budget_decision(62.00, target_cost=62.0, hard_limit=65.0), "stop_target")
        self.assertEqual(budget_decision(65.01, target_cost=62.0, hard_limit=65.0), "stop_hard_limit")


if __name__ == "__main__":
    unittest.main()
