import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


class FakeWeave(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.inits = []
        self.ops = []

    def init(self, project, **kwargs):
        self.inits.append((project, kwargs))
        return {"project": project}

    def op(self, **kwargs):
        self.ops.append(kwargs)

        def decorator(func):
            def wrapped(*args, **inner_kwargs):
                return func(*args, **inner_kwargs)

            wrapped.__name__ = func.__name__
            return wrapped

        return decorator


class EvalWeaveOpsTests(unittest.TestCase):
    def test_default_project_matches_wandb_workspace(self):
        from eval.weave_ops import DEFAULT_WEAVE_PROJECT

        self.assertEqual(
            DEFAULT_WEAVE_PROJECT,
            "sasha-krigel-massachusetts-institute-of-technology/inference-hack",
        )

    def test_init_weave_calls_default_project(self):
        fake = FakeWeave()

        with patch.dict(sys.modules, {"weave": fake}), patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
            from eval.weave_ops import init_weave

            result = init_weave()

        self.assertEqual(result, {"project": "sasha-krigel-massachusetts-institute-of-technology/inference-hack"})
        self.assertEqual(fake.inits, [("sasha-krigel-massachusetts-institute-of-technology/inference-hack", {})])

    def test_init_weave_passes_global_attributes(self):
        fake = FakeWeave()

        with patch.dict(sys.modules, {"weave": fake}), patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
            from eval.weave_ops import init_weave

            init_weave(global_attributes={"commit": "abc123", "run_id": "run-1"})

        self.assertEqual(
            fake.inits,
            [
                (
                    "sasha-krigel-massachusetts-institute-of-technology/inference-hack",
                    {"global_attributes": {"commit": "abc123", "run_id": "run-1"}},
                )
            ],
        )

    def test_init_weave_requires_noninteractive_auth(self):
        fake = FakeWeave()

        with patch.dict(sys.modules, {"weave": fake}), patch.dict("os.environ", {}, clear=True):
            from eval import weave_ops

            with (
                patch.object(weave_ops.netrc, "netrc", side_effect=FileNotFoundError),
                self.assertRaisesRegex(weave_ops.WeaveSetupError, "W&B auth is missing"),
            ):
                weave_ops.init_weave()

        self.assertEqual(fake.inits, [])

    def test_weave_op_is_noop_when_weave_is_not_installed(self):
        with patch.dict(sys.modules, {"weave": None}):
            from eval.weave_ops import weave_op

            def identity(value):
                return value

            decorated = weave_op(name="eval.identity")(identity)

        self.assertIs(decorated, identity)
        self.assertEqual(decorated("trace"), "trace")

    def test_weave_smoke_path_initializes_project_before_running_eval(self):
        from eval import bench

        with (
            patch.object(bench, "init_weave") as init_mock,
            patch.object(bench, "run_smoke_traced", new=AsyncMock(return_value={"ok": True})) as run_mock,
        ):
            payload = bench.run_smoke_with_weave(
                "sasha-krigel-massachusetts-institute-of-technology/inference-hack"
            )

        init_mock.assert_called_once_with("sasha-krigel-massachusetts-institute-of-technology/inference-hack")
        run_mock.assert_awaited_once()
        self.assertEqual(payload, {"ok": True})


if __name__ == "__main__":
    unittest.main()
