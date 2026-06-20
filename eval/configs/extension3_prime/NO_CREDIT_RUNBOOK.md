# Extension 3 Prime No-Credit Runbook

This runbook validates setup before any Prime training credits are used.

## No-Credit Smoke

1. Run:

```bash
python -m eval.agent_loop_prime --output-dir eval/artifacts/prime_readiness --smoke-docs 60 --task-count 3
```

2. Run:

```bash
python -m eval.agent_loop --n-docs 60 --task-count 3 --max-steps 5
```

3. Run:

```bash
modal run inference/modal_app.py::extension3_agent_loop_smoke --n-docs 60 --task-count 3 --max-steps 5
```

4. Run:

```bash
prime --help
```

Pass criteria:

- Mean truth gain >= `0.1`.
- Pass rate >= `0.8`.
- No command in this section may start with `prime train run`.

## Paid Training Gate

Only after the smoke passes and the user explicitly approves credit use:

```bash
prime train run eval/configs/extension3_prime/prime_train.example.toml
```

Stop immediately if baseline eval artifacts are missing, the heldout split is empty, or the first pilot run fails to improve heldout reward.
