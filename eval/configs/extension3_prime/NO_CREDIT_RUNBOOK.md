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
prime --help
```

4. Optional if resuming. Run:

```bash
prime train checkpoints <existing-run-id>
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

Checkpoint settings:

- Target hardware: `8x H100 80GB on Prime`.
- Full checkpoints every `25` steps with `keep_cloud = true`.
- Adapter uploads every `25` steps; keep last `4` adapters.
- To resume, run `prime train checkpoints <run-id>` and set top-level `checkpoint_id`.

Stop immediately if baseline eval artifacts are missing, the heldout split is empty, no checkpoint appears after the first interval, or the first pilot run fails to improve heldout reward.
