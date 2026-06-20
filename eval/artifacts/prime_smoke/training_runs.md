# Prime Hosted Training Runs

This ledger records the first Prime Hosted Training validation runs for the
Extension 3 query-refinement environment.

## Environment Versions

- `inference/extension3-agent-loop@0.1.0`: eval worked, but hosted training
  failed when Prime tried to serialize rollout samples containing dict answers.
- `inference/extension3-agent-loop@0.1.1`: stores `answer` as JSON text and
  parses it inside reward functions. This is the current passing version.

## Runs

| Run ID | Config | Status | Notes |
|---|---|---:|---|
| `mkyr5jlfznstwd5cnhqhkhdf` | `prime_train.example.toml` against env `0.1.0` | Failed | Reached step 0, then `pyarrow.lib.ArrowTypeError: Expected bytes, got a 'dict' object` in Prime sample logging. Cost was about `$0.0016`. |
| `pp76jxcpojf2t5uvipy5tf8j` | `prime_train.smoke.toml` against env `0.1.1` | Completed | Two-step smoke passed rollout logging, optimizer, distribution logging, and final checkpoint write. Cost was about `$0.0007`. |
| `plaj70en51ut15joiol4sx5j` | `prime_train.example.toml` against env `0.1.1` | Completed | 50-step pilot completed cleanly. Cost was about `$0.11`. |

## 50-Step Pilot Summary

- Model: `Qwen/Qwen3.5-2B`.
- Environment: `inference/extension3-agent-loop@0.1.1`.
- Max steps: `50`.
- Batch size: `64`.
- Rollouts per example: `8`.
- Runtime: `8m 16s` in the orchestrator step loop.
- Metric rows: `50`.
- First training metric row: step `0`, reward `0.682982`, eval avg@1 `0.15`.
- Last training metric row: step `49`, reward `0.975`, truncation `0.0`.
- Last target-term coverage: `1.0`.
- Last initial-query gain: `0.9`.
- Last JSON format score: `1.0`.
- Components: orchestrator `COMPLETED`, train env-server `SUCCEEDED`, eval env-server `SUCCEEDED`.
- Prime checkpoint listing: checkpoint `ymmr6zjbr9ea4wczx1nt2w97`, step `25`, status `READY`, size `166.8 MB`.

## Commands

```bash
prime --plain train eval/configs/extension3_prime/prime_train.smoke.toml
prime --plain train eval/configs/extension3_prime/prime_train.example.toml
prime --plain train get plaj70en51ut15joiol4sx5j
prime --plain train checkpoints plaj70en51ut15joiol4sx5j
prime --plain train metrics plaj70en51ut15joiol4sx5j
prime --plain train usage plaj70en51ut15joiol4sx5j
```
