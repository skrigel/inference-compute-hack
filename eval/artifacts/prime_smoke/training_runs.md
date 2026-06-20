# Prime Hosted Training Runs

This ledger records the first Prime Hosted Training validation runs for the
Extension 3 query-refinement environment.

## Environment Versions

- `inference/extension3-agent-loop@0.1.0`: eval worked, but hosted training
  failed when Prime tried to serialize rollout samples containing dict answers.
- `inference/extension3-agent-loop@0.1.1`: stores `answer` as JSON text and
  parses it inside reward functions. This is the current passing version.
- `inference/extension3-agent-loop@0.2.0`: expands the task distribution to
  1,152 train rows and 288 eval rows across 12 retrieval-style domains, with
  positive evidence ids, hard negative ids, exclusion terms, and longer prompts.

## Runs

| Run ID | Config | Status | Notes |
|---|---|---:|---|
| `mkyr5jlfznstwd5cnhqhkhdf` | `prime_train.example.toml` against env `0.1.0` | Failed | Reached step 0, then `pyarrow.lib.ArrowTypeError: Expected bytes, got a 'dict' object` in Prime sample logging. Cost was about `$0.0016`. |
| `pp76jxcpojf2t5uvipy5tf8j` | `prime_train.smoke.toml` against env `0.1.1` | Completed | Two-step smoke passed rollout logging, optimizer, distribution logging, and final checkpoint write. Cost was about `$0.0007`. |
| `plaj70en51ut15joiol4sx5j` | `prime_train.example.toml` against env `0.1.1` | Completed | 50-step pilot completed cleanly. Cost was about `$0.11`. |
| `ap08781gs4mj999nhhe1th09` | `prime_train.full.toml` against env `0.2.0` | Stopped | Stopped at step `26` and `$5.29` after confirming training quality was healthy but periodic heldout eval cadence was misconfigured. Step 0 eval reward was `0.9049`; step 25 train reward was `0.9619`; errors and truncation were `0.0%`. |
| pending | `prime_train.full.toml` against env `0.2.0` | Planned | Corrected budgeted 35B run. The config now uses `[eval] interval = 25` plus `[[eval.env]]`. Because run `ap08781gs4mj999nhhe1th09` already spent `$5.29`, target new-run spend is about `$55` with `python -m eval.prime_budget_monitor <run-id> --target-cost 55 --hard-limit 60 --poll-seconds 15`. |

## 35B Env 0.2.0 Baseline Smoke

- Model: `Qwen/Qwen3.5-35B-A3B`.
- Environment: `inference/extension3-agent-loop@0.2.0`.
- Eval command shape: `prime eval run ... --sampling-args '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}},"max_tokens":256,"temperature":0}'`.
- Result path: `eval/artifacts/prime_smoke/eval_baseline_v020_35b_chat_template_no_think`.
- Result: `4` eval examples, average reward `0.920`, average target-term coverage `0.958`, evidence-id recall `1.000`, hard-negative rejection `1.000`, JSON format score `1.000`, truncation `0.000`.
- Root-cause note: leaving Qwen thinking enabled returns `reasoning_content` with `content=null` under Prime eval, so the reward sees no final JSON. Passing `enable_thinking` as a bare sampling field also fails through the local OpenAI client; the working request path is `extra_body.chat_template_kwargs.enable_thinking=false`.

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
prime --plain train eval/configs/extension3_prime/prime_train.full.toml
python -m eval.prime_budget_monitor <run-id> --target-cost 60 --hard-limit 65 --poll-seconds 15
```
