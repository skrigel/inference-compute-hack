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
| `wf0d5vznfpewcstt2jp6dlmi` | `prime_train.full.toml` against env `0.2.0` | Stopped | Corrected budgeted 35B run. Stopped manually at step `225` and `$43.85` new-run spend after heldout reward flattened from `0.9878` at step 175 to `0.9881` at step 225 while zero-advantage filtering rose to `25%`. Combined with stopped run `ap08781gs4mj999nhhe1th09`, total Prime training spend was about `$49.14`, leaving more budget for benchmarking/comparisons. |

## 35B Env 0.2.0 Baseline Smoke

- Model: `Qwen/Qwen3.5-35B-A3B`.
- Environment: `inference/extension3-agent-loop@0.2.0`.
- Eval command shape: `prime eval run ... --sampling-args '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}},"max_tokens":256,"temperature":0}'`.
- Result path: `eval/artifacts/prime_smoke/eval_baseline_v020_35b_chat_template_no_think`.
- Result: `4` eval examples, average reward `0.920`, average target-term coverage `0.958`, evidence-id recall `1.000`, hard-negative rejection `1.000`, JSON format score `1.000`, truncation `0.000`.
- Root-cause note: leaving Qwen thinking enabled returns `reasoning_content` with `content=null` under Prime eval, so the reward sees no final JSON. Passing `enable_thinking` as a bare sampling field also fails through the local OpenAI client; the working request path is `extra_body.chat_template_kwargs.enable_thinking=false`.

## 35B Budgeted Run Summary

- Run ID: `wf0d5vznfpewcstt2jp6dlmi`.
- Model: `Qwen/Qwen3.5-35B-A3B`.
- Environment: `inference/extension3-agent-loop@0.2.0`.
- Config: `eval/configs/extension3_prime/prime_train.full.toml`.
- Status: stopped manually at step `225`.
- New-run cost: `$43.85` (`$33.26` training, `$7.52` inference input, `$3.08` inference output).
- Prior stopped setup-validation run cost: `$5.29`.
- Combined Prime training cost for this 35B attempt: about `$49.14`.
- Stop rationale: heldout reward was still rising through step 175 but flattened after that, while saturation signals increased. Step 225 improved only `+0.00015` over step 200, with `25%` zero-advantage filtering and `25%` solve-all, so more training was unlikely to buy enough quality to justify spending toward the original hard cap.
- Cleanup note: `prime train stop --force wf0d5vznfpewcstt2jp6dlmi` stopped the run successfully. The forced cleanup log reported `HTTP 401` while submitting Prime's final summary, but metrics, usage, progress, and checkpoints remained queryable from the CLI.

### Heldout Eval Milestones

| Step | Heldout avg@1 | Error | Truncation | Evidence recall | Hard-negative rejection | Initial-query gain | JSON format | Zero-advantage filter | Solve-all |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9372 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.6429 | 1.0000 | 0.0000 | 0.0000 |
| 25 | 0.9427 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.7018 | 1.0000 | 0.0000 | 0.0000 |
| 50 | 0.9483 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.7820 | 1.0000 | 0.0625 | 0.0625 |
| 75 | 0.9605 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.7246 | 1.0000 | 0.0000 | 0.0000 |
| 100 | 0.9775 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.8177 | 1.0000 | 0.0625 | 0.0625 |
| 125 | 0.9809 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.7437 | 1.0000 | 0.0625 | 0.0625 |
| 150 | 0.9853 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.8427 | 1.0000 | 0.1250 | 0.1250 |
| 175 | 0.9878 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.8064 | 1.0000 | 0.0625 | 0.0625 |
| 200 | 0.9880 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.8591 | 1.0000 | 0.1875 | 0.1875 |
| 225 | 0.9881 | 0.0 | 0.0 | 1.0000 | 1.0000 | 0.8156 | 1.0000 | 0.2500 | 0.2500 |

### Ready Checkpoints

| Checkpoint ID | Step | Status | Size |
|---|---:|---|---:|
| `ijp6gp6ipyg887v1syiwatbe` | 25 | READY | 9.4 GB |
| `pzrf80fnqim75800ey902zzs` | 50 | READY | 9.4 GB |
| `f22ividj6gyutw2xqrwt266e` | 75 | READY | 9.4 GB |
| `uh0292xgiali2mopfqdbudwm` | 100 | READY | 9.4 GB |
| `kbl71djuoow7hvv20ijkve1e` | 125 | READY | 9.4 GB |
| `zjxzx2x2scyxaglu2kxi6s7z` | 150 | READY | 9.4 GB |
| `lb4gs2o7rxugvqgwqs4p0qtb` | 175 | READY | 9.4 GB |
| `vnzrv0ha2afgas4letsk1kqy` | 200 | READY | 9.4 GB |

The step `225` checkpoint save was logged before manual stop, but it was not yet listed by `prime train checkpoints` when final artifacts were collected. Use step `200` as the latest confirmed ready checkpoint unless the Prime checkpoint list later shows the step `225` artifact.

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
