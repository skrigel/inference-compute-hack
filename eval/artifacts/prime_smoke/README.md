# Prime Smoke Eval

This directory records the first successful Prime-backed smoke eval for the
Extension 3 query-refinement environment before launching post-training.

## Command

```bash
prime eval run inference/extension3-agent-loop \
  --provider prime \
  --model Qwen/Qwen3.5-2B \
  --num-examples 2 \
  --rollouts-per-example 1 \
  --max-tokens 128 \
  --temperature 0 \
  --output-dir eval/artifacts/prime_smoke/eval_baseline \
  --save-results \
  --skip-upload \
  --disable-tui \
  --abbreviated-summary \
  --timeout 120 \
  --env-args '{"split":"eval","max_examples":2,"include_hard":true}'
```

## Result

- Status: passed with exit code 0.
- Environment: `inference/extension3-agent-loop` version `0.1.0`.
- Model: `Qwen/Qwen3.5-2B`.
- Examples: 2.
- Rollouts per example: 1.
- Average reward: `0.6416666667`.
- Target-term coverage: `0.6666666667`.
- Initial-query gain: `0.3000000000`.
- Anti-select-all: `1.0`.
- JSON format score: `1.0`.
- Average input tokens: `149`.
- Average output tokens: `18`.
- Eval runtime reported by Verifiers: `3.79s`.

Local result files are under
`eval/artifacts/prime_smoke/eval_baseline/evals/extension3-agent-loop--Qwen--Qwen3.5-2B/0b4d1de2/`.
