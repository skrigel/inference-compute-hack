# Prime Smoke Eval

This directory records Prime-backed smoke evals for the Extension 3
query-refinement environment before launching post-training.

The current passing environment version is `inference/extension3-agent-loop`
`0.1.1`. Version `0.1.1` stores dataset answers as JSON strings so Prime
Hosted Training can serialize rollout samples to Parquet while reward functions
still parse the same answer payload for scoring.

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

## Current Result

- Status: passed with exit code 0.
- Environment: `inference/extension3-agent-loop` version `0.1.1`.
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
- Eval runtime reported by Verifiers: `3.28s`.

Local result files are under
`eval/artifacts/prime_smoke/eval_baseline_v011/evals/extension3-agent-loop--Qwen--Qwen3.5-2B/e4f0d9c9/`.

The earlier `eval_baseline` artifact is retained as the first successful eval
against environment version `0.1.0`; it passed eval but the first hosted
training attempt showed that dict answers were incompatible with Prime's rollout
sample logger.
