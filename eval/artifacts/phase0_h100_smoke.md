# Phase 0 H100 Smoke Status

Status: deferred to Phase 04.

This Phase 0 branch was implemented and verified on the local Mac path with
`SCORER_BACKEND=mock`. The local environment does not expose the 8x H100 node, so
the vLLM Yes/No logprob smoke and prefix-cache token-prefix verification were not
run here.

Phase 04 must verify:

- vLLM returns constrained Yes/No logprobs with `max_tokens=1` and `logprobs=20`.
- Two predicates over the same chunk share the same tokenized
  `[instruction + chunk]` prefix.
- Real and padded token counts are recorded separately.
- Prefix length, suffix length, model id, and hardware constants are recorded
  before any measured performance numbers are put on slides.

Until then, candidate-set scoping remains the primary refine-performance
mechanism and warm-prefix reuse is treated as a measured bonus.
