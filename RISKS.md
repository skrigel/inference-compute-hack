# Risk Register

| Risk | Phase | Impact | Mitigation |
|---|---|---|---|
| Contract drift across owners | 00 | Parallel work breaks at integration | Freeze `CONTRACTS.md` first; one source of truth for scorer, chunk, SSE, ops, env vars |
| Prefix-cache assumption fails | 00/04 | Warm suffix story weakens | Treat candidate-set scoping as primary; measure prefix reuse before claiming it |
| Score quality below gate | 04 | Fast but wrong | Validate F1/AUC before speed sweeps; swap model or narrow demo predicates |
| Cold baseline contaminated by warm cache | 00 | MFU and ladder numbers invalid | Capture cold floor first; reset cache or use fresh corpus ids |
| Threshold drag accidentally calls backend | 01 | Zero-inference claim breaks | Add a spy/test around adapter calls; recut from client score cache |
| Refine classifier misroutes phrase | 02 | Demo confusing | Rules-first classifier for scripted phrases; low-confidence removable chips |
| Scoped clause recompute has logic bug | 02 | Wrong survivors | Keep `REFINE_MODE=full` correctness fallback; compare scoped trace against full trace |
| vLLM/AWQ setup eats Phase 04 | 04 | Live path unavailable | Keep mock/replay demo path; use real box only for verified metrics |
| Quantization overclaim | 04/05 | Judge catches bad performance framing | Separate 4-bit weight capacity from FP8 throughput in `METRICS.md` and slides |
| KV cache exceeds HBM | 04 | Warm path fails at 20k chunks | Use `performance/docs/04_constants_to_verify.md`; mark crossover and hand off to scoping |
| Demo scope creep | 03/05 | Core loop gets brittle | Enforce H14 cut line; defer Tier-2 and second domain |
| Replay provenance unclear | 05/06 | Fallback looks fake | Label fixtures by backend, model, corpus, commit, and run time |

## Highest-Leverage Controls

- Contract huddle before feature code.
- Mock-first implementation that shares schemas with real scorer path.
- Performance counters in the cache path from day one.
- Real H100 verification before any slide number is frozen.
- Rehearsed fallback ladder: vLLM -> mock -> canned SSE -> manual staged loop.
