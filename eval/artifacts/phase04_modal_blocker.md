# Phase 04 Modal Validation Blocker

Status: real Modal/H100 validation is blocked by the Modal workspace spend limit.

Attempted command:

```bash
ENABLE_MFU_METRICS=1 GPU_MEMORY_UTILIZATION=0.92 KV_CACHE_DTYPE=auto modal run inference/modal_app.py::test
```

Modal auth succeeded for workspace `joshveer`, and earlier runs reached H100/vLLM startup.

Issues found and fixed before the spend-limit blocker:

- Modal 1.5 compatibility: replaced deprecated `container_idle_timeout` with `scaledown_window`.
- Modal 1.5 compatibility: replaced deprecated `allow_concurrent_inputs` with `@modal.concurrent(...)`.
- Removed forced `VLLM_ATTENTION_BACKEND=FLASHINFER`; the image did not install FlashInfer, so forcing it killed startup.
- Pinned `transformers>=4.51.1,<5`; `vllm==0.8.5` resolved to `transformers 5.x`, which broke Qwen tokenizer startup.
- Parameterized `KV_CACHE_DTYPE`; `fp8` caused a Flash Attention dtype mismatch in this Modal stack, so rerun with `KV_CACHE_DTYPE=auto`.
- Parameterized `GPU_MEMORY_UTILIZATION`; run with `GPU_MEMORY_UTILIZATION=0.92` and sweep `0.80,0.85,0.90,0.92,0.95` after spend limit is cleared.

Observed caveat:

- `enable_mfu_metrics` is requested, but `vllm==0.8.5` rejected the Python API kwarg. The direct cluster `inference/serve.sh` still passes `--enable-mfu-metrics` to the vLLM server CLI; Modal Python API health reports MFU inactive when the kwarg is rejected.

Next unblock step:

1. Raise/reset Modal spend limit or run on the GPU cluster.
2. Rerun the command above.
3. If green, run:

```bash
SCORER_BACKEND=modal python -m eval.bench --backend modal --gate-only --weave
SCORER_BACKEND=modal python -m eval.bench --backend modal --tag freeze --weave
```

Do not claim Phase 04 complete until the real scorer gate/freeze artifacts are regenerated from a successful real backend run.
