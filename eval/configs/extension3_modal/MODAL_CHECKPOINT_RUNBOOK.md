# Extension 3 Modal Checkpoint Runbook

This packet is for the Modal post-training path when only an $18 budget is available.
It is intentionally checkpoint-heavy and budget-capped. It does not launch GPU training by itself.

## Budget

- GPU request: `H100:6`.
- H100 price assumption: `$0.001097` per GPU-second.
- Full-budget wall time: `2734` seconds.
- Planned training window: `2187` seconds / `36.45` minutes.
- Planned GPU-only cost: `$14.39`.
- Reserve: `$3.61` for CPU, memory, startup, checkpoint writes, eval, and pricing drift.

## Checkpoint Policy

- Save every `180` seconds or `25` optimizer steps, whichever comes first.
- Save the first checkpoint after `5` optimizer steps.
- Keep last `4` checkpoints plus best checkpoint.
- Update `latest.json` only after checkpoint state is fully written.
- Resume from `latest.json` by default.

## No-Credit Dry Run

```bash
python -m eval.agent_loop_modal_checkpoints --output-dir eval/artifacts/modal_checkpoints --total-steps 12 --save-every-steps 3
```

This writes fake checkpoint metadata locally and proves the resume pointer path without using GPUs.

## Paid Modal Stages

1. Stage 0: Run local checkpoint dry-run and Modal CPU smoke only.
2. Stage 1: Optional paid 6-H100 run capped at first checkpoint and <= $2 GPU estimate.
3. Stage 2: Full $18 budget-capped run after stage 1 produces a resumable checkpoint.

## Stop Conditions

- estimated spend reaches 80% of budget
- no checkpoint has been written within the first 6 minutes
- reward is NaN or constant for two checkpoint windows
- selected bytes trend toward full-corpus selection
- latest checkpoint cannot be loaded in a resume dry run

## Future Paid Launch Shape

```bash
modal run inference/modal_app.py::extension3_checkpointed_posttrain \
  --config eval/configs/extension3_modal/modal_6h100_checkpoint_train.example.toml
```

The launch entrypoint is intentionally not implemented yet. Add it only after the dry run passes, the Modal account has budget, and the user approves the paid run.
