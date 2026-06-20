# 04 — Constants to verify on the real 8×H100 node

Every figure and worked example regenerates from the `Hardware` and `Model`
dataclasses at the top of `theory.py`. They are **best-estimates**. Fix them once
on the real box and everything updates together. Do this before any number goes on
a slide.

## Hardware (`Hardware` dataclass)
| Field | Assumed | How to verify |
|---|---|---|
| `bf16_flops` | 989.5e12 /GPU | microbench a large dense GEMM; don't trust the datasheet peak blindly |
| `fp8_flops` | 1979e12 /GPU | microbench FP8 GEMM (~2× BF16 expected) |
| `hbm_bw` | 3.35e12 B/s /GPU | `nvidia-smi -q -d MEMORY` / DCGM; confirm SXM vs NVL vs PCIe variant |
| `hbm_bytes` | 80e9 /GPU | `nvidia-smi --query-gpu=memory.total` |
| `n_gpus` | 8 | `nvidia-smi -L` |

Note: the **ridge point** (~295 FLOP/byte) and both roofline ceilings move with
these. The whole "compute-bound" claim depends on the ridge being well left of the
scan's batch intensity — re-check after measuring real peaks.

## Model (`Model` dataclass)
| Field | Assumed | How to verify |
|---|---|---|
| `params` | 8.0e9 | model card / `sum(p.numel() for p in model.parameters())` |
| `n_layers` | 32 | `print(model.config.num_hidden_layers)` |
| `n_kv_heads` | 8 | `model.config.num_key_value_heads` — **GQA: KV heads, not query heads.** This is what sizes the cache; getting it wrong throws the whole KV-capacity figure. |
| `head_dim` | 128 | `hidden_size / num_attention_heads` |
| `d_model` | 4096 | `model.config.hidden_size` |
| `prefix_len` | 350 | tokenize a real `[instruction + chunk]` and count |
| `suffix_len` | 15 | tokenize a real predicate and count |

## Calibrate from `eval/bench.py` (not hardware constants, but needed for figures)
| Quantity | Used by | How |
|---|---|---|
| `c_scan` | compute-vs-churn | time/energy of one full corpus scan |
| `c_rag_query` | compute-vs-churn | time/energy of one embed + ANN + rerank |
| `c_index` | compute-vs-churn | time/energy of one index build over the corpus |
| `rho` (ρ) | area-under-loop | measured survivor fraction per refine turn (per canonical session) |
| achieved MFU per rung | mfu_waterfall | `theory.implied_mfu(n_tok, measured_seconds)` |

## Sanity checks after editing
Run `python theory.py` and confirm:
- cold scan @ 40% MFU stays inside the §11 30–60 s window (or update the spec target);
- attention fraction stays ~1% (if it jumps, your chunk got long — revisit batching);
- the FP16 KV crossover prints a chunk count; compare it to your real OOM point in
  the scaling sweep. A large gap = prefix-dedup benefit, worth reporting.

## Reminder on the quantization split (artifact 6)
When you set up the ladder, keep two **separate** levers:
- 4-bit **weights** → memory/capacity (bigger warm cache, more replicas), *not* scan speed.
- FP8 **compute** → prefill throughput (raises the compute ceiling).
Don't collapse them into one "4-bit → 4× throughput" rung — it's wrong in this
compute-bound regime and a judge will catch it.
