# 01 — Optimization artifacts (the math)

Six theoretical artifacts. Each has: the claim, the closed form, the graph it
produces, and where it lands in the spec. All are implemented in `theory.py`;
function names are noted inline.

---

## 1. Roofline — the backbone framing
**Claim.** Single-token scoring is compute-bound and sits near the flat compute
ceiling; generation serving is bandwidth-bound and sits on the diagonal wall.

**Math.** Arithmetic intensity of a prefill GEMM batch:
```
FLOPs        = 2 · P · T_batch
weight bytes = 2 · P            (bf16 weights streamed once per batch)
intensity I  = FLOPs / bytes = T_batch       (tokens per weight load)
```
Roofline ridge (where bandwidth ceiling meets compute ceiling):
```
ridge = peak_FLOPs / HBM_bandwidth ≈ 990e12 / 3.35e12 ≈ 295 FLOP/byte
```
With a large batch, `I = T_batch ≫ 295` → firmly compute-bound. Achievable
throughput = `min(compute_ceiling, bandwidth · I)`.

**Graph.** Roofline (log–log): x = intensity, y = TFLOP/s. Two ceilings (BF16
~7.9 PFLOP/s node, FP8 ~15.8), the ridge line, and three points — generation
decode (far left, BW-bound), RAG rerank (middle), our scan (far right, on the
compute ceiling).

**`theory.py`:** `prefill_intensity`, `achievable_tflops`, `Hardware.ridge`,
`fig_roofline`. **Spec:** new lead slide before §11.

---

## 2. FLOP accounting → MFU
**Claim.** Replace "measure, don't assume" throughput targets with a theoretical
anchor, and report **MFU = achieved / theoretical-peak** — the number that signals
you understand the hardware.

**Math.** Forward-pass FLOPs for `T` tokens through a `P`-param decoder:
```
FLOPs_scan ≈ 2 · P · T          (matmul-dominated)
   + attention ≈ 4 · n_layers · L² · d_model  per chunk   (≈1% at L≈350 — negligible)
```
Worked example (8B model, ~7M corpus tokens):
```
FLOPs ≈ 2 · 8e9 · 7e6 ≈ 1.13e17 = 113 PFLOP
8×H100 BF16 peak ≈ 7.9 PFLOP/s
@ 40% MFU → 3.2 PFLOP/s effective → cold full scan ≈ 36 s   ✓ inside §11's 30–60 s
```
The math **closes** with the spec's own target — show that. Prefill at large
batch genuinely hits 40–55% MFU on H100, so this reports as strong, not
embarrassing.

**Graph.** Per-rung bars (Ladder A): theoretical floor vs achieved time, gap
labeled as MFU.

**`theory.py`:** `scan_flops`, `attn_fraction`, `predict_cold_scan_seconds`,
`implied_mfu`, `fig_mfu_waterfall`. **Spec:** §11 Ladder A.

---

## 3a. Suffix-only re-prefill — predicted, not just measured
**Claim.** Refining re-prefills only the short predicate against cached prefix KV.

**Math.** Full re-score prefills `L` tokens/chunk; suffix-only prefills `s`. The
FFN/projection FLOPs (dominant) scale with token count:
```
speedup ≈ L / s              (e.g. L=365, s=15 → ~24×, upper bound)
```
Lower bound corrects for each suffix token still attending over the `p`-length
cached prefix (`O(s·p)` attention that doesn't disappear): predicted band ≈
**12×–24×**. Plot measured against the derived band, not a bare ratio.

**`theory.py`:** `suffix_speedup`. **Spec:** §6 #2, §9 "suffix-only vs full".

## 3b. Candidate-set scoping — the "area under the loop"
**Claim.** The single graph that wins the room.

**Math.** With survivor fraction `ρ` per turn, turn-*k* re-scores `ρ^(k-1)·N`
chunks. Cumulative work over a *k*-turn session is geometric and **bounded**:
```
Σ_{i=1}^{k} ρ^(i-1)·N = N·(1 − ρ^k)/(1 − ρ)  →  N/(1−ρ)   as k→∞
```
At ρ=0.55 the ceiling is **2.2N** regardless of how many turns. Four lines:
```
full re-score :  k·N                       (linear, steep)
suffix warm   :  k·N·(s/L)                  (linear, shallow)
scoped        :  N·(1−ρ^k)/(1−ρ)            (SATURATES — flattens)
RAG           :  k·N_query + re-index steps (linear + step jumps on data change)
```
**The visual punchline:** our cost flattens while RAG climbs and step-jumps every
time data changes — §3a/§3b made into one picture.

**`theory.py`:** `cumulative_work`, `scoped_asymptote`, `fig_area_under_loop`.
**Spec:** top of §9 ("money shot is iteration, measure the area under the loop").

---

## 4. KV capacity — a real model, because it's load-bearing
**Claim.** The "10–20k items warm-cache in 640 GB" assertion (§4/§6) needs
checking; at the top of the range it forces a precision decision.

**Math.**
```
bytes/token = 2 · n_layers · n_kv_heads · head_dim · bytes_per_elem   (GQA: KV heads!)
warm_KV     = bytes/token · mean_prefix_len · N_chunks
```
8B-class (32 layers, 8 KV heads, head_dim 128), prefix_len 350:
```
FP16 : 128 KiB/token →  20k chunks ≈ 918 GB   ✗ does NOT fit 640
4-bit:  32 KiB/token →  20k chunks ≈ 229 GB   ✓
FP16 :                  10k chunks ≈ 459 GB   ✓
```
**Correction to §4/§6:** FP16 warm-KV crosses 640 GB at **~14k chunks**, so at the
top of the 10–20k range **4-bit KV is required, not optional.** Own this on your
own slide.

**Graph.** Warm-KV footprint vs corpus size, three lines (FP16/8-bit/4-bit) + a
640 GB line; the intersections *are* the KV crossover — derive it analytically,
confirm in the scaling study.

**`theory.py`:** `kv_bytes_per_token`, `warm_kv_bytes`, `kv_crossover_chunks`,
`fig_kv_capacity`. **Spec:** §9 scaling-study "mark the KV crossover" row.

---

## 5. RAG comparison as a compute-for-storage exchange rate
**Claim.** Make §3b quantitative instead of rhetorical.

**Math.** Over a session of `Q` queries with `D` data-change events:
```
RAG total  = D · C_index + Q · C_rag_query     (C_rag_query = embed + ANN + rerank)
Ours total = Q · C_scan                         (no index, no D term)
```
- **Break-even churn:** RAG wins only while `D · C_index < Q · (C_scan − C_rag_query)`.
  Solve for `D*`. For static data (`D=0`) RAG wins per-query compute — *say so,
  it's honest and costs nothing.* The break-even `D` is small; for streaming/logs
  (§14) `D→∞`, so we win by construction.
- **Bandwidth floor (honest next bottleneck):** compute the HBM bandwidth needed
  to sustain target chunks/sec; show it sits *under* 3.35 TB/s — we're
  compute-bound with headroom, which is *why* recompute is viable.

**Graph.** Total compute vs data-change rate; two lines crossing at `D*`; shade
the region where recompute wins.

**`theory.py`:** `RagModel` (`ours_total`, `rag_total`, `breakeven_changes`),
`recompute_bandwidth_demand`, `fig_compute_vs_churn`. **Spec:** §3b, §14.

---

## 6. Quantization claim — correct it before a judge does
**Claim (corrected).** "4-bit → ~4× throughput" is **wrong** in the compute-bound
prefill regime.

**Why.** AWQ/GPTQ are *weight-only* (w4a16): weights stored 4-bit, but the matmul
dequantizes and runs in BF16, so it does **not** raise tensor-core FLOP/s. The 4×
is a *memory-traffic* win — it pays off in the bandwidth-bound *decode* regime,
not our scan.

**Accurate (and more impressive) framing — split the ladder rung:**
- **4-bit weights → 4× memory** → buys a *bigger warm cache and more replicas*,
  not raw scan speed. (Capacity lever.)
- **FP8 compute on H100 → ~2× BF16 tensor-core** → the real *prefill throughput*
  lever; doubles the compute ceiling (the second roofline in artifact 1).
  (Throughput lever.)

Report them on separate axes/rungs. **Spec:** §5/§6 quantization line, §11 ladder.

---

## 7. Quality calibration — so "fast" isn't "fast and wrong"
**Claim.** Make §12's "don't optimize the speed of being wrong" a metric. The
score is a renormalized two-way softmax `P(Yes)/(P(Yes)+P(No))` — a Bernoulli
probability, directly analyzable.

- **ROC / PR + AUC** vs gold labels; set the threshold-handle default from a
  target-precision operating point, not by eyeballing the histogram. The
  on-histogram threshold drag becomes "sliding along the ROC curve."
- **ECE / reliability diagram** — does score 0.8 mean 80% precision? Raw LLM
  logprobs are usually miscalibrated; if so, that *motivates* the Tier-2 cascade
  as the calibration fix for the uncertain band (ties §6's stretch to a measured
  deficiency, not a guess).

Near-zero added cost — you already compute P/R/F1 vs gold in §9. **Spec:** §9
quality row, §6 cascade, §1 threshold handle.

---

## The four artifacts that most change how the demo reads
1. **roofline** (artifact 1)
2. **MFU theoretical-vs-achieved bars** (artifact 2)
3. **cumulative-work "area under the loop"** (artifact 3b)
4. **compute-vs-data-churn break-even** (artifact 5)

All derive from numbers `eval/bench.py` already plans to log, plus a handful of
hardware constants (see `04_constants_to_verify.md`).
