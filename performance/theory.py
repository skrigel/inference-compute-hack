"""
theory.py  --  Closed-form compute models for "Grep for Meaning".

Purpose
-------
Predict the performance curves *before* you measure them, so eval/bench.py can
plot measured-vs-predicted. Every function here is a closed form derived from
hardware constants + model shape; none of it requires running the model.

The four headline artifacts (see make_all_figures):
    1. roofline           -- workload sits on the COMPUTE ceiling, not the BW wall
    2. mfu_waterfall      -- theoretical floor vs achieved, gap = MFU, per rung
    3. area_under_loop    -- cumulative compute over a refine session (the money shot)
    4. compute_vs_churn   -- recompute-over-store vs RAG as a function of data churn

Plus two supporting ones:
    5. kv_capacity        -- warm-KV footprint vs corpus size, marks the KV crossover
    6. suffix_speedup     -- predicted L/s band for suffix-only re-prefill

USAGE
-----
    python performance/theory.py     # prints worked examples, writes performance/figures/*.png+svg
    from theory import HW, MODEL, predict_cold_scan_seconds, ...

!!! VERIFY THESE CONSTANTS ON YOUR ACTUAL 8x H100 NODE BEFORE TRUSTING NUMBERS !!!
nvidia-smi for memory/bandwidth; a microbench for real peak TFLOP/s by precision;
print(model.config) for the true layer/head/dim numbers.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# HARDWARE CONSTANTS  -- H100 SXM, 8-GPU node. EDIT TO MATCH YOUR BOX.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hardware:
    n_gpus: int = 8
    # Dense tensor-core peak per GPU (FLOP/s). Sparsity ~2x these; we use dense.
    bf16_flops: float = 989.5e12       # H100 SXM BF16/FP16 dense
    fp8_flops:  float = 1979.0e12      # H100 SXM FP8 dense (~2x BF16)
    hbm_bw:     float = 3.35e12        # bytes/s per GPU (HBM3, H100 SXM 80GB)
    hbm_bytes:  float = 80e9           # bytes per GPU

    @property
    def node_bf16(self) -> float: return self.n_gpus * self.bf16_flops
    @property
    def node_fp8(self)  -> float: return self.n_gpus * self.fp8_flops
    @property
    def node_hbm(self)  -> float: return self.n_gpus * self.hbm_bytes
    @property
    def node_bw(self)   -> float: return self.n_gpus * self.hbm_bw
    def ridge(self, fp8: bool = False) -> float:
        """Roofline ridge point: arithmetic intensity (FLOP/byte) where
        bandwidth ceiling meets compute ceiling. Right of this = compute-bound."""
        peak = self.fp8_flops if fp8 else self.bf16_flops
        return peak / self.hbm_bw


# ---------------------------------------------------------------------------
# MODEL SHAPE  -- Tier-1 filter, ~8B class. EDIT TO MATCH YOUR MODEL.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    params: float = 8.0e9
    n_layers: int = 32
    n_kv_heads: int = 8          # GQA: KV heads, NOT query heads -- this is what sizes the cache
    head_dim: int = 128
    d_model: int = 4096
    # chunk geometry
    prefix_len: int = 350        # [instruction + chunk] -- the stable, cached prefix
    suffix_len: int = 15         # [predicate] -- the short, changing part
    @property
    def chunk_len(self) -> int: return self.prefix_len + self.suffix_len


HW = Hardware()
MODEL = Model()


# ---------------------------------------------------------------------------
# 1. FLOP ACCOUNTING  (Artifact 2: MFU)
# ---------------------------------------------------------------------------
def scan_flops(n_tokens: float, model: Model = MODEL, include_attn: bool = True) -> float:
    """Forward-pass FLOPs to push n_tokens through the model.

    Dominant term: 2 * P * T (every parameter is one multiply-add per token).
    Attention term: ~4 * n_layers * L^2 * d_model per chunk -- quadratic in L but
    tiny for L~350 (worth showing it's ~1% so nobody accuses you of hand-waving).
    """
    matmul = 2.0 * model.params * n_tokens
    if not include_attn:
        return matmul
    n_chunks = n_tokens / model.chunk_len
    attn = 4.0 * model.n_layers * (model.chunk_len ** 2) * model.d_model * n_chunks
    return matmul + attn


def attn_fraction(model: Model = MODEL) -> float:
    """What fraction of scan FLOPs is attention (the 'negligible' claim, quantified)."""
    per_chunk_matmul = 2.0 * model.params * model.chunk_len
    per_chunk_attn = 4.0 * model.n_layers * (model.chunk_len ** 2) * model.d_model
    return per_chunk_attn / (per_chunk_matmul + per_chunk_attn)


def predict_cold_scan_seconds(n_tokens: float, mfu: float = 0.40,
                              fp8: bool = False, hw: Hardware = HW,
                              model: Model = MODEL) -> float:
    """Theoretical wall-clock for one full cold scan at a given MFU."""
    peak = hw.node_fp8 if fp8 else hw.node_bf16
    return scan_flops(n_tokens, model) / (peak * mfu)


def implied_mfu(n_tokens: float, measured_seconds: float, fp8: bool = False,
                hw: Hardware = HW, model: Model = MODEL) -> float:
    """Invert a measured scan time into an achieved MFU (for bench.py overlay)."""
    peak = hw.node_fp8 if fp8 else hw.node_bf16
    return scan_flops(n_tokens, model) / (measured_seconds * peak)


# ---------------------------------------------------------------------------
# 2. ARITHMETIC INTENSITY / ROOFLINE  (Artifact 1)
# ---------------------------------------------------------------------------
def prefill_intensity(batch_tokens: float, model: Model = MODEL) -> float:
    """Arithmetic intensity (FLOP/byte) of a prefill GEMM batch.

    FLOPs = 2 * P * T ;  weight bytes read once per batch = 2 * P (bf16).
    => intensity ~= T  (tokens per weight load). Large batch -> deep into
    compute-bound territory. This is the whole roofline argument in one line.
    """
    flops = 2.0 * model.params * batch_tokens
    weight_bytes = 2.0 * model.params           # bf16 weights streamed once
    return flops / weight_bytes                 # == batch_tokens, but derived honestly


def achievable_tflops(intensity: float, fp8: bool = False, hw: Hardware = HW) -> float:
    """Roofline: min(compute ceiling, bandwidth * intensity)."""
    peak = hw.node_fp8 if fp8 else hw.node_bf16
    return min(peak, hw.node_bw * intensity)


# ---------------------------------------------------------------------------
# 3. REFINEMENT MODELS  (Artifact 3: area under the loop)
# ---------------------------------------------------------------------------
def suffix_speedup(model: Model = MODEL) -> tuple[float, float]:
    """Predicted speedup band for suffix-only re-prefill.

    Upper (FFN-dominated) bound: L / s.
    Lower bound: corrected for each suffix token still attending over the
    p-length cached prefix (O(s*p) attention work that doesn't disappear).
    Returns (low, high).
    """
    L, s, p = model.chunk_len, model.suffix_len, model.prefix_len
    high = L / s
    # suffix work ~ ffn(s) + attn(s over p); full work ~ ffn(L) + attn(L over L)
    # approximate with token-equivalents:
    suffix_equiv = s + (s * p) / L
    low = L / suffix_equiv
    return low, high


def cumulative_work(turns: int, N: int, rho: float, mode: str,
                    model: Model = MODEL, n_query: int | None = None,
                    index_cost_chunks: float = 0.0, change_turns: tuple = ()):
    """Cumulative compute (in chunk-score units) over a refine session.

    mode='full'    : re-score whole corpus each turn        -> k*N        (linear, steep)
    mode='suffix'  : warm + suffix-only                     -> k*N*(s/L)  (linear, shallow)
    mode='scoped'  : candidate-set scoping                  -> N*(1-rho^k)/(1-rho) (SATURATES)
    mode='rag'     : per-query retrieve + re-index on change-> linear + step jumps

    Returns array of cumulative work after each of `turns` turns.
    """
    k = np.arange(1, turns + 1)
    s_over_L = model.suffix_len / model.chunk_len
    if mode == "full":
        return k * N
    if mode == "suffix":
        return k * N * s_over_L
    if mode == "scoped":
        if abs(1 - rho) < 1e-9:
            return k * N
        return N * (1 - rho ** k) / (1 - rho)
    if mode == "rag":
        nq = n_query if n_query is not None else N
        per_query = np.full(turns, nq, dtype=float)
        idx = np.zeros(turns, dtype=float)
        for t in change_turns:
            if 1 <= t <= turns:
                idx[t - 1] += index_cost_chunks
        return np.cumsum(per_query + idx)
    raise ValueError(mode)


def scoped_asymptote(N: int, rho: float) -> float:
    """Total scoped work converges to N/(1-rho) regardless of turn count."""
    return N / (1 - rho)


# ---------------------------------------------------------------------------
# 4. KV CAPACITY  (Artifact 5: warm-cache crossover)
# ---------------------------------------------------------------------------
def kv_bytes_per_token(model: Model = MODEL, bits: int = 16) -> float:
    """KV cache bytes per token. 2 = (K and V); GQA uses n_kv_heads."""
    bytes_per_elem = bits / 8.0
    return 2.0 * model.n_layers * model.n_kv_heads * model.head_dim * bytes_per_elem


def warm_kv_bytes(n_chunks: float, model: Model = MODEL, bits: int = 16) -> float:
    """Footprint of warming the whole-corpus prefix KV at ingest."""
    return kv_bytes_per_token(model, bits) * model.prefix_len * n_chunks


def kv_crossover_chunks(hw: Hardware = HW, model: Model = MODEL, bits: int = 16) -> float:
    """Corpus size (chunks) at which warm-KV exactly fills node HBM.
    Past this, warm-on-ingest (#3) hands off to candidate-set scoping (#4)."""
    return hw.node_hbm / (kv_bytes_per_token(model, bits) * model.prefix_len)


# ---------------------------------------------------------------------------
# 5. RAG EXCHANGE RATE  (Artifact 4: compute-for-storage)
# ---------------------------------------------------------------------------
@dataclass
class RagModel:
    """Per-event costs in chunk-score-equivalent units (calibrate from bench.py)."""
    c_scan: float          # our cost: full prefill scan of corpus, per query
    c_rag_query: float     # RAG: query-embed + ANN + top-k rerank, per query
    c_index: float         # RAG: (re)build index over the corpus, per data-change event

    def ours_total(self, n_queries: int, n_changes: int) -> float:
        return n_queries * self.c_scan                      # no D term -- nothing derived

    def rag_total(self, n_queries: int, n_changes: int) -> float:
        return n_changes * self.c_index + n_queries * self.c_rag_query

    def breakeven_changes(self, n_queries: int) -> float:
        """Data-change count D below which RAG wins on compute.
        RAG wins while D*c_index < Q*(c_scan - c_rag_query)."""
        margin = self.c_scan - self.c_rag_query
        if margin <= 0:
            return float("inf")     # RAG always cheaper per query on static data -- say so honestly
        return n_queries * margin / self.c_index


def recompute_bandwidth_demand(chunks_per_sec: float, model: Model = MODEL) -> float:
    """Bytes/s of weight traffic to sustain a scan rate (the honest 'next bottleneck').
    Weights stream once per batch; at batch B the per-chunk share is 2P/B, so this is
    a conservative upper bound assuming small batches. Compare to hw.node_bw."""
    tokens_per_sec = chunks_per_sec * model.chunk_len
    # weight bytes per token at large batch -> ~0; this returns the *raw re-stream* floor:
    raw_bytes_per_token = 2.0      # ~2 bytes/token of raw text re-read each pass
    return tokens_per_sec * raw_bytes_per_token


# ===========================================================================
# FIGURES
# ===========================================================================
def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

INK = "#1a1a2e"; OURS = "#e94560"; ALT = "#0f7173"; GREY = "#9a9a9a"; FP8C = "#f2a900"


def fig_roofline(path_stem: str):
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    I = np.logspace(-1, 4, 500)
    bf16 = np.minimum(HW.node_bf16, HW.node_bw * I) / 1e12
    fp8  = np.minimum(HW.node_fp8,  HW.node_bw * I) / 1e12
    ax.plot(I, bf16, color=INK, lw=2.2, label="BF16 roofline (8×H100)")
    ax.plot(I, fp8,  color=FP8C, lw=2.0, ls="--", label="FP8 roofline (2× ceiling)")

    # ridge
    r = HW.ridge()
    ax.axvline(r, color=GREY, ls=":", lw=1.2)
    ax.text(r*1.08, 40, f"ridge ≈ {r:.0f} FLOP/byte", rotation=90,
            va="bottom", ha="left", color=GREY, fontsize=9)

    # ceiling labels
    ax.text(6000, HW.node_bf16/1e12*1.06, "BF16 compute ceiling", ha="right",
            fontsize=8.5, color=INK)
    ax.text(6000, HW.node_fp8/1e12*1.06, "FP8 compute ceiling", ha="right",
            fontsize=8.5, color=FP8C)

    # workload points: (intensity, label, color, marker, fp8, text-offset)
    pts = [
        (1.0,    "generation decode\n(1 tok / weight load)", ALT,  "v", False, (10, -4)),
        (32.0,   "RAG cross-encoder\nrerank (small batch)",  GREY, "s", False, (10, 4)),
        (1500.0, "OUR scan/score\n(batch ≫ ridge,\ncompute-bound)", OURS, "o", False, (-12, -64)),
    ]
    for I0, lab, c, m, use_fp8, off in pts:
        y = achievable_tflops(I0, fp8=use_fp8) / 1e12
        ax.scatter([I0], [y], s=140, color=c, marker=m, zorder=6, edgecolor="white", lw=1.3)
        ax.annotate(lab, (I0, y), textcoords="offset points",
                    xytext=off, fontsize=9, color=c, weight="bold",
                    ha="right" if off[0] < 0 else "left")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / byte)")
    ax.set_ylabel("achievable throughput  (TFLOP/s)")
    ax.set_title("Semantic scoring lives on the compute ceiling,\nnot the bandwidth wall",
                 loc="left", weight="bold")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(5, 30000)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path_stem + ".png"); fig.savefig(path_stem + ".svg")
    plt.close(fig)


def fig_mfu_waterfall(path_stem: str, n_tokens: float = 7e6):
    import matplotlib.pyplot as plt
    _style()
    # Ladder A rungs: theoretical floor at increasing MFU as you stack optimizations.
    # The bar = achieved time; the gap to the dashed theoretical-minimum = MFU loss.
    rungs = ["naive\n(cold, B=1)", "+continuous\nbatching", "+data-parallel\n×8",
             "+FP8 compute", "theoretical\nfloor"]
    mfus  = [0.03, 0.18, 0.38, 0.52, 1.00]
    fp8   = [False, False, False, True, True]
    times = [predict_cold_scan_seconds(n_tokens, m, f) for m, f in zip(mfus, fp8)]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(rungs))
    bars = ax.bar(x, times, color=[GREY, GREY, ALT, OURS, INK], width=0.62,
                  edgecolor="white")
    floor = times[-1]
    ax.axhline(floor, color=INK, ls="--", lw=1.2, alpha=0.7)
    ax.text(len(rungs)-1, floor*1.15, f"FP8 floor ≈ {floor:.1f}s", ha="right",
            fontsize=9, color=INK)
    for xi, (t, m) in enumerate(zip(times, mfus)):
        ax.text(xi, t*1.04, f"{t:.0f}s" if t >= 1 else f"{t:.1f}s", ha="center",
                va="bottom", fontsize=9, weight="bold")
        ax.text(xi, t*0.5, f"MFU\n{m*100:.0f}%", ha="center", va="center",
                fontsize=8.5, color="white", weight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(rungs, fontsize=8.5)
    ax.set_ylabel("cold full-scan time  (s, log)")
    ax.set_title(f"Theoretical floor vs achieved — gap is MFU\n"
                 f"(8B model, {n_tokens/1e6:.0f}M corpus tokens)",
                 loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(path_stem + ".png"); fig.savefig(path_stem + ".svg")
    plt.close(fig)


def fig_area_under_loop(path_stem: str, turns: int = 10, N: int = 20000, rho: float = 0.55):
    import matplotlib.pyplot as plt
    _style()
    k = np.arange(1, turns + 1)
    full   = cumulative_work(turns, N, rho, "full")   / N
    suffix = cumulative_work(turns, N, rho, "suffix")  / N
    scoped = cumulative_work(turns, N, rho, "scoped")  / N
    rag    = cumulative_work(turns, N, rho, "rag", n_query=N,
                             index_cost_chunks=N, change_turns=(4, 8)) / N

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(k, full,   color=GREY, lw=2, marker="o", ms=4, label="full re-score  (k·N)")
    ax.plot(k, rag,    color=INK,  lw=2, marker="D", ms=4,
            label="RAG: re-retrieve + re-index on change")
    ax.plot(k, suffix, color=ALT,  lw=2, marker="^", ms=4,
            label="warm + suffix-only  (k·N·s/L)")
    ax.plot(k, scoped, color=OURS, lw=2.6, marker="s", ms=5,
            label="candidate-set scoped  (saturates)")

    asy = scoped_asymptote(N, rho) / N
    ax.axhline(asy, color=OURS, ls=":", lw=1.3, alpha=0.8)
    ax.text(turns, asy*1.04, f"scoped ceiling = N/(1−ρ) = {asy:.1f}N",
            ha="right", color=OURS, fontsize=9)
    # mark RAG re-index jumps
    for t in (4, 8):
        ax.annotate("re-index", (t, rag[t-1]), textcoords="offset points",
                    xytext=(2, 10), fontsize=8, color=INK)

    ax.set_xlabel("refine turn  k")
    ax.set_ylabel("cumulative compute  (× one full scan)")
    ax.set_title(f"The money shot: area under the refine loop\n(N={N:,}, survivor fraction ρ={rho})",
                 loc="left", weight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlim(1, turns); ax.set_ylim(0, max(full.max(), rag.max())*1.05)
    fig.tight_layout()
    fig.savefig(path_stem + ".png"); fig.savefig(path_stem + ".svg")
    plt.close(fig)


def fig_compute_vs_churn(path_stem: str, n_queries: int = 8):
    import matplotlib.pyplot as plt
    _style()
    # costs in units of one full scan (c_scan = 1.0). Calibrate from bench.py later.
    rag = RagModel(c_scan=1.0, c_rag_query=0.04, c_index=0.9)
    D = np.linspace(0, 12, 200)
    ours = np.full_like(D, rag.ours_total(n_queries, 0))
    rag_tot = rag.c_index * D + n_queries * rag.c_rag_query

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(D, ours, color=OURS, lw=2.6, label="recompute-over-store (ours)")
    ax.plot(D, rag_tot, color=INK, lw=2.2, label="RAG (index + query)")
    be = rag.breakeven_changes(n_queries)
    if np.isfinite(be):
        ax.axvline(be, color=GREY, ls=":", lw=1.3)
        ax.scatter([be], [rag.ours_total(n_queries, 0)], color=GREY, zorder=5, s=60)
        ax.text(be+0.2, ours[0]*1.04, f"break-even\nD ≈ {be:.1f} changes",
                fontsize=9, color=GREY)
    ax.fill_between(D, ours, rag_tot, where=(rag_tot > ours), color=OURS, alpha=0.07)
    ax.annotate("streaming / logs:\nD → ∞, we win by construction",
                (11.5, rag_tot[-1]), textcoords="offset points", xytext=(-150, -10),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.set_xlabel("data-change events per session  D")
    ax.set_ylabel(f"total compute over {n_queries} queries  (× one full scan)")
    ax.set_title("Compute-for-storage exchange rate\nRAG wins only below the break-even churn",
                 loc="left", weight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlim(0, 12); ax.set_ylim(0, max(rag_tot.max(), ours[0])*1.1)
    fig.tight_layout()
    fig.savefig(path_stem + ".png"); fig.savefig(path_stem + ".svg")
    plt.close(fig)


def fig_kv_capacity(path_stem: str):
    import matplotlib.pyplot as plt
    _style()
    N = np.linspace(1000, 60000, 300)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for bits, c, lab in [(16, INK, "FP16 KV"), (8, ALT, "8-bit KV"), (4, OURS, "4-bit KV")]:
        gb = warm_kv_bytes(N, bits=bits) / 1e9
        ax.plot(N/1000, gb, color=c, lw=2.2, label=lab)
        x = kv_crossover_chunks(bits=bits)
        if x < N.max():
            ax.scatter([x/1000], [HW.node_hbm/1e9], color=c, zorder=5, s=55,
                       edgecolor="white")
    ax.axhline(HW.node_hbm/1e9, color=GREY, ls="--", lw=1.4)
    ax.text(1, HW.node_hbm/1e9*1.03, f"node HBM = {HW.node_hbm/1e9:.0f} GB",
            fontsize=9, color=GREY)
    ax.set_xlabel("corpus size  (thousand chunks)")
    ax.set_ylabel("warm-KV footprint  (GB)")
    ax.set_title("Warm-cache capacity & the KV crossover\n"
                 "(intersections = where #3 hands off to #4)",
                 loc="left", weight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlim(0, 60); ax.set_ylim(0, max(warm_kv_bytes(N, bits=16))/1e9*1.05)
    fig.tight_layout()
    fig.savefig(path_stem + ".png"); fig.savefig(path_stem + ".svg")
    plt.close(fig)


def make_all_figures(outdir: str | None = None):
    import os
    if outdir is None:
        outdir = str(Path(__file__).resolve().parent / "figures")
    os.makedirs(outdir, exist_ok=True)
    fig_roofline(f"{outdir}/1_roofline")
    fig_mfu_waterfall(f"{outdir}/2_mfu_waterfall")
    fig_area_under_loop(f"{outdir}/3_area_under_loop")
    fig_compute_vs_churn(f"{outdir}/4_compute_vs_churn")
    fig_kv_capacity(f"{outdir}/5_kv_capacity")
    print(f"wrote 5 figures (png+svg) to {outdir}/")


# ===========================================================================
# WORKED EXAMPLES  (these should match the spec; if they don't, the math moved)
# ===========================================================================
def _worked_examples():
    line = "-" * 70
    print(line); print("WORKED EXAMPLES  (verify constants on real hardware!)"); print(line)

    T = 7e6
    flops = scan_flops(T)
    print(f"\n[1] FLOP accounting  (8B model, {T/1e6:.0f}M corpus tokens)")
    print(f"    scan FLOPs           = {flops:.3e}  = {flops/1e15:.1f} PFLOP")
    print(f"    attention fraction   = {attn_fraction()*100:.2f}%  (negligible, as claimed)")
    print(f"    node BF16 peak       = {HW.node_bf16/1e15:.2f} PFLOP/s")
    for mfu in (0.40, 0.55):
        print(f"    cold scan @ {mfu*100:.0f}% MFU  = {predict_cold_scan_seconds(T, mfu):.1f} s"
              f"   (spec target: 30-60s)")
    print(f"    cold scan @ 55% FP8  = {predict_cold_scan_seconds(T, 0.55, fp8=True):.1f} s")

    print(f"\n[2] Roofline")
    print(f"    BF16 ridge point     = {HW.ridge():.0f} FLOP/byte")
    print(f"    our batch intensity  ~ batch_tokens  ->> ridge  => COMPUTE-bound")

    lo, hi = suffix_speedup()
    print(f"\n[3] Suffix-only re-prefill  (L={MODEL.chunk_len}, s={MODEL.suffix_len})")
    print(f"    predicted speedup    = {lo:.1f}x .. {hi:.1f}x   (lower = attn-corrected)")

    print(f"\n[4] Candidate-set scoping  (N=20k, rho=0.55)")
    print(f"    cumulative ceiling   = N/(1-rho) = {scoped_asymptote(20000, 0.55)/20000:.2f}N"
          f"  (bounded over infinite turns)")

    print(f"\n[5] KV capacity  (prefix_len={MODEL.prefix_len})")
    for bits in (16, 8, 4):
        bpt = kv_bytes_per_token(bits=bits)
        for n in (10000, 20000):
            gb = warm_kv_bytes(n, bits=bits)/1e9
            fit = "fits" if gb <= HW.node_hbm/1e9 else "DOES NOT FIT"
            print(f"    {bits:>2}-bit, {n//1000:>2}k chunks = {gb:6.0f} GB  ({fit} in "
                  f"{HW.node_hbm/1e9:.0f} GB)")
        print(f"        crossover @ {bits}-bit = {kv_crossover_chunks(bits=bits)/1000:.1f}k chunks")

    print(f"\n[6] RAG exchange rate  (c_scan=1, c_rag_query=0.04, c_index=0.9)")
    rag = RagModel(1.0, 0.04, 0.9)
    print(f"    break-even churn (Q=8) = {rag.breakeven_changes(8):.1f} data changes")
    print(f"    -> static data (D=0): RAG cheaper per query (say so). D>break-even: we win.")
    print(line)


if __name__ == "__main__":
    _worked_examples()
    make_all_figures()
