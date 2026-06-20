from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeRegime:
    scoped: bool = True
    score_cache: bool = True
    warm_kv: bool = True
    prefix_caching: bool = True

    @classmethod
    def baseline(cls) -> "ComputeRegime":
        return cls(scoped=False, score_cache=False, warm_kv=False, prefix_caching=False)

    @classmethod
    def warm(cls) -> "ComputeRegime":
        return cls(scoped=False, score_cache=False, warm_kv=True, prefix_caching=True)

    @classmethod
    def scoped(cls) -> "ComputeRegime":
        return cls(scoped=True, score_cache=False, warm_kv=True, prefix_caching=True)

    @classmethod
    def cached(cls) -> "ComputeRegime":
        return cls(scoped=True, score_cache=True, warm_kv=True, prefix_caching=True)


REGIMES: dict[str, ComputeRegime] = {
    "B0_baseline": ComputeRegime.baseline(),
    "B1_warm": ComputeRegime.warm(),
    "B2_scoped": ComputeRegime.scoped(),
    "B3_cached": ComputeRegime.cached(),
}

THROUGHPUT_LADDER: tuple[str, ...] = (
    "A0_baseline",
    "A1_batching",
    "A2_replicas",
    "A3_fp8_compute",
)

CAPACITY_LADDER: tuple[str, ...] = (
    "C0_baseline",
    "C1_4bit_weights_kv",
)
