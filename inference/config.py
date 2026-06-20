from __future__ import annotations

import os

from inference.mock_scorer import MockScorer
from inference.scorer import ScorerClient


def make_scorer() -> ScorerClient:
    backend = os.environ.get("SCORER_BACKEND", "mock").lower()
    if backend == "mock":
        return MockScorer()
    if backend == "vllm":
        raise NotImplementedError("VLLMScorer is introduced after the Phase 0 mock contract is stable")
    raise ValueError(f"Unknown SCORER_BACKEND={backend!r}; expected 'mock' or 'vllm'")
