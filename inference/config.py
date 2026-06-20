from __future__ import annotations

import os

from inference.mock_scorer import MockScorer
from inference.scorer import ScorerClient


def make_scorer() -> ScorerClient:
    """
    Factory for ScorerClient implementations.

    Backends:
    - mock: GPU-free deterministic scorer for local dev (default)
    - modal: Modal-deployed vLLM on 6× H100 GPUs
    - vllm: Direct vLLM connection (for on-prem H100 box)

    Set via SCORER_BACKEND environment variable.
    """
    backend = os.environ.get("SCORER_BACKEND", "mock").lower()

    if backend == "mock":
        return MockScorer()

    if backend == "modal":
        from inference.modal_client import ModalScorerAsync
        return ModalScorerAsync()

    if backend == "vllm":
        from inference.vllm_scorer import VLLMScorer
        return VLLMScorer.from_env()

    raise ValueError(f"Unknown SCORER_BACKEND={backend!r}; expected 'mock', 'modal', or 'vllm'")
