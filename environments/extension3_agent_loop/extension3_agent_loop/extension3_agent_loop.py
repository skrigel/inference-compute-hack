from __future__ import annotations

from typing import Any

import verifiers as vf
from datasets import Dataset

from .retrieval_data import SYSTEM_PROMPT, build_retrieval_rows, score_completion


def load_environment(
    split: str = "train",
    max_examples: int = -1,
    include_hard: bool = True,
    passages_per_prompt: int = 16,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(
        split=split,
        max_examples=max_examples,
        include_hard=include_hard,
        passages_per_prompt=passages_per_prompt,
    )
    rubric = vf.Rubric(
        funcs=[
            target_term_coverage,
            evidence_id_recall,
            hard_negative_rejection,
            exclude_term_use,
            initial_query_gain,
            anti_select_all,
            format_jsonish,
        ],
        weights=[0.30, 0.25, 0.15, 0.10, 0.10, 0.05, 0.05],
    )
    return vf.SingleTurnEnv(
        dataset=dataset if split == "train" else None,
        eval_dataset=dataset if split != "train" else dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        env_id="extension3-agent-loop",
        env_args={
            "split": split,
            "max_examples": max_examples,
            "include_hard": include_hard,
            "passages_per_prompt": passages_per_prompt,
        },
        **kwargs,
    )


def target_term_coverage(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["target_term_coverage"]


def evidence_id_recall(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["evidence_id_recall"]


def hard_negative_rejection(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["hard_negative_rejection"]


def exclude_term_use(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["exclude_term_use"]


def initial_query_gain(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["initial_query_gain"]


def anti_select_all(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["anti_select_all"]


def format_jsonish(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str, **kwargs: Any) -> float:
    return score_completion(completion, answer)["format_jsonish"]


def _build_dataset(split: str, max_examples: int, include_hard: bool, passages_per_prompt: int) -> Dataset:
    rows = build_retrieval_rows(
        split=split,
        max_examples=max_examples,
        include_hard=include_hard,
        passages_per_prompt=passages_per_prompt,
    )
    return Dataset.from_list(rows)
