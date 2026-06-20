from __future__ import annotations

import re
from typing import Any

import verifiers as vf
from datasets import Dataset


TOPICS = {
    "retry_backoff": {
        "initial": "retry",
        "target": "networking layer retry without exponential backoff",
        "terms": ["networking", "retry", "backoff"],
        "positives": [
            "The networking layer retries transient HTTP failures but never applies exponential backoff.",
            "Connection pool retry logic in the networking stack skips exponential backoff under load.",
        ],
        "distractors": [
            "A background worker retries failed tasks with capped exponential backoff.",
            "The database session retries a deadlock and applies exponential backoff.",
        ],
    },
    "ir_retrieval": {
        "initial": "retrieval",
        "target": "information retrieval ranking metrics for code search",
        "terms": ["information", "retrieval", "ranking", "code", "search"],
        "positives": [
            "A paper studies information retrieval ranking metrics for code search.",
            "Neural retrieval for code search compares ranking metrics and semantic quality.",
        ],
        "distractors": [
            "A backend cache stores relevance scores for threshold dragging.",
            "A frontend component renders corpus cards and filters.",
        ],
    },
    "cache_threshold": {
        "initial": "cache",
        "target": "cached threshold drag without rescoring",
        "terms": ["cache", "threshold", "drag", "rescoring"],
        "positives": [
            "Histogram brushing updates the threshold without rescoring cached chunks.",
            "The results cache lets threshold drag move instantly with zero inference.",
        ],
        "distractors": [
            "A paper studies semantic search ranking metrics.",
            "A networking adapter retries transient HTTP failures.",
        ],
    },
}


SYSTEM_PROMPT = """You refine search queries for a dynamic corpus.
Return only JSON:
{"refined_query": "...", "stop": true}
The refined query should recover positive evidence while avoiding broad
queries that select every chunk."""


def load_environment(
    split: str = "train",
    max_examples: int = -1,
    include_hard: bool = True,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(split=split, max_examples=max_examples, include_hard=include_hard)
    rubric = vf.Rubric(
        funcs=[
            target_term_coverage,
            initial_query_gain,
            anti_select_all,
            format_jsonish,
        ],
        weights=[0.55, 0.25, 0.15, 0.05],
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
        },
        **kwargs,
    )


def target_term_coverage(completion: list[dict[str, Any]] | str, answer: dict[str, Any], **kwargs: Any) -> float:
    text = _completion_text(completion)
    terms = [str(term).lower() for term in answer["target_terms"]]
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)


def initial_query_gain(completion: list[dict[str, Any]] | str, answer: dict[str, Any], **kwargs: Any) -> float:
    text = _completion_text(completion)
    initial_terms = set(_tokens(answer["initial_query"]))
    target_terms = set(_tokens(answer["target_query"]))
    added = target_terms - initial_terms
    if not added:
        return 0.0
    return sum(1 for term in added if term in text) / len(added)


def anti_select_all(completion: list[dict[str, Any]] | str, **kwargs: Any) -> float:
    text = _completion_text(completion)
    broad_markers = [
        "all chunks",
        "everything",
        "entire corpus",
        "select all",
        "any document",
    ]
    if any(marker in text for marker in broad_markers):
        return 0.0
    token_count = len(_tokens(text))
    if token_count > 24:
        return 0.35
    return 1.0


def format_jsonish(completion: list[dict[str, Any]] | str, **kwargs: Any) -> float:
    text = _completion_text(completion)
    if "refined_query" in text and "{" in text and "}" in text:
        return 1.0
    return 0.0


def _build_dataset(split: str, max_examples: int, include_hard: bool) -> Dataset:
    rows: list[dict[str, Any]] = []
    repetitions = 3 if split == "train" else 1
    for rep in range(repetitions):
        for topic_name, spec in TOPICS.items():
            rows.append(_row(topic_name, spec, rep, hard=False))
            if include_hard:
                rows.append(_row(topic_name, spec, rep, hard=True))
    if max_examples is not None and max_examples > 0:
        rows = rows[:max_examples]
    return Dataset.from_list(rows)


def _row(topic_name: str, spec: dict[str, Any], rep: int, hard: bool) -> dict[str, Any]:
    distractors = spec["distractors"] if hard else spec["distractors"][:1]
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task id: {topic_name}-{rep}-{'hard' if hard else 'base'}\n"
                f"Initial query: {spec['initial']}\n"
                "Positive evidence examples:\n"
                + "\n".join(f"- {item}" for item in spec["positives"])
                + "\nNear-miss distractors:\n"
                + "\n".join(f"- {item}" for item in distractors)
                + "\nReturn the refined query JSON."
            ),
        },
    ]
    return {
        "prompt": prompt,
        "answer": {
            "topic": topic_name,
            "initial_query": spec["initial"],
            "target_query": spec["target"],
            "target_terms": spec["terms"],
            "hard": hard,
        },
    }


def _completion_text(completion: list[dict[str, Any]] | str) -> str:
    if isinstance(completion, str):
        return completion.lower()
    messages = [msg for msg in completion if _message_role(msg) == "assistant"]
    if not messages:
        messages = completion[-1:] if completion else []
    content = " ".join(str(_message_content(msg)) for msg in messages)
    return content.lower()


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())
