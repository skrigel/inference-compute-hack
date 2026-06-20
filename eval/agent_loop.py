from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data.schema import Chunk, ChunkMeta, chunk_id_of
from inference.mock_scorer import MockScorer
from inference.scorer import ScoreRequest, ScorerClient


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
TOPIC_SPECS = {
    "retry_backoff": {
        "initial_query": "retry",
        "target_query": "networking layer retry without exponential backoff",
        "positive_terms": ("retry", "networking", "backoff"),
        "positive_templates": (
            "The networking layer retries transient HTTP failures but never applies exponential backoff.",
            "Connection pool retry logic in the networking stack skips exponential backoff under load.",
            "The async HTTP client retries network calls without jitter or exponential backoff.",
        ),
        "distractor_templates": (
            "A background worker retries a failed task with capped exponential backoff.",
            "The database session retries a deadlock and applies exponential backoff.",
            "A UI histogram updates thresholds without rescoring cached result chunks.",
            "A paper studies ranking metrics for neural retrieval systems.",
        ),
        "refinement_terms": ("networking layer", "without backoff", "HTTP", "connection pool", "jitter"),
    },
    "ir_retrieval": {
        "initial_query": "retrieval",
        "target_query": "information retrieval ranking metrics for code search",
        "positive_terms": ("retrieval", "ranking", "search"),
        "positive_templates": (
            "A paper studies information retrieval ranking metrics for code search.",
            "Neural retrieval for code search compares ranking metrics and semantic quality.",
            "The IR system evaluates search quality with recall and precision metrics.",
        ),
        "distractor_templates": (
            "A backend cache stores relevance scores for threshold dragging.",
            "The networking layer retries HTTP requests without backoff.",
            "A frontend component renders corpus cards and filters.",
            "A database worker handles deadlock retry policy.",
        ),
        "refinement_terms": ("ranking metrics", "code search", "information retrieval", "semantic quality", "IR sense"),
    },
    "cache_threshold": {
        "initial_query": "cache",
        "target_query": "cached threshold drag without rescoring",
        "positive_terms": ("cache", "threshold", "rescoring"),
        "positive_templates": (
            "Histogram brushing updates the threshold without rescoring cached chunks.",
            "The results cache lets threshold drag move instantly with zero inference.",
            "Cached score reads update visible threshold results without rescoring or another model call.",
        ),
        "distractor_templates": (
            "A paper studies semantic search ranking metrics.",
            "A networking adapter retries transient HTTP failures.",
            "A background worker runs exponential backoff on failed jobs.",
            "A corpus modal creates synthetic documents for search.",
        ),
        "refinement_terms": ("threshold drag", "without rescoring", "zero inference", "cached scores", "histogram brushing"),
    },
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected and predicted)
    fp = sum(1 for expected, predicted in zip(y_true, y_pred) if not expected and predicted)
    fn = sum(1 for expected, predicted in zip(y_true, y_pred) if expected and not predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


@dataclass(frozen=True)
class QueryTask:
    task_id: str
    target_topic: str
    initial_query: str
    target_query: str
    chunks: list[Chunk]
    positive_chunk_ids: set[str]
    memory_bytes_total: int
    positive_bytes: int


@dataclass(frozen=True)
class LoopStep:
    step_idx: int
    query: str
    action: str
    reward: float
    quality: dict[str, float]
    selected_count: int
    bytes_moved: int
    selected_bytes: int
    score_latency_ms: float


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    target_topic: str
    initial_query: str
    best_query: str
    best_reward: float
    best_quality: dict[str, float]
    steps: list[LoopStep]
    metrics: dict[str, float | int | str | None]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        return payload


def generate_dynamic_query_task(
    *,
    task_id: str,
    n_docs: int,
    target_topic: str = "retry_backoff",
    positive_ratio: float = 0.08,
) -> QueryTask:
    if target_topic not in TOPIC_SPECS:
        raise ValueError(f"unknown target_topic={target_topic!r}; expected one of {sorted(TOPIC_SPECS)}")
    if n_docs < 10:
        raise ValueError("n_docs must be at least 10 so the task has positives and distractors")

    spec = TOPIC_SPECS[target_topic]
    n_positive = max(3, min(n_docs // 2, int(n_docs * positive_ratio)))
    chunks: list[Chunk] = []
    positive_chunk_ids: set[str] = set()
    for idx in range(n_docs):
        is_positive = idx < n_positive
        templates = spec["positive_templates"] if is_positive else spec["distractor_templates"]
        text = templates[idx % len(templates)]
        if is_positive:
            text = f"{text} Evidence shard {idx}."
        else:
            text = f"{text} Distractor shard {idx}."
        doc_id = f"dynamic:{task_id}:{target_topic}:{idx}"
        chunk = Chunk(
            chunk_id=chunk_id_of(doc_id, idx, text),
            doc_id=doc_id,
            type="code" if idx % 2 == 0 else "paper",
            title=f"{target_topic}-{idx:05d}",
            text=text,
            meta=ChunkMeta(
                category=target_topic if is_positive else "distractor",
                year=2026,
                path=f"dynamic/{target_topic}/{idx}.txt",
                lang="text",
                repo="dynamic",
                source="synthetic-agent-loop",
            ),
        )
        chunks.append(chunk)
        if is_positive:
            positive_chunk_ids.add(chunk.chunk_id)

    memory_bytes_total = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
    positive_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in chunks if chunk.chunk_id in positive_chunk_ids)
    return QueryTask(
        task_id=task_id,
        target_topic=target_topic,
        initial_query=spec["initial_query"],
        target_query=spec["target_query"],
        chunks=chunks,
        positive_chunk_ids=positive_chunk_ids,
        memory_bytes_total=memory_bytes_total,
        positive_bytes=positive_bytes,
    )


async def run_query_refinement_episode(
    task: QueryTask,
    *,
    scorer: ScorerClient,
    threshold: float = 0.5,
    max_steps: int = 5,
    beam_width: int = 5,
) -> EpisodeResult:
    candidate_queries = _candidate_queries(task, max_steps=max_steps, beam_width=beam_width)
    steps: list[LoopStep] = []
    best_step: LoopStep | None = None
    for step_idx, (query, action) in enumerate(candidate_queries):
        step = await _evaluate_query(task, scorer=scorer, query=query, action=action, step_idx=step_idx, threshold=threshold)
        steps.append(step)
        if best_step is None or step.reward > best_step.reward:
            best_step = step

    if best_step is None:
        raise RuntimeError("episode produced no steps")
    metrics = _episode_metrics(task, steps, best_step)
    return EpisodeResult(
        task_id=task.task_id,
        target_topic=task.target_topic,
        initial_query=task.initial_query,
        best_query=best_step.query,
        best_reward=best_step.reward,
        best_quality=best_step.quality,
        steps=steps,
        metrics=metrics,
    )


async def run_agent_loop_experiment(
    *,
    n_docs: int = 1_000,
    task_count: int = 3,
    max_steps: int = 5,
    beam_width: int = 5,
    threshold: float = 0.5,
    human_turn_ms: float = 30_000.0,
    scorer: ScorerClient | None = None,
) -> dict[str, Any]:
    scorer = scorer or MockScorer()
    topics = list(TOPIC_SPECS)
    episodes: list[EpisodeResult] = []
    started = time.perf_counter()
    for idx in range(task_count):
        task = generate_dynamic_query_task(
            task_id=f"agent-loop-{idx}",
            n_docs=n_docs,
            target_topic=topics[idx % len(topics)],
        )
        episode = await run_query_refinement_episode(
            task,
            scorer=scorer,
            threshold=threshold,
            max_steps=max_steps,
            beam_width=beam_width,
        )
        episodes.append(episode)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "run_id": f"agent-loop-{uuid.uuid4().hex[:8]}",
        "commit": _git_commit(),
        "model_id": scorer.model_id(),
        "n_docs": n_docs,
        "task_count": task_count,
        "threshold": threshold,
        "max_steps": max_steps,
        "beam_width": beam_width,
        "elapsed_ms": elapsed_ms,
        "framework": _framework_description(),
        "episodes": [episode.to_dict() for episode in episodes],
        "dataset_metrics": _dataset_metrics(episodes, elapsed_ms=elapsed_ms, human_turn_ms=human_turn_ms),
    }


def write_agent_loop_artifacts(payload: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown_report(payload) + "\n")


def _candidate_queries(task: QueryTask, *, max_steps: int, beam_width: int) -> list[tuple[str, str]]:
    spec = TOPIC_SPECS[task.target_topic]
    queries = [(task.initial_query, "initial")]
    current = task.initial_query
    for term in spec["refinement_terms"][: max(0, beam_width - 1)]:
        if term.lower() not in current.lower():
            current = f"{current} {term}"
        queries.append((current, f"add:{term}"))
    if task.target_query not in {query for query, _ in queries}:
        queries.append((task.target_query, "oracle-target-query"))
    return queries[: max_steps]


async def _evaluate_query(
    task: QueryTask,
    *,
    scorer: ScorerClient,
    query: str,
    action: str,
    step_idx: int,
    threshold: float,
) -> LoopStep:
    requests = [ScoreRequest(chunk.chunk_id, chunk.text, query) for chunk in task.chunks]
    started = time.perf_counter()
    scores = await scorer.score_batch(requests)
    score_latency_ms = (time.perf_counter() - started) * 1000.0
    selected = {score.chunk_id for score in scores if score.score >= threshold}
    selected_chunks = [chunk for chunk in task.chunks if chunk.chunk_id in selected]
    y_true = [chunk.chunk_id in task.positive_chunk_ids for chunk in task.chunks]
    y_pred = [chunk.chunk_id in selected for chunk in task.chunks]
    quality = _classification_metrics(y_true, y_pred)
    bytes_moved = sum(len(chunk.text.encode("utf-8")) for chunk in task.chunks)
    selected_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in selected_chunks)
    reward = _reward(quality, bytes_moved=bytes_moved, selected_bytes=selected_bytes, memory_bytes_total=task.memory_bytes_total)
    return LoopStep(
        step_idx=step_idx,
        query=query,
        action=action,
        reward=reward,
        quality=quality,
        selected_count=len(selected),
        bytes_moved=bytes_moved,
        selected_bytes=selected_bytes,
        score_latency_ms=score_latency_ms,
    )


def _reward(quality: dict[str, float], *, bytes_moved: int, selected_bytes: int, memory_bytes_total: int) -> float:
    movement_penalty = bytes_moved / max(memory_bytes_total, 1)
    selected_penalty = selected_bytes / max(memory_bytes_total, 1)
    return float(quality["f1"] + 0.25 * quality["recall"] - 0.05 * movement_penalty - 0.10 * selected_penalty)


def _episode_metrics(task: QueryTask, steps: list[LoopStep], best_step: LoopStep) -> dict[str, float | int | str | None]:
    initial = steps[0]
    rewards = [step.reward for step in steps]
    selected_counts = [step.selected_count for step in steps]
    first_passing = next((step.step_idx for step in steps if step.quality["recall"] >= 0.8 and step.quality["precision"] >= 0.5), None)
    return {
        "task_token_length": len(task.target_query.split()),
        "branching_factor": len(steps),
        "reward_variance": statistics.pvariance(rewards) if len(rewards) > 1 else 0.0,
        "max_reward": best_step.reward,
        "initial_reward": initial.reward,
        "truth_gain": best_step.quality["f1"] - initial.quality["f1"],
        "max_score": best_step.quality["f1"],
        "steps_to_threshold": first_passing,
        "tool_calls": len(steps),
        "reasoning_steps": max(0, len(steps) - 1),
        "memory_selectivity": task.positive_bytes / max(task.memory_bytes_total, 1),
        "movement_selectivity": best_step.selected_bytes / max(task.memory_bytes_total, 1),
        "bytes_moved_total": sum(step.bytes_moved for step in steps),
        "bytes_selected_best": best_step.selected_bytes,
        "trajectory_entropy": _entropy(selected_counts),
        "cost_proxy_model_calls": len(steps) * len(task.chunks),
        "best_query": best_step.query,
    }


def _dataset_metrics(episodes: list[EpisodeResult], *, elapsed_ms: float, human_turn_ms: float) -> dict[str, Any]:
    best_rewards = [episode.best_reward for episode in episodes]
    best_f1 = [episode.best_quality["f1"] for episode in episodes]
    reward_variance = [float(episode.metrics["reward_variance"]) for episode in episodes]
    pass_flags = [episode.best_quality["recall"] >= 0.8 and episode.best_quality["precision"] >= 0.5 for episode in episodes]
    total_agent_turns = sum(len(episode.steps) for episode in episodes)
    estimated_human_ms = total_agent_turns * human_turn_ms
    return {
        "mean_best_reward": statistics.fmean(best_rewards) if best_rewards else 0.0,
        "mean_best_f1": statistics.fmean(best_f1) if best_f1 else 0.0,
        "pass_rate": sum(pass_flags) / len(pass_flags) if pass_flags else 0.0,
        "mean_reward_variance": statistics.fmean(reward_variance) if reward_variance else 0.0,
        "trajectory_entropy": statistics.fmean(float(episode.metrics["trajectory_entropy"]) for episode in episodes)
        if episodes
        else 0.0,
        "task_diversity": len({episode.target_topic for episode in episodes}) / max(len(TOPIC_SPECS), 1),
        "mean_steps_to_threshold": _mean_present(
            [episode.metrics["steps_to_threshold"] for episode in episodes if episode.metrics["steps_to_threshold"] is not None]
        ),
        "mean_memory_selectivity": statistics.fmean(float(episode.metrics["memory_selectivity"]) for episode in episodes)
        if episodes
        else 0.0,
        "mean_movement_selectivity": statistics.fmean(float(episode.metrics["movement_selectivity"]) for episode in episodes)
        if episodes
        else 0.0,
        "agent_elapsed_ms": elapsed_ms,
        "estimated_human_ms": estimated_human_ms,
        "agent_vs_human_speedup_estimate": estimated_human_ms / max(elapsed_ms, 1e-9),
        "cost_quality_frontier": [
            {
                "task_id": episode.task_id,
                "cost_proxy_model_calls": episode.metrics["cost_proxy_model_calls"],
                "quality": episode.best_quality["f1"],
                "reward": episode.best_reward,
            }
            for episode in episodes
        ],
    }


def _framework_description() -> dict[str, Any]:
    return {
        "primitives": {
            "T": "query-refinement task over a labeled synthetic dynamic corpus",
            "M": "scorer.score_batch(query, chunks) returning relevance probabilities",
            "V": "verifier comparing selected chunks to task positive ids",
            "y": "candidate refined query and selected evidence set",
            "r": "F1 + recall bonus - byte movement/storage penalties",
        },
        "axes": {
            "memory_capacity": "positive/evidence bytes divided by total stored corpus bytes",
            "memory_bandwidth": "selected bytes and scored bytes moved per rollout",
            "answer_error": "1 - F1 plus missing-evidence recall error",
        },
        "rule_of_thumb": "spend extra inference on planning and verification, not blind storage or byte movement",
    }


def _entropy(values: list[int]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log2(p)
    return entropy


def _mean_present(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def _markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload["dataset_metrics"]
    lines = [
        "# Extension 3 Agent Loop Experiment",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- commit: `{payload['commit']}`",
        f"- model: `{payload['model_id']}`",
        f"- docs per task: `{payload['n_docs']}`",
        f"- tasks: `{payload['task_count']}`",
        f"- elapsed_ms: `{payload['elapsed_ms']:.3f}`",
        "",
        "## Dataset Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "mean_best_reward",
        "mean_best_f1",
        "pass_rate",
        "task_diversity",
        "trajectory_entropy",
        "mean_memory_selectivity",
        "mean_movement_selectivity",
        "agent_vs_human_speedup_estimate",
    ):
        value = metrics[key]
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            "| task | topic | best query | best reward | precision | recall | F1 | movement selectivity |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for episode in payload["episodes"]:
        q = episode["best_query"].replace("|", "/")
        quality = episode["best_quality"]
        lines.append(
            "| "
            f"{episode['task_id']} | {episode['target_topic']} | {q} | "
            f"{episode['best_reward']:.6f} | {quality['precision']:.6f} | "
            f"{quality['recall']:.6f} | {quality['f1']:.6f} | "
            f"{episode['metrics']['movement_selectivity']:.6f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Extension 3 agentic query-refinement environment.")
    parser.add_argument("--n-docs", type=int, default=1_000)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--human-turn-ms", type=float, default=30_000.0)
    parser.add_argument("--output-json", type=Path, default=ARTIFACT_DIR / "extension3_agent_loop.json")
    parser.add_argument("--output-md", type=Path, default=ARTIFACT_DIR / "extension3_agent_loop.md")
    args = parser.parse_args(argv)

    payload = asyncio.run(
        run_agent_loop_experiment(
            n_docs=args.n_docs,
            task_count=args.task_count,
            max_steps=args.max_steps,
            beam_width=args.beam_width,
            threshold=args.threshold,
            human_turn_ms=args.human_turn_ms,
        )
    )
    write_agent_loop_artifacts(payload, output_json=args.output_json, output_md=args.output_md)
    print(json.dumps(payload["dataset_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
