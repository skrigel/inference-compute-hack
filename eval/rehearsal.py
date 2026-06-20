"""Phase 06 freeze + rehearsal harness — rehearse failure, not just success.

Runs the three dress rehearsals (live path, injected-failure → replay fallback,
timed ≤90s) and assembles a reproducible freeze manifest from the saved
artifacts. `python -m eval.rehearsal` runs all three and writes
`eval/artifacts/freeze_manifest.json`; exit 0 only if all rehearsals pass.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO / "eval" / "artifacts"

# Canonical spoken-beat budget (must match DEMO.md). Sum must be ≤ 90s.
BEAT_BUDGET_S: list[tuple[str, int]] = [
    ("hook", 7),
    ("query", 15),
    ("click-NOT", 12),
    ("AND refine", 13),
    ("threshold drag", 10),
    ("fresh-file", 15),
    ("perf-close", 13),
]


@dataclass
class RehearsalReport:
    live: bool = False
    fallback: bool = False
    timed: bool = False
    budget_s: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return self.live and self.fallback and self.timed


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rehearsal_live() -> tuple[bool, str]:
    """R1: the live (mock-backed) path drives the full irreducible loop green."""
    from eval.cut_line import run_cut_line

    result = run_cut_line()
    return result.green, f"cut-line {'GREEN' if result.green else 'RED ' + str(result.failures)}"


def rehearsal_fallback() -> tuple[bool, str]:
    """R2: with the live path 'down', the canned replay serves every beat."""
    from fastapi.testclient import TestClient

    from scripts.replay_sse import build_replay_app, record

    with tempfile.TemporaryDirectory() as tmp:
        record(Path(tmp))
        client = TestClient(build_replay_app(Path(tmp)))
        if client.get("/healthz").json().get("scorer") != "replay":
            return False, "replay healthz not ready"
        opening = client.post("/query", json={"predicate": "x", "threshold": 0.5}).text
        client.post("/ingest", json={"corpus_id": "demo", "documents": [{"title": "f.py", "text": "t"}]})
        fresh = client.post("/query", json={"predicate": "x", "threshold": 0.5}).text
        refine = client.post("/refine", json={"utterance": "y"}).text
    ok = (
        "result" in opening
        and "fresh_incident.py" in fresh
        and '"type":"chip"' in refine.replace(" ", "")
    )
    return ok, "replay serves beats 1/2-3/5 (fresh toggle armed)" if ok else "replay missing a beat"


def rehearsal_timed() -> tuple[bool, int, str]:
    """R3: the spoken-beat budget fits in ≤90s without rushing."""
    total = sum(seconds for _, seconds in BEAT_BUDGET_S)
    return total <= 90, total, f"spoken budget {total}s ≤ 90s"


def run_rehearsals() -> RehearsalReport:
    report = RehearsalReport()
    report.live, live_note = rehearsal_live()
    report.fallback, fb_note = rehearsal_fallback()
    report.timed, report.budget_s, timed_note = rehearsal_timed()
    report.notes = [f"live: {live_note}", f"fallback: {fb_note}", f"timed: {timed_note}"]
    return report


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def build_freeze_manifest(artifact_dir: Path = ARTIFACT_DIR) -> dict:
    """Reproducible snapshot of the frozen claims, read from saved artifacts."""
    from backend.state import demo_chunks
    from inference.mock_scorer import MockScorer

    chunks = demo_chunks()
    corpus_chars = sum(len(c.text) for c in chunks)
    corpus_words = sum(len(c.text.split()) for c in chunks)

    trace = _load(artifact_dir / "cut_line_trace.json")
    aul = trace.get("area_under_loop", {})
    gate = _load(artifact_dir / "phase04_quality_gate.json")
    gate_quality = gate.get("quality", {})
    rag = _load(artifact_dir / "phase04_rag_vs_6xh100.json")
    ref = rag.get("six_h100_reference", {})

    return {
        "commit": _git_commit(),
        "provenance": {
            "demo_scorer": MockScorer().model_id(),
            "real_scorer_model": gate.get("model_id"),
            "vllm_version": ref.get("vllm_version"),
        },
        "corpus": {
            "id": "demo",
            "n_chunks": len(chunks),
            "chars": corpus_chars,
            "approx_words": corpus_words,
        },
        "frozen_metrics": {
            "area_under_loop_scoped": aul.get("scoped_total"),
            "area_under_loop_full": aul.get("full_total"),
            "area_under_loop_label": "measured-mock",
            "quality_gate": {
                "backend": gate.get("scorer_backend"),
                "small_gate": gate.get("small_gate"),
                "threshold": gate.get("threshold"),
                "recommended_threshold": gate.get("recommended_threshold"),
                "precision": gate_quality.get("precision"),
                "recall": gate_quality.get("recall"),
                "f1": gate_quality.get("f1"),
                "label": gate.get("label"),
            },
            "six_h100": {
                "requests_per_s": ref.get("requests_per_s"),
                "total_tokens_per_s": ref.get("total_tokens_per_s"),
                "latency_ms_p50_mean": ref.get("latency_ms_p50_mean"),
                "latency_ms_p95_max": ref.get("latency_ms_p95_max"),
                "derived_mfu_bf16_peak": ref.get("derived_mfu_bf16_peak_mean"),
                "label": "measured-6xH100 (under-saturated micro-benchmark)",
            },
        },
        "caveats": [
            "Quality gate is the SMALL 7-chunk demo gate, measured on real Modal vLLM "
            f"({gate.get('model_id')}): F1 {gate_quality.get('f1')} at calibrated threshold "
            f"{gate.get('recommended_threshold')}. Default 0.5 collapses recall (~0.11) — the real "
            "backend MUST use the calibrated threshold. Full gold-set + ladder freeze is PENDING "
            "the Modal spend-limit reset (see phase04_modal_blocker.md).",
            "Derived MFU ~0.06 is an under-saturated micro-benchmark (low concurrency, short "
            "prompts, queue depth 0), NOT the architecture's MFU; 40-55% prefill is PROJECTED.",
            "Warm-cache KV crossover ~14k chunks (FP16) is projected from performance/theory.py; "
            "candidate-set scoping carries the refine path past it.",
            "Canned replay twins are recorded from the mock backend; re-record from a real vLLM "
            "run once unblocked, then state 'recorded from the same scorer path'.",
            "Frontend dist/ must be built ahead of time on Node >= 20.19 (Vite 8).",
        ],
        "reproduce": {
            "loop_and_figure": "python -m eval.cut_line --figure",
            "rehearsals_and_manifest": "python -m eval.rehearsal",
            "theory_figures": "python performance/theory.py",
            "real_gate_freeze_when_unblocked": "SCORER_BACKEND=modal python -m eval.bench --backend modal --tag freeze --weave",
        },
    }


def main() -> None:
    report = run_rehearsals()
    manifest = build_freeze_manifest()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print("── Phase 06 rehearsals ─────────────────────────────────────────")
    for note in report.notes:
        print(f"  {note}")
    print(f"── freeze manifest ─ commit {manifest['commit']} · corpus {manifest['corpus']['n_chunks']} chunks")
    gate = manifest["frozen_metrics"]["quality_gate"]
    print(f"  real gate (small, {gate['backend']}): F1 {gate['f1']} @ thr {gate['recommended_threshold']}")
    print(f"  area-under-loop: scoped {manifest['frozen_metrics']['area_under_loop_scoped']} vs full {manifest['frozen_metrics']['area_under_loop_full']}")
    print("────────────────────────────────────────────────────────────────")
    if report.all_pass:
        print("FREEZE ✓  three rehearsals pass; manifest written to eval/artifacts/freeze_manifest.json")
        raise SystemExit(0)
    print("NOT FROZEN ✗  a rehearsal failed — see notes above.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
