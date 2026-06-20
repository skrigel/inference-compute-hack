from __future__ import annotations

import json
import re
from typing import Any


TRAIN_ROWS_PER_DOMAIN = 96
EVAL_ROWS_PER_DOMAIN = 24
DEFAULT_PASSAGES_PER_PROMPT = 16


DOMAIN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "domain": "scientific_claims",
        "initial_query": "study evidence",
        "target_query": "randomized trial evidence for aspirin reducing recurrent stroke risk",
        "target_terms": ("randomized", "trial", "aspirin", "recurrent", "stroke", "risk"),
        "exclude_terms": ("observational", "headache", "primary prevention"),
        "positive_templates": (
            "Randomized trial {n} reports that aspirin reduced recurrent stroke risk in adults with prior ischemic events.",
            "A controlled clinical study {n} finds lower recurrent stroke risk when aspirin is continued after ischemic stroke.",
            "The meta-analysis shard {n} pools randomized trials of aspirin after stroke and reports reduced recurrence.",
            "Clinical evidence note {n} links aspirin therapy to lower recurrent ischemic stroke risk in secondary prevention.",
        ),
        "hard_negative_templates": (
            "Observational cohort {n} discusses aspirin use for headache relief without measuring recurrent stroke outcomes.",
            "Primary prevention review {n} studies aspirin in healthy adults and excludes prior stroke patients.",
            "Case report {n} describes bleeding risk during aspirin therapy but does not evaluate recurrence reduction.",
            "Animal model note {n} tests platelet inhibition without randomized human stroke endpoints.",
        ),
    },
    {
        "domain": "finance_risk",
        "initial_query": "bank stress",
        "target_query": "regional bank liquidity stress from uninsured deposit outflows and rate risk",
        "target_terms": ("regional", "bank", "liquidity", "uninsured", "deposit", "rate"),
        "exclude_terms": ("credit card", "retail loan", "crypto exchange"),
        "positive_templates": (
            "Risk memo {n} identifies regional bank liquidity stress after uninsured deposit outflows accelerated during rate hikes.",
            "Supervisory note {n} ties rate risk in securities portfolios to deposit flight at a regional bank.",
            "Treasury desk brief {n} models uninsured deposit runoff and liquidity pressure at regional lenders.",
            "Banking analysis {n} links higher rates, unrealized losses, and regional bank funding stress.",
        ),
        "hard_negative_templates": (
            "Credit card delinquency report {n} analyzes consumer loan charge-offs rather than uninsured deposit flight.",
            "Crypto exchange note {n} describes stablecoin withdrawals without regional bank rate-risk exposure.",
            "Mortgage pipeline update {n} covers retail loan demand rather than securities losses and liquidity stress.",
            "Insurance capital memo {n} discusses reserve adequacy with no deposit outflow mechanism.",
        ),
    },
    {
        "domain": "code_retrieval",
        "initial_query": "retry",
        "target_query": "network client retry loop missing exponential backoff and jitter",
        "target_terms": ("network", "client", "retry", "exponential", "backoff", "jitter"),
        "exclude_terms": ("database deadlock", "cron job", "ui refresh"),
        "positive_templates": (
            "Source excerpt {n} shows a network client retry loop that immediately retries HTTP calls without exponential backoff.",
            "Incident snippet {n} traces retry storms to a missing jitter delay in the network client.",
            "Code review note {n} flags an HTTP retry loop that lacks exponential backoff under transient failures.",
            "Runtime trace {n} shows connection retries from the network client happening with no backoff interval.",
        ),
        "hard_negative_templates": (
            "Database deadlock handler {n} retries transactions with exponential backoff and randomized jitter.",
            "Cron job worker {n} retries failed tasks with capped backoff unrelated to network clients.",
            "UI refresh routine {n} repeats render attempts after animation frames but does not call HTTP services.",
            "Queue consumer {n} uses backoff correctly and has no network retry storm.",
        ),
    },
    {
        "domain": "security_incidents",
        "initial_query": "token issue",
        "target_query": "OAuth refresh token replay caused by missing nonce binding",
        "target_terms": ("oauth", "refresh", "token", "replay", "nonce", "binding"),
        "exclude_terms": ("password reset", "csrf cookie", "api quota"),
        "positive_templates": (
            "Security ticket {n} attributes OAuth refresh token replay to missing nonce binding during token rotation.",
            "Auth log analysis {n} finds repeated refresh token use because the nonce was not bound to the session.",
            "Incident review {n} recommends nonce binding to prevent replay of OAuth refresh tokens.",
            "Identity service trace {n} shows replayed refresh tokens accepted after rotation because binding checks were absent.",
        ),
        "hard_negative_templates": (
            "Password reset incident {n} covers expired email links and does not involve OAuth refresh token replay.",
            "CSRF cookie report {n} describes same-site cookie settings without nonce-bound refresh tokens.",
            "API quota alert {n} shows excessive requests from one client but no replayed credential artifact.",
            "SAML metadata note {n} rotates certificates without OAuth nonce binding changes.",
        ),
    },
    {
        "domain": "biomedical_safety",
        "initial_query": "dose toxicity",
        "target_query": "kinase inhibitor dose escalation associated with cardiotoxicity signal",
        "target_terms": ("kinase", "inhibitor", "dose", "escalation", "cardiotoxicity", "signal"),
        "exclude_terms": ("antibiotic", "rash", "pediatric vaccine"),
        "positive_templates": (
            "Safety table {n} reports a cardiotoxicity signal during dose escalation of the kinase inhibitor.",
            "Clinical pharmacology note {n} links higher kinase inhibitor exposure to cardiac adverse events.",
            "Trial appendix {n} flags cardiotoxicity during escalation cohorts for the targeted kinase inhibitor.",
            "Dose review {n} recommends cardiac monitoring after kinase inhibitor escalation signals.",
        ),
        "hard_negative_templates": (
            "Antibiotic dosing sheet {n} tracks renal adjustment but not kinase inhibitor cardiotoxicity.",
            "Dermatology adverse-event memo {n} describes rash during therapy with no cardiac safety signal.",
            "Pediatric vaccine schedule {n} discusses fever rates rather than oncology dose escalation.",
            "Manufacturing deviation note {n} covers batch sterility and no clinical cardiotoxicity endpoint.",
        ),
    },
    {
        "domain": "climate_measurement",
        "initial_query": "emissions satellite",
        "target_query": "satellite methane plume detection from oil and gas super emitters",
        "target_terms": ("satellite", "methane", "plume", "oil", "gas", "emitters"),
        "exclude_terms": ("carbon dioxide", "forest fire", "urban heat"),
        "positive_templates": (
            "Remote sensing report {n} identifies methane plume detections from oil and gas super emitters.",
            "Satellite pass {n} maps methane plumes above compressor stations in a gas basin.",
            "Atmospheric retrieval note {n} quantifies oil-field methane super emitter events from satellite pixels.",
            "Monitoring brief {n} compares repeated methane plume detections over oil and gas infrastructure.",
        ),
        "hard_negative_templates": (
            "Carbon dioxide inventory {n} estimates annual combustion emissions without satellite plume detection.",
            "Forest fire smoke note {n} tracks aerosols rather than methane from oil and gas equipment.",
            "Urban heat island map {n} uses thermal bands with no methane retrieval.",
            "Agricultural nitrous oxide report {n} excludes oil and gas super emitters.",
        ),
    },
    {
        "domain": "legal_contracts",
        "initial_query": "indemnity",
        "target_query": "vendor indemnification carveout for data breach caused by subcontractor",
        "target_terms": ("vendor", "indemnification", "carveout", "data", "breach", "subcontractor"),
        "exclude_terms": ("employment", "lease", "tax audit"),
        "positive_templates": (
            "Contract clause {n} states the vendor indemnification carveout does not apply to subcontractor-caused data breaches.",
            "Negotiation note {n} narrows vendor indemnification around data breach liability from subcontractors.",
            "Legal memo {n} explains why the subcontractor breach carveout changes vendor indemnity exposure.",
            "Redline {n} preserves indemnification for data breach claims caused by vendor subcontractors.",
        ),
        "hard_negative_templates": (
            "Employment agreement {n} covers worker indemnity and excludes vendor data breach obligations.",
            "Commercial lease clause {n} discusses property damage indemnity without subcontractor data handling.",
            "Tax audit provision {n} allocates accounting fees and does not mention data breach carveouts.",
            "Patent license section {n} covers IP infringement indemnity with no breach-response language.",
        ),
    },
    {
        "domain": "product_analytics",
        "initial_query": "activation funnel",
        "target_query": "mobile activation funnel drop-off after notification permission prompt",
        "target_terms": ("mobile", "activation", "funnel", "drop-off", "notification", "permission"),
        "exclude_terms": ("desktop onboarding", "billing churn", "email unsubscribe"),
        "positive_templates": (
            "Analytics slice {n} shows mobile activation funnel drop-off immediately after the notification permission prompt.",
            "Experiment readout {n} isolates notification permission copy as the largest mobile activation loss.",
            "Cohort query {n} links permission prompt denial to lower next-day activation on mobile.",
            "Growth memo {n} recommends delaying notification permission to reduce activation funnel drop-off.",
        ),
        "hard_negative_templates": (
            "Desktop onboarding report {n} studies workspace invite completion and not mobile permission prompts.",
            "Billing churn dashboard {n} tracks failed payments after activation rather than onboarding drop-off.",
            "Email unsubscribe analysis {n} concerns lifecycle messaging with no push notification prompt.",
            "Search ranking test {n} evaluates result order and not mobile activation.",
        ),
    },
    {
        "domain": "policy_procurement",
        "initial_query": "vendor privacy",
        "target_query": "AI vendor privacy review requiring subprocessors and data retention limits",
        "target_terms": ("ai", "vendor", "privacy", "subprocessors", "data", "retention"),
        "exclude_terms": ("office supplies", "hardware warranty", "travel policy"),
        "positive_templates": (
            "Procurement checklist {n} requires AI vendor privacy review of subprocessors and data retention limits.",
            "Risk assessment {n} blocks approval until the AI vendor lists subprocessors and retention windows.",
            "Privacy questionnaire {n} asks the vendor to disclose training-data retention and subprocessors.",
            "Legal approval note {n} conditions AI vendor onboarding on data retention and subprocessor controls.",
        ),
        "hard_negative_templates": (
            "Office supplies purchase order {n} covers stationery vendors without privacy subprocessors.",
            "Hardware warranty quote {n} includes replacement terms and no data retention review.",
            "Travel policy update {n} discusses expense approvals unrelated to AI vendor privacy.",
            "Facilities contract {n} lists janitorial subprocessors but no model data retention limits.",
        ),
    },
    {
        "domain": "systems_performance",
        "initial_query": "gpu utilization",
        "target_query": "vLLM prefill bottleneck low GPU utilization from small batch concurrency",
        "target_terms": ("vllm", "prefill", "bottleneck", "gpu", "utilization", "concurrency"),
        "exclude_terms": ("disk cache", "frontend css", "database index"),
        "positive_templates": (
            "Benchmark note {n} attributes low GPU utilization to vLLM prefill bottlenecks at small batch concurrency.",
            "Profiler output {n} shows prefill dominates latency while GPU utilization remains low.",
            "Load-test report {n} recommends increasing concurrency to improve vLLM prefill throughput.",
            "Serving trace {n} connects low GPU occupancy to underbatched prefill requests.",
        ),
        "hard_negative_templates": (
            "Disk cache profile {n} shows filesystem misses and no vLLM prefill measurements.",
            "Frontend CSS report {n} covers layout shift unrelated to GPU utilization.",
            "Database index plan {n} improves SQL scan time without model serving concurrency.",
            "Network egress graph {n} shows bandwidth throttling outside the inference server.",
        ),
    },
    {
        "domain": "education_assessment",
        "initial_query": "rubric grading",
        "target_query": "math rubric grading rewards intermediate reasoning consistency not final answer only",
        "target_terms": ("math", "rubric", "grading", "intermediate", "reasoning", "consistency"),
        "exclude_terms": ("essay grammar", "attendance", "multiple choice"),
        "positive_templates": (
            "Assessment guide {n} grades math work using intermediate reasoning consistency rather than final answer only.",
            "Rubric revision {n} assigns points for coherent algebra steps in math problem solutions.",
            "Evaluation note {n} penalizes inconsistent intermediate reasoning even when the final answer is correct.",
            "Teacher calibration {n} compares math reasoning traces against a step-level rubric.",
        ),
        "hard_negative_templates": (
            "Essay grammar rubric {n} scores punctuation and style rather than math reasoning.",
            "Attendance policy {n} affects course grade but not intermediate solution consistency.",
            "Multiple-choice item bank {n} records selected options without reasoning traces.",
            "Science lab rubric {n} grades procedure safety and not algebraic consistency.",
        ),
    },
    {
        "domain": "supply_chain",
        "initial_query": "supplier delay",
        "target_query": "semiconductor supplier delay caused by substrate shortage and port congestion",
        "target_terms": ("semiconductor", "supplier", "delay", "substrate", "shortage", "port"),
        "exclude_terms": ("labor strike", "retail stockout", "weather closure"),
        "positive_templates": (
            "Supply-chain update {n} attributes semiconductor supplier delay to substrate shortage and port congestion.",
            "Operations memo {n} links package substrate shortages with delayed chip supplier shipments.",
            "Logistics brief {n} combines port congestion data with semiconductor supplier lead-time increases.",
            "Procurement alert {n} warns that substrate allocation will delay semiconductor deliveries.",
        ),
        "hard_negative_templates": (
            "Labor strike notice {n} affects warehouse staffing but not semiconductor substrate supply.",
            "Retail stockout report {n} covers finished goods demand without supplier lead-time causes.",
            "Weather closure alert {n} delays a distribution center and not port congestion.",
            "Packaging redesign note {n} changes labels and does not mention chip substrates.",
        ),
    },
)


SYSTEM_PROMPT = """You refine search queries for retrieval over a dynamic corpus.
Return only JSON with this shape:
{"refined_query": "...", "evidence_ids": ["..."], "exclude_terms": ["..."], "stop": true}
The refined query should recover positive evidence, reject hard negatives, and
avoid broad queries that select every chunk."""


def build_retrieval_rows(
    *,
    split: str = "train",
    max_examples: int = -1,
    include_hard: bool = True,
    passages_per_prompt: int = DEFAULT_PASSAGES_PER_PROMPT,
) -> list[dict[str, Any]]:
    if split not in {"train", "eval", "heldout"}:
        raise ValueError("split must be train, eval, or heldout")
    examples_per_domain = TRAIN_ROWS_PER_DOMAIN if split == "train" else EVAL_ROWS_PER_DOMAIN
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(DOMAIN_SPECS):
        for idx in range(examples_per_domain):
            rows.append(
                _row(
                    spec=spec,
                    spec_index=spec_index,
                    row_index=idx,
                    split=split,
                    include_hard=include_hard,
                    passages_per_prompt=passages_per_prompt,
                )
            )
    if max_examples is not None and max_examples > 0:
        return rows[:max_examples]
    return rows


def score_completion(completion: list[dict[str, Any]] | str, answer: dict[str, Any] | str) -> dict[str, float]:
    payload = answer_payload(answer)
    completion_text = _completion_text(completion)
    completion_json = _completion_json(completion_text)
    refined_query = str(completion_json.get("refined_query") or completion_text).lower()
    evidence_ids = _string_set(completion_json.get("evidence_ids"))
    exclude_terms = _string_set(completion_json.get("exclude_terms"))

    target_terms = [str(term).lower() for term in payload["target_terms"]]
    positive_ids = [str(item) for item in payload["positive_ids"]]
    hard_negative_ids = [str(item) for item in payload["hard_negative_ids"]]
    expected_excludes = [str(item).lower() for item in payload["exclude_terms"]]

    target_term_coverage = _fraction(target_terms, lambda term: term in refined_query or term in completion_text)
    evidence_id_recall = _fraction(positive_ids, lambda item: item in evidence_ids or item.lower() in completion_text)
    hard_negative_rate = _fraction(hard_negative_ids, lambda item: item in evidence_ids or item.lower() in completion_text)
    hard_negative_rejection = max(0.0, 1.0 - hard_negative_rate)
    exclude_term_use = _fraction(expected_excludes, lambda item: item in exclude_terms or item in completion_text)
    initial_query_gain = _initial_query_gain(refined_query, payload)
    anti_select_all = _anti_select_all(completion_text)
    format_jsonish = 1.0 if completion_json else 0.0

    reward = (
        0.30 * target_term_coverage
        + 0.25 * evidence_id_recall
        + 0.15 * hard_negative_rejection
        + 0.10 * exclude_term_use
        + 0.10 * initial_query_gain
        + 0.05 * anti_select_all
        + 0.05 * format_jsonish
    )
    return {
        "reward": round(min(1.0, max(0.0, reward)), 6),
        "target_term_coverage": round(target_term_coverage, 6),
        "evidence_id_recall": round(evidence_id_recall, 6),
        "hard_negative_rejection": round(hard_negative_rejection, 6),
        "exclude_term_use": round(exclude_term_use, 6),
        "initial_query_gain": round(initial_query_gain, 6),
        "anti_select_all": round(anti_select_all, 6),
        "format_jsonish": round(format_jsonish, 6),
    }


def answer_payload(answer: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(answer, dict):
        return answer
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "domain": str(payload.get("domain", "")),
        "initial_query": str(payload.get("initial_query", "")),
        "target_query": str(payload.get("target_query", "")),
        "target_terms": [str(item) for item in payload.get("target_terms", [])],
        "exclude_terms": [str(item) for item in payload.get("exclude_terms", [])],
        "positive_ids": [str(item) for item in payload.get("positive_ids", [])],
        "hard_negative_ids": [str(item) for item in payload.get("hard_negative_ids", [])],
    }


def _row(
    *,
    spec: dict[str, Any],
    spec_index: int,
    row_index: int,
    split: str,
    include_hard: bool,
    passages_per_prompt: int,
) -> dict[str, Any]:
    domain = str(spec["domain"])
    task_id = f"{split}:{domain}:{row_index:04d}"
    positives = _passages("pos", spec, row_index, count=4)
    hard_negatives = _passages("hard", spec, row_index, count=4 if include_hard else 2)
    background = _background_passages(spec, row_index, count=max(0, passages_per_prompt - len(positives) - len(hard_negatives)))
    passages = positives + hard_negatives + background
    answer = {
        "domain": domain,
        "task_id": task_id,
        "initial_query": spec["initial_query"],
        "target_query": spec["target_query"],
        "target_terms": list(spec["target_terms"]),
        "exclude_terms": list(spec["exclude_terms"]),
        "positive_ids": [item["id"] for item in positives],
        "hard_negative_ids": [item["id"] for item in hard_negatives],
    }
    user_prompt = _user_prompt(spec, task_id=task_id, passages=passages, split=split, spec_index=spec_index)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "answer": json.dumps(answer, sort_keys=True),
        "task_id": task_id,
        "domain": domain,
        "cohort_id": f"{split}:{domain}",
        "difficulty": "hard" if include_hard else "medium",
        "prompt_word_count": len(user_prompt.split()),
    }


def _passages(kind: str, spec: dict[str, Any], row_index: int, *, count: int) -> list[dict[str, str]]:
    templates = spec["positive_templates"] if kind == "pos" else spec["hard_negative_templates"]
    passages: list[dict[str, str]] = []
    for offset in range(count):
        template = templates[(row_index + offset) % len(templates)]
        passage_id = f"{kind}:{spec['domain']}:{row_index:04d}:{offset}"
        passages.append(
            {
                "id": passage_id,
                "label": "positive" if kind == "pos" else "hard_negative",
                "text": template.format(n=f"{row_index}-{offset}"),
            }
        )
    return passages


def _background_passages(spec: dict[str, Any], row_index: int, *, count: int) -> list[dict[str, str]]:
    passages: list[dict[str, str]] = []
    domain = str(spec["domain"])
    for offset in range(count):
        exclude = spec["exclude_terms"][(row_index + offset) % len(spec["exclude_terms"])]
        target = spec["target_terms"][(row_index + offset) % len(spec["target_terms"])]
        passages.append(
            {
                "id": f"bg:{domain}:{row_index:04d}:{offset}",
                "label": "background",
                "text": (
                    f"Background shard {row_index}-{offset} mentions {target} near {exclude}, "
                    f"but it lacks the full causal chain needed for the target query and should not be selected."
                ),
            }
        )
    return passages


def _user_prompt(spec: dict[str, Any], *, task_id: str, passages: list[dict[str, str]], split: str, spec_index: int) -> str:
    passage_lines = "\n".join(
        f"- id={item['id']} label={item['label']}: {item['text']}" for item in passages
    )
    return (
        f"Task id: {task_id}\n"
        f"Split: {split}\n"
        f"Domain bucket: {spec['domain']} / cohort {spec_index}\n"
        f"Initial user query: {spec['initial_query']}\n\n"
        "Goal: refine the query so a retrieval system finds the true supporting passages while avoiding hard negatives. "
        "Return only JSON. Include evidence_ids for the positive passages you believe should be retrieved, exclude_terms "
        "for terms that distinguish hard negatives, and stop=true when the refined query is specific enough. "
        "A high-quality action is narrow, evidence-grounded, and cheap in bytes moved. A low-quality action is a broad "
        "query that selects every passage, copies the initial query, or includes hard-negative ids.\n\n"
        "Candidate corpus passages:\n"
        f"{passage_lines}\n\n"
        "Rubric reminders: cover the target concept, include positive evidence ids, reject hard-negative ids, use "
        "exclude_terms for misleading interpretations, and avoid phrases such as all documents, entire corpus, or select all. "
        "The final answer must be a single JSON object with keys refined_query, evidence_ids, exclude_terms, and stop."
    )


def _completion_text(completion: list[dict[str, Any]] | str) -> str:
    if isinstance(completion, str):
        return completion.lower()
    messages = [msg for msg in completion if _message_role(msg) == "assistant"]
    if not messages:
        messages = completion[-1:] if completion else []
    return " ".join(str(_message_content(msg)) for msg in messages).lower()


def _completion_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, list):
        return {str(item).lower() for item in value}
    return set()


def _fraction(items: list[str], predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(str(item).lower())) / len(items)


def _initial_query_gain(refined_query: str, payload: dict[str, Any]) -> float:
    initial_terms = set(_tokens(payload["initial_query"]))
    target_terms = set(_tokens(payload["target_query"]))
    added_terms = target_terms - initial_terms
    if not added_terms:
        return 0.0
    return sum(1 for term in added_terms if term in refined_query) / len(added_terms)


def _anti_select_all(text: str) -> float:
    broad_markers = ("all chunks", "everything", "entire corpus", "select all", "all documents", "any document")
    if any(marker in text for marker in broad_markers):
        return 0.0
    token_count = len(_tokens(text))
    if token_count > 42:
        return 0.35
    return 1.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())
