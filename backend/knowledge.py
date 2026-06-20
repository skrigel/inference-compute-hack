from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from baseline.rag import RagBaseline
from data.schema import Chunk
from backend.schemas import FreshDocument


def generated_documents(
    *,
    source_kind: str = "mixed",
    size: int = 100,
    query: str = "retry without backoff",
) -> list[FreshDocument]:
    """Create a deterministic knowledge-source compartment for demos/tests."""
    count = max(1, min(size, 100_000))
    docs: list[FreshDocument] = []
    for idx in range(count):
        kind = _kind_at(source_kind, idx)
        if kind == "code":
            docs.append(
                FreshDocument(
                    title=f"service_{idx:05d}/retry_policy.py",
                    text=(
                        f"Code shard {idx}: {query}. The networking layer retries "
                        "HTTP calls and records queue saturation, throughput, and "
                        "latency counters for agentic search."
                    ),
                    type="code",
                    category="python" if idx % 3 else "go",
                    year=2026 - (idx % 5),
                    path=f"src/service_{idx:05d}/retry_policy.py",
                    lang="python" if idx % 3 else "go",
                    repo="synthetic-code",
                )
            )
        else:
            docs.append(
                FreshDocument(
                    title=f"Paper {idx:05d}: inference-time search metrics",
                    text=(
                        f"Paper shard {idx}: {query}. This abstract discusses "
                        "reward variance, trajectory entropy, retrieval ranking, "
                        "truth verification, and post-RL lift prediction."
                    ),
                    type="paper",
                    category="cs.LG" if idx % 2 else "cs.IR",
                    year=2026 - (idx % 4),
                    path=f"arxiv:synthetic.{idx:05d}",
                    lang=None,
                    repo="synthetic-papers",
                )
            )
    return docs


def browsecomp_documents(*, size: int = 100) -> list[FreshDocument]:
    """Load a bounded BrowseComp-Plus slice as MCP source documents."""
    from data.browsecomp_loader import load_browsecomp_corpus

    count = max(1, min(size, 100_000))
    chunks = load_browsecomp_corpus(limit=count)[:count]
    return [_fresh_document_from_chunk(chunk) for chunk in chunks]


def documents_from_dicts(raw_documents: list[dict[str, Any]]) -> list[FreshDocument]:
    return [document if isinstance(document, FreshDocument) else FreshDocument(**document) for document in raw_documents]


def fetch_arxiv_documents(query: str, *, max_results: int = 25) -> list[FreshDocument]:
    """Fetch papers from the public arXiv Atom API.

    Network use is isolated here so tests can cover the rest of the source
    compartment without relying on arXiv availability.
    """
    encoded = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(max_results, 100)),
        }
    )
    url = f"https://export.arxiv.org/api/query?{encoded}"
    with urllib.request.urlopen(url, timeout=20) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    docs = []
    for entry in root.findall("atom:entry", ns):
        title = _text(entry, "atom:title", ns) or "untitled arxiv paper"
        summary = _text(entry, "atom:summary", ns) or ""
        paper_id = _text(entry, "atom:id", ns) or title
        published = _text(entry, "atom:published", ns)
        category = entry.find("atom:category", ns)
        docs.append(
            FreshDocument(
                title=" ".join(title.split()),
                text=" ".join(summary.split()),
                type="paper",
                category=category.attrib.get("term") if category is not None else "arxiv",
                year=_year(published),
                path=paper_id,
                lang=None,
                repo="arxiv",
            )
        )
    return docs


def rag_search_documents(
    documents: list[FreshDocument],
    query: str,
    *,
    refinements: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    started = time.perf_counter()
    docs = [(document.title, document.text) for document in documents]
    rag = RagBaseline()
    stats = rag.build_index(docs)
    turns = [query, *(refinements or [])]
    hits = []
    retrieve_ms_total = 0.0
    for turn in turns:
        turn_hits, timing = rag.retrieve(turn, top_k=top_k)
        retrieve_ms_total += timing["query_embed_ms"] + timing["ann_ms"] + timing["rerank_ms"]
        hits.append({"query": turn, "hits": turn_hits})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "tool_name": "search_source_rag",
        "backend": stats.backend,
        "n_docs": len(documents),
        "turns": len(turns),
        "top_k": top_k,
        "index_build_ms": stats.index_build_ms,
        "embed_ms": stats.embed_ms,
        "retrieve_ms_total": retrieve_ms_total,
        "elapsed_ms": elapsed_ms,
        "work_units": len(documents) * len(turns),
        "hits": hits,
        "steps": [
            "embed corpus",
            "build vector index",
            *[f"retrieve top-{top_k}: {turn}" for turn in turns],
        ],
    }


def _kind_at(source_kind: str, idx: int) -> str:
    if source_kind == "code":
        return "code"
    if source_kind == "papers":
        return "paper"
    return "code" if idx % 2 else "paper"


def _fresh_document_from_chunk(chunk: Chunk) -> FreshDocument:
    return FreshDocument(
        title=chunk.title,
        text=chunk.text,
        type=chunk.type,
        category=chunk.meta.category,
        year=chunk.meta.year,
        path=chunk.meta.path or chunk.doc_id,
        lang=chunk.meta.lang,
        repo=chunk.meta.repo or "browsecomp",
    )


def _text(entry: ET.Element, path: str, ns: dict[str, str]) -> str | None:
    child = entry.find(path, ns)
    return child.text if child is not None else None


def _year(value: str | None) -> int | None:
    if not value or len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None
