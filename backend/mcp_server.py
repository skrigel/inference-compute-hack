"""Agent mode (Axis 3, Mode 2): expose the three compute axes as MCP tools.

This is the thin Model-Context-Protocol surface over ``backend.agent.AgentSession``
so an autonomous agent can drive Memory / Movement / Truth the same way the UI
does. The ``mcp`` package is an *optional* dependency: importing this module
never fails, and ``build_server()`` / ``main()`` raise a clear, actionable error
if it is absent — so the backend and its tests take no hard dependency on it.

Run it (after ``pip install mcp``):

    python -m backend.mcp_server

and point an MCP-capable client at the stdio transport.
"""

from __future__ import annotations

import time
from typing import Any

from backend.agent import AgentSession
from backend.knowledge import (
    documents_from_dicts,
    fetch_arxiv_documents,
    generated_documents,
    rag_search_documents,
)
from backend.schemas import FreshDocument

try:  # optional dependency — agent mode only
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without mcp installed
    FastMCP = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


def mcp_available() -> bool:
    """Whether the optional ``mcp`` runtime is importable."""
    return _MCP_AVAILABLE


def build_server(session: AgentSession | None = None, *, name: str = "grepmeaning-agent"):
    """Register the three axes as MCP tools over a single ``AgentSession``.

    Raises ``RuntimeError`` (not ImportError at module load) if ``mcp`` is
    missing, so callers get an actionable message and the module stays importable
    for testing the tool wiring without the runtime.
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "Agent mode needs the 'mcp' package. Install it with: pip install mcp"
        )
    session = session or AgentSession()
    server = FastMCP(name)
    source_compartments: dict[str, list[FreshDocument]] = {}

    @server.tool()
    async def ingest(corpus_id: str = "demo", limit: int | None = None) -> dict:
        """Load a corpus ('demo' or 'browsecomp') into the session."""
        return await session.ingest(corpus_id, limit=limit)

    @server.tool()
    async def query(
        predicate: str,
        compute_budget: float = 1.0,
        threshold: float = 0.5,
        top_k: int | None = None,
    ) -> dict:
        """Axis 1 (Memory): score the budgeted slice of the corpus; return ranked survivors."""
        return await session.query(
            predicate, compute_budget=compute_budget, threshold=threshold, top_k=top_k
        )

    @server.tool()
    def select(
        mode: str = "threshold",
        precision_target: float = 0.85,
        movement_budget: int = 5,
        beam_width: int = 4,
    ) -> dict:
        """Axis 2 (Movement): auto-threshold or max-coverage smart-select (zero inference)."""
        return session.select(
            mode=mode,
            precision_target=precision_target,
            movement_budget=movement_budget,
            beam_width=beam_width,
        )

    @server.tool()
    async def refine(utterance: str, beam_width: int = 4) -> dict:
        """Axis 3 (Truth): run the predicate beam and apply the objective-selected winner."""
        return await session.refine(utterance, beam_width=beam_width)

    @server.tool()
    def results(threshold: float | None = None, top_k: int | None = None) -> dict:
        """Ranked slice of the current clause — a pure cache read."""
        return session.results(threshold=threshold, top_k=top_k)

    @server.tool()
    async def ingest_source(
        source_id: str = "demo-source",
        source_kind: str = "mixed",
        size: int = 100,
        arxiv_query: str | None = None,
        documents: list[dict[str, Any]] | None = None,
    ) -> dict:
        """Create a source compartment from code, papers, arXiv, or explicit docs."""
        if documents:
            loaded = documents_from_dicts(documents)
        elif arxiv_query:
            loaded = fetch_arxiv_documents(arxiv_query, max_results=size)
        else:
            loaded = generated_documents(source_kind=source_kind, size=size)
        source_compartments[source_id] = loaded
        result = await session.ingest_documents(source_id, loaded)
        return {
            **result,
            "source_kind": source_kind,
            "arxiv_query": arxiv_query,
            "document_titles": [document.title for document in loaded[:5]],
        }

    @server.tool()
    async def search_source_ours(
        source_id: str,
        query: str,
        refinements: list[str] | None = None,
        source_kind: str = "mixed",
        size: int = 100,
        compute_budget: float = 1.0,
        precision_target: float = 0.85,
        movement_budget: int = 5,
        beam_width: int = 4,
        top_k: int = 5,
    ) -> dict:
        """Search one source compartment with score-cache + select + refine."""
        documents_for_source = _source_documents(source_compartments, source_id, source_kind, size)
        source_compartments[source_id] = documents_for_source
        started = time.perf_counter()
        await session.ingest_documents(source_id, documents_for_source)
        query_result = await session.query(
            query,
            compute_budget=compute_budget,
            threshold=0.5,
            top_k=top_k,
        )
        selection = session.select(
            mode="smart",
            precision_target=precision_target,
            movement_budget=movement_budget,
            beam_width=beam_width,
        )
        refine_results = []
        for refine_text in refinements or []:
            refine_results.append(await session.refine(refine_text, beam_width=beam_width))
        final_results = session.results(threshold=0.5, top_k=top_k)
        refined_work = sum(result.get("matched", 0) for result in refine_results)
        work_units = query_result["corpus_scored"] + refined_work
        return {
            "tool_name": "search_source_ours",
            "source_id": source_id,
            "n_docs": len(documents_for_source),
            "query": query,
            "refinements": refinements or [],
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "work_units": work_units,
            "corpus_total": query_result["corpus_total"],
            "corpus_scored": query_result["corpus_scored"],
            "selected_count": len(selection["selected_ids"]),
            "selection": selection,
            "results": final_results["results"],
            "steps": [
                "score source once",
                "smart-select over cached scores",
                *[f"refine cached survivors: {text}" for text in refinements or []],
                "return final cache slice",
            ],
        }

    @server.tool()
    async def search_source_rag(
        source_id: str,
        query: str,
        refinements: list[str] | None = None,
        source_kind: str = "mixed",
        size: int = 100,
        top_k: int = 5,
    ) -> dict:
        """Search one source compartment with the RAG baseline."""
        documents_for_source = _source_documents(source_compartments, source_id, source_kind, size)
        source_compartments[source_id] = documents_for_source
        return rag_search_documents(documents_for_source, query, refinements=refinements, top_k=top_k)

    @server.tool()
    async def compare_source_search(
        source_id: str,
        query: str,
        refinements: list[str] | None = None,
        source_kind: str = "mixed",
        size: int = 100,
        top_k: int = 5,
    ) -> dict:
        """Run RAG and this project's source-search tool side by side."""
        rag = await search_source_rag(
            source_id=source_id,
            query=query,
            refinements=refinements,
            source_kind=source_kind,
            size=size,
            top_k=top_k,
        )
        ours = await search_source_ours(
            source_id=source_id,
            query=query,
            refinements=refinements,
            source_kind=source_kind,
            size=size,
            top_k=top_k,
        )
        speedup = rag["work_units"] / max(ours["work_units"], 1)
        return {
            "source_id": source_id,
            "query": query,
            "refinements": refinements or [],
            "rag": rag,
            "ours": ours,
            "work_units_speedup": speedup,
            "speedup_basis": "RAG work units scan/retrieve the source every turn; ours scores once then refines cached survivors.",
        }

    return server


def _source_documents(
    source_compartments: dict[str, list[FreshDocument]],
    source_id: str,
    source_kind: str,
    size: int,
) -> list[FreshDocument]:
    existing = source_compartments.get(source_id)
    if existing:
        return existing
    return generated_documents(source_kind=source_kind, size=size)


def main() -> None:  # pragma: no cover - process entry point
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
