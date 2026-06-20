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

from backend.agent import AgentSession

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

    return server


def main() -> None:  # pragma: no cover - process entry point
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
