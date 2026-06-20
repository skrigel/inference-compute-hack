import asyncio
import unittest
from unittest.mock import patch

from data.schema import Chunk, ChunkMeta


class FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


class McpKnowledgeToolTests(unittest.TestCase):
    def test_agent_session_ingests_arbitrary_documents_as_source_compartment(self):
        from backend.agent import AgentSession
        from backend.schemas import FreshDocument
        from inference.mock_scorer import MockScorer

        session = AgentSession(MockScorer())
        result = asyncio.run(
            session.ingest_documents(
                "papers",
                [
                    FreshDocument(
                        title="Reward variance predicts lift",
                        text="reward variance and heldout lift for RL data selection",
                        type="paper",
                        category="cs.LG",
                        year=2026,
                        path="arxiv:0000.00001",
                        repo="arxiv",
                    )
                ],
            )
        )

        self.assertEqual(result["source_id"], "papers")
        self.assertEqual(result["n_chunks"], 1)
        query = asyncio.run(session.query("reward variance", top_k=1))
        self.assertEqual(query["corpus_total"], 1)
        self.assertEqual(query["results"][0]["title"], "Reward variance predicts lift")

    def test_mcp_registers_rag_ours_and_comparison_tools(self):
        from backend import mcp_server

        with patch.object(mcp_server, "_MCP_AVAILABLE", True), patch.object(
            mcp_server, "FastMCP", FakeFastMCP
        ):
            server = mcp_server.build_server()

        self.assertIn("ingest_source", server.tools)
        self.assertIn("search_source_ours", server.tools)
        self.assertIn("search_source_rag", server.tools)
        self.assertIn("compare_source_search", server.tools)

        ingest = asyncio.run(
            server.tools["ingest_source"](
                source_id="code",
                source_kind="code",
                size=12,
            )
        )
        self.assertEqual(ingest["source_id"], "code")
        self.assertEqual(ingest["n_chunks"], 12)

        comparison = asyncio.run(
            server.tools["compare_source_search"](
                source_id="code",
                query="retry without backoff",
                refinements=["only python"],
                top_k=5,
            )
        )

        self.assertEqual(comparison["source_id"], "code")
        self.assertIn("rag", comparison)
        self.assertIn("ours", comparison)
        self.assertGreater(comparison["work_units_speedup"], 1.0)
        self.assertGreater(comparison["ours"]["selected_count"], 0)

    def test_mcp_comparison_supports_browsecomp_source_kind(self):
        from backend import mcp_server

        browsecomp_slice = [
            Chunk(
                chunk_id="bc-1",
                doc_id="browsecomp:1",
                type="paper",
                title="BrowseComp reward variance",
                text="reward variance and verifier scores predict post-RL lift",
                meta=ChunkMeta("browsecomp", 2026, "https://example.test/1", None, None, "browsecomp"),
            ),
            Chunk(
                chunk_id="bc-2",
                doc_id="browsecomp:2",
                type="paper",
                title="BrowseComp trajectory entropy",
                text="trajectory entropy and held-out benchmark improvement for RLAIF data",
                meta=ChunkMeta("browsecomp", 2026, "https://example.test/2", None, None, "browsecomp"),
            ),
        ]

        with patch.object(mcp_server, "_MCP_AVAILABLE", True), patch.object(
            mcp_server, "FastMCP", FakeFastMCP
        ), patch("data.browsecomp_loader.load_browsecomp_corpus", return_value=browsecomp_slice):
            server = mcp_server.build_server()

        comparison = asyncio.run(
            server.tools["compare_source_search"](
                source_id="browsecomp-demo",
                source_kind="browsecomp",
                size=2,
                query="reward variance verifier scores",
                refinements=["must discuss RLAIF data"],
                top_k=2,
            )
        )

        self.assertEqual(comparison["source_id"], "browsecomp-demo")
        self.assertEqual(comparison["rag"]["tool_name"], "search_source_rag")
        self.assertEqual(comparison["ours"]["tool_name"], "search_source_ours")
        self.assertEqual(comparison["rag"]["n_docs"], 2)
        self.assertEqual(comparison["ours"]["n_docs"], 2)
        self.assertEqual(comparison["ours"]["corpus_total"], 2)
        self.assertGreaterEqual(comparison["work_units_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
