import asyncio
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
