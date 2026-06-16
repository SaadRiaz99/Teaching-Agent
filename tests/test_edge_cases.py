"""Edge case and boundary tests for the SMIT Teaching Agent — 25+ scenarios."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

from src.agents.teaching_agent import TeachingAgent
from src.agents.query_router import QueryRouter
from src.generation.memory import ConversationMemory
from src.rag.pipeline import RAGPipeline
from src.rag.self_rag import SelfRAGVerifier


# =====================================================================
# EDGE CASE: QUERY ROUTER (8 scenarios)
# =====================================================================

class TestQueryRouterEdgeCases:
    """Scenarios 83-90: Unusual or malformed queries."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_empty_query(self, mock_llm):
        """Scenario 83: Empty query routes to general."""
        mode = self.router.route("")
        assert mode == "general"

    def test_query_with_only_spaces(self, mock_llm):
        """Scenario 84: Whitespace-only query routes to general."""
        mode = self.router.route("     ")
        assert mode == "general"

    def test_query_with_special_characters(self, mock_llm):
        """Scenario 85: Special characters in query."""
        mode = self.router.route("!@#$%^&*()")
        assert mode in self.router.MODES  # Should not crash

    def test_sql_injection_attempt(self, mock_llm):
        """Scenario 86: SQL injection pattern in query."""
        mode = self.router.route("'; DROP TABLE students; --")
        assert mode in self.router.MODES

    def test_xss_attempt(self, mock_llm):
        """Scenario 87: XSS pattern in query."""
        mode = self.router.route("<script>alert('xss')</script>")
        assert mode in self.router.MODES

    def test_unicode_and_emoji(self, mock_llm):
        """Scenario 88: Unicode and emoji in query."""
        mode = self.router.route("What is AI? 🤖 人工智能")
        assert mode in self.router.MODES

    def test_excessively_long_query(self, mock_llm):
        """Scenario 89: Very long query (10,000+ chars)."""
        long_query = "hello " * 2000
        mode = self.router.route(long_query)
        assert mode in self.router.MODES

    def test_query_with_only_numbers(self, mock_llm):
        """Scenario 90: Query with only numbers."""
        mode = self.router.route("12345 67890")
        assert mode in self.router.MODES


# =====================================================================
# EDGE CASE: CONVERSATION MEMORY (6 scenarios)
# =====================================================================

class TestMemoryEdgeCases:
    """Scenarios 91-96: Memory boundary conditions."""

    def test_session_id_with_special_chars(self, tmp_path):
        """Scenario 91: Session ID with special characters."""
        mem = ConversationMemory(session_id="test/session:special!chars@91", max_turns=10)
        mem.add_turn(query="test", answer="answer", sources=[], confidence=0.5)
        path = mem._path()
        assert path.exists()

    def test_excessively_long_session_id(self, tmp_path):
        """Scenario 92: Very long session ID."""
        long_id = "session_" + "a" * 500
        mem = ConversationMemory(session_id=long_id, max_turns=5)
        mem.add_turn(query="q", answer="a", sources=[], confidence=0.5)
        assert len(mem.turns) == 1

    def test_add_turn_with_empty_sources(self, tmp_path):
        """Scenario 93: Turn with empty sources list."""
        mem = ConversationMemory(session_id="edge_93", max_turns=10)
        mem.add_turn(query="q", answer="a", sources=[], confidence=0.0, mode="general")
        assert mem.turns[0].sources == []

    def test_add_turn_with_zero_confidence(self, tmp_path):
        """Scenario 94: Turn with zero confidence."""
        mem = ConversationMemory(session_id="edge_94", max_turns=10)
        mem.add_turn(query="q", answer="a", sources=[], confidence=0.0)
        assert mem.turns[0].confidence == 0.0

    def test_memory_with_null_empty_queries(self, tmp_path):
        """Scenario 95: Memory handles empty query string."""
        mem = ConversationMemory(session_id="edge_95", max_turns=10)
        mem.add_turn(query="", answer="response", sources=[], confidence=0.5)
        assert mem.turns[0].query == ""

    def test_memory_clear_nonexistent_session(self, tmp_path):
        """Scenario 96: Clearing non-existent session doesn't error."""
        mem = ConversationMemory(session_id="never_created_96", max_turns=10)
        mem.clear()  # Should not raise
        assert mem.is_empty


# =====================================================================
# EDGE CASE: RAG PIPELINE (6 scenarios)
# =====================================================================

class TestRAGPipelineEdgeCases:
    """Scenarios 97-102: RAG pipeline boundary conditions."""

    def test_pipeline_with_no_chunks(self, mock_llm, mock_embedder):
        """Scenario 97: Pipeline handles empty retrieval gracefully."""
        pipeline = RAGPipeline()
        with patch.object(pipeline.retriever, "search", return_value=[]):
            result = pipeline.run(query="test", mode="general", use_cache=False)
        assert isinstance(result.answer, str)
        assert result.sources == []

    def test_pipeline_with_single_chunk(self, mock_llm, mock_embedder):
        """Scenario 98: Pipeline works with a single retrieved chunk."""
        from src.vector_store.base import ScoredChunk
        pipeline = RAGPipeline()
        chunk = ScoredChunk(chunk=MagicMock(text="single chunk", metadata={"source": "doc.md", "filename": "doc.md"}), score=0.5)
        with patch.object(pipeline, "_retrieve", return_value=[chunk]):
            result = pipeline.run(query="test", mode="general", use_cache=False)
        assert isinstance(result.answer, str)

    def test_pipeline_format_context_empty(self):
        """Scenario 99: Format context with empty list returns empty string."""
        pipeline = RAGPipeline()
        ctx = pipeline._format_context([])
        assert ctx == ""

    def test_pipeline_format_sources_empty(self):
        """Scenario 100: Format sources with empty list returns empty list."""
        pipeline = RAGPipeline()
        sources = pipeline._format_sources([])
        assert sources == []

    def test_pipeline_expand_query_fallback(self, mock_llm):
        """Scenario 101: Query expansion returns at least original query."""
        pipeline = RAGPipeline()
        queries = pipeline._expand_query("original query")
        assert len(queries) >= 1
        assert "original query" in queries

    def test_pipeline_cache_corrupted(self, tmp_path):
        """Scenario 102: Corrupted cache file returns None without crashing."""
        pipeline = RAGPipeline()
        from src.rag.pipeline import CACHE_DIR
        key = "test_corrupted_key"
        cache_file = CACHE_DIR / f"response_{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("this is not valid json", encoding="utf-8")
        result = pipeline._check_cache(key)
        assert result is None


# =====================================================================
# EDGE CASE: SELF-RAG (4 scenarios)
# =====================================================================

class TestSelfRAGEdgeCases:
    """Scenarios 103-106: Self-RAG boundary conditions."""

    def test_verify_with_empty_answer(self, mock_llm):
        """Scenario 103: Verify handles empty answer."""
        verifier = SelfRAGVerifier()
        result = verifier.verify(
            query="test", answer="",
            sources=[{"content": "ctx", "source": "doc.md", "filename": "doc.md", "score": 0.5}],
        )
        assert isinstance(result, dict)

    def test_verify_with_excessively_long_sources(self, mock_llm):
        """Scenario 104: Verify handles very long source content."""
        verifier = SelfRAGVerifier()
        result = verifier.verify(
            query="test",
            answer="answer",
            sources=[{"content": "x" * 10000, "source": "doc.md", "filename": "doc.md", "score": 0.5}],
        )
        assert isinstance(result, dict)

    def test_verify_with_many_sources(self, mock_llm):
        """Scenario 105: Verify handles many sources."""
        verifier = SelfRAGVerifier()
        sources = [{"content": f"source {i}", "source": "doc.md", "filename": "doc.md", "score": 0.5} for i in range(50)]
        result = verifier.verify(query="test", answer="answer", sources=sources)
        assert isinstance(result, dict)

    def test_verify_and_correct_with_no_sources_no_crash(self, mock_llm):
        """Scenario 106: verify_and_correct with no sources doesn't crash."""
        verifier = SelfRAGVerifier()
        result = verifier.verify_and_correct("query", "answer", [])
        assert isinstance(result, str)


# =====================================================================
# EDGE CASE: TEACHING AGENT (10 scenarios)
# =====================================================================

class TestTeachingAgentEdgeCases:
    """Scenarios 107-116: Agent boundary conditions."""

    def test_agent_handles_empty_query(self, mock_llm):
        """Scenario 107: Agent handles empty string query."""
        agent = TeachingAgent()
        result = agent.ask("")
        assert result is not None
        assert isinstance(result.answer, str)

    def test_agent_handles_special_chars_query(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 108: Agent handles special characters."""
        agent = TeachingAgent()
        result = agent.ask("!@#$%^&*()_+-=[]{}|;':,./<>?")
        assert isinstance(result.answer, str)

    def test_agent_handles_very_long_query(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 109: Agent handles very long query."""
        agent = TeachingAgent()
        long_q = "what is " + "binary " * 500 + "search"
        result = agent.ask(long_q)
        assert isinstance(result.answer, str)

    def test_agent_handles_repeated_identical_queries(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 110: Repeated identical queries (cache behavior)."""
        agent = TeachingAgent()
        q = "explain binary search"
        r1 = agent.ask(q, session_id="repeat_110")
        r2 = agent.ask(q, session_id="repeat_110")
        assert isinstance(r2.answer, str)

    def test_agent_rapid_session_switching(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 111: Rapid switching between sessions."""
        agent = TeachingAgent()
        for i in range(20):
            agent.ask(f"query_{i}", session_id=f"rapid_session_{i % 5}")
        # Verify sessions are independent
        for sid in [f"rapid_session_{i}" for i in range(5)]:
            hist = agent.get_history(sid)
            assert isinstance(hist, list)

    def test_agent_multiple_simultaneous_sessions(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 112: Multiple simultaneous sessions maintain isolation."""
        agent = TeachingAgent()
        sessions = [f"multi_session_{i}" for i in range(10)]
        for sid in sessions:
            agent.ask(f"question for {sid}", session_id=sid)
        for sid in sessions:
            hist = agent.get_history(sid)
            assert len(hist) == 1
            assert sid in hist[0]["query"]

    def test_agent_unicode_and_emoji_queries(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 113: Agent handles unicode and emoji."""
        agent = TeachingAgent()
        result = agent.ask("Explain quantum computing 🚀 量子コンピュータ")
        assert isinstance(result.answer, str)

    def test_agent_query_with_newlines_tabs(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 114: Query with newlines and tabs."""
        agent = TeachingAgent()
        result = agent.ask("what is\nmachine\tlearning?\r\n")
        assert isinstance(result.answer, str)

    def test_agent_ingest_nonexistent_path(self, mock_llm, mock_embedder, mock_vector_store):
        """Scenario 115: Ingest non-existent path returns zeros (handles gracefully)."""
        agent = TeachingAgent()
        result = agent.ingest("data/nonexistent_path_xyz")
        assert isinstance(result, dict)
        assert result["files"] == 0

    def test_agent_get_history_nonexistent_session(self, mock_llm):
        """Scenario 116: Get history for non-existent session returns empty list."""
        agent = TeachingAgent()
        history = agent.get_history("nonexistent_session_xyz")
        assert history == []


# =====================================================================
# EDGE CASE: DATA HANDLING (6 scenarios)
# =====================================================================

class TestDataHandlingEdgeCases:
    """Scenarios 117-122: Data integrity and boundary tests."""

    def test_raw_teacher_ids_unique(self, raw_teachers):
        """Scenario 117: All teacher IDs are unique."""
        ids = [t["teacher_id"] for t in raw_teachers]
        assert len(ids) == len(set(ids))

    def test_raw_student_ids_unique(self, raw_students):
        """Scenario 118: All student IDs are unique."""
        ids = [s["student_id"] for s in raw_students]
        assert len(ids) == len(set(ids))

    def test_raw_course_codes_unique(self, raw_courses):
        """Scenario 119: All course codes are unique."""
        codes = [c["course_code"] for c in raw_courses]
        assert len(codes) == len(set(codes))

    def test_raw_email_format(self, raw_teachers, raw_students):
        """Scenario 120: All emails follow SMIT domain format."""
        import re
        for t in raw_teachers:
            assert re.match(r".+@smit\.edu\.in$", t["email"]), f"Bad email: {t['email']}"
        for s in raw_students:
            assert re.match(r".+@smit\.edu\.in$", s["email"]), f"Bad email: {s['email']}"

    def test_raw_cgpa_bounds(self, raw_students):
        """Scenario 121: All CGPAs are within valid range."""
        for s in raw_students:
            assert 0.0 <= s["cgpa"] <= 10.0, f"Invalid CGPA: {s['cgpa']}"

    def test_raw_semester_bounds(self, raw_students):
        """Scenario 122: All semesters are valid (1-8)."""
        for s in raw_students:
            assert 1 <= s["semester"] <= 8, f"Invalid semester: {s['semester']}"
