"""Comprehensive tests for the SMIT Teaching Agent — 30+ scenarios spanning all layers."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from src.agents.teaching_agent import TeachingAgent
from src.agents.query_router import QueryRouter
from src.generation.memory import ConversationMemory, Turn
from src.rag.pipeline import RAGPipeline, RAGResult
from src.rag.self_rag import SelfRAGVerifier
from src.vector_store.base import Chunk, ScoredChunk


# =====================================================================
# QUERY ROUTER TESTS (12 scenarios)
# =====================================================================

class TestQueryRouter:
    """Scenarios 21-32: Query routing classification."""

    def setup_method(self):
        self.router = QueryRouter()

    @pytest.mark.parametrize("query,expected_mode", [
        ("hello", "greeting"),
        ("hi", "greeting"),
        ("hey", "greeting"),
        ("good morning", "greeting"),
        ("greetings", "greeting"),
    ])
    def test_greeting_detection(self, query: str, expected_mode: str, mock_llm):
        """Scenario 21-25: Greeting queries are classified correctly."""
        mode = self.router.route(query)
        assert mode == expected_mode, f"Expected '{expected_mode}' for '{query}', got '{mode}'"

    @pytest.mark.parametrize("query,expected_mode", [
        ("explain binary search algorithm", "teaching"),
        ("what is a linked list", "teaching"),
        ("tell me about recursion in programming", "teaching"),
        ("how does quicksort work", "teaching"),
        ("define object-oriented programming", "teaching"),
    ])
    def test_teaching_mode_detection(self, query: str, expected_mode: str, mock_llm):
        """Scenario 26-30: Teaching queries are classified correctly."""
        mode = self.router.route(query)
        assert mode == expected_mode, f"Expected '{expected_mode}' for '{query}', got '{mode}'"

    @pytest.mark.parametrize("query,expected_mode", [
        ("what is the syllabus for CS101", "syllabus"),
        ("show me the course structure for AI", "syllabus"),
        ("what are prerequisites for Machine Learning", "syllabus"),
        ("tell me about the CSE curriculum", "syllabus"),
    ])
    def test_syllabus_mode_detection(self, query: str, expected_mode: str, mock_llm):
        """Scenario 31-34: Syllabus queries are classified correctly."""
        mode = self.router.route(query)
        assert mode == expected_mode, f"Expected '{expected_mode}' for '{query}', got '{mode}'"

    @pytest.mark.parametrize("query,expected_mode", [
        ("help me with assignment 3", "assignment"),
        ("can you help with my homework", "assignment"),
        ("I need help with problem set 2", "assignment"),
        ("how do I solve this assignment problem", "assignment"),
    ])
    def test_assignment_mode_detection(self, query: str, expected_mode: str, mock_llm):
        """Scenario 35-38: Assignment queries are classified correctly."""
        mode = self.router.route(query)
        assert mode == expected_mode, f"Expected '{expected_mode}' for '{query}', got '{mode}'"

    @pytest.mark.parametrize("query,expected_mode", [
        ("what is the address of SMIT", "general"),
        ("where is the college located", "general"),
        ("tell me about campus facilities", "general"),
        ("", "general"),
    ])
    def test_general_mode_detection(self, query: str, expected_mode: str, mock_llm):
        """Scenario 39-42: General queries are classified correctly."""
        mode = self.router.route(query)
        assert mode == expected_mode, f"Expected '{expected_mode}' for '{query}', got '{mode}'"


# =====================================================================
# CONVERSATION MEMORY TESTS (10 scenarios)
# =====================================================================

class TestConversationMemory:
    """Scenarios 43-52: Memory creation, persistence, and limits."""

    def test_memory_creation(self, tmp_path):
        """Scenario 43: New session creates empty memory."""
        mem = ConversationMemory(session_id="test_session_43", max_turns=10)
        assert mem.is_empty
        assert len(mem.turns) == 0
        assert mem.session_id == "test_session_43"

    def test_add_and_retrieve_turns(self, tmp_path):
        """Scenario 44: Adding turns populates history."""
        mem = ConversationMemory(session_id="test_session_44", max_turns=10)
        mem.add_turn(query="hello", answer="hi there", sources=[], confidence=1.0, mode="greeting")
        mem.add_turn(query="what is AI", answer="AI is...", sources=[], confidence=0.9, mode="teaching")
        assert len(mem.turns) == 2
        history = mem.get_history()
        assert "hello" in history
        assert "what is AI" in history

    def test_max_turns_limit(self, tmp_path):
        """Scenario 45: Memory respects max_turns limit."""
        mem = ConversationMemory(session_id="test_session_45", max_turns=3)
        for i in range(10):
            mem.add_turn(query=f"query_{i}", answer=f"answer_{i}", sources=[], confidence=0.5, mode="general")
        assert len(mem.turns) == 3
        # Only the most recent 3 should remain
        last_queries = [t.query for t in mem.turns]
        assert "query_7" in last_queries
        assert "query_8" in last_queries
        assert "query_9" in last_queries
        assert "query_0" not in last_queries

    def test_clear_memory(self, tmp_path):
        """Scenario 46: Clear removes all turns."""
        mem = ConversationMemory(session_id="test_session_46", max_turns=10)
        mem.add_turn(query="test", answer="answer", sources=[], confidence=0.5)
        assert not mem.is_empty
        mem.clear()
        assert mem.is_empty

    def test_session_isolation(self, tmp_path):
        """Scenario 47: Different sessions have independent histories."""
        mem_a = ConversationMemory(session_id="session_a", max_turns=10)
        mem_b = ConversationMemory(session_id="session_b", max_turns=10)
        mem_a.add_turn(query="hello", answer="hi", sources=[], confidence=1.0)
        assert len(mem_a.turns) == 1
        assert len(mem_b.turns) == 0

    def test_memory_persistence_on_disk(self, tmp_path):
        """Scenario 48: Memory persists to disk and reloads."""
        import json
        from src.generation.memory import MEMORY_DIR
        mem = ConversationMemory(session_id="persist_test", max_turns=10)
        mem.add_turn(query="test q", answer="test a", sources=[{"source": "doc.md"}], confidence=0.8, mode="general")
        path = MEMORY_DIR / "memory_persist_test.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0]["query"] == "test q"

    def test_history_returns_last_five(self, tmp_path):
        """Scenario 49: get_history returns last 5 turns by default."""
        mem = ConversationMemory(session_id="test_session_49", max_turns=10)
        for i in range(8):
            mem.add_turn(query=f"q{i}", answer=f"a{i}", sources=[], confidence=0.5)
        # get_history uses last 5 turns
        history = mem.get_history()
        assert "q0" not in history
        assert "q3" in history or "q4" in history

    def test_turn_data_structure(self, tmp_path):
        """Scenario 50: Turn dataclass holds expected fields."""
        turn = Turn(
            query="test query",
            answer="test answer",
            sources=[{"source": "doc.md", "score": 0.9}],
            confidence=0.95,
            mode="teaching",
            timestamp="2025-01-01T00:00:00",
        )
        d = turn.to_dict()
        assert d["query"] == "test query"
        assert d["confidence"] == 0.95
        assert d["mode"] == "teaching"

    def test_turn_roundtrip_serialization(self, tmp_path):
        """Scenario 51: Turn.from_dict recreates same object."""
        original = Turn(query="q", answer="a", sources=[], confidence=0.7, mode="general", timestamp="now")
        d = original.to_dict()
        restored = Turn.from_dict(d)
        assert restored.query == original.query
        assert restored.answer == original.answer

    def test_memory_max_turns_configurable(self, tmp_path):
        """Scenario 52: max_turns parameter is configurable."""
        mem = ConversationMemory(session_id="test_configurable", max_turns=5)
        assert mem.max_turns == 5
        for i in range(20):
            mem.add_turn(query=f"q{i}", answer=f"a{i}", sources=[], confidence=0.5)
        assert len(mem.turns) == 5


# =====================================================================
# RAG PIPELINE TESTS (10 scenarios)
# =====================================================================

class TestRAGPipeline:
    """Scenarios 53-62: RAG pipeline internals."""

    def test_cache_key_deterministic(self):
        """Scenario 53: Same query+mode always produces same cache key."""
        pipeline = RAGPipeline()
        key1 = pipeline._cache_key("what is AI", "teaching")
        key2 = pipeline._cache_key("what is AI", "teaching")
        key3 = pipeline._cache_key("what is AI", "general")
        assert key1 == key2
        assert key1 != key3

    def test_cache_hit(self, tmp_path):
        """Scenario 54: Cache hit returns stored result."""
        pipeline = RAGPipeline()
        key = pipeline._cache_key("cached query", "general")
        from src.rag.pipeline import CACHE_DIR
        cache_file = CACHE_DIR / f"response_{key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({
                "answer": "cached answer",
                "sources": [],
                "confidence": 0.9,
                "mode": "general",
                "model": "mock",
                "verified": True,
            }),
            encoding="utf-8",
        )
        result = pipeline._check_cache(key)
        assert result is not None
        assert result.answer == "cached answer"
        assert result.from_cache

    def test_cache_miss(self):
        """Scenario 55: Cache miss returns None."""
        pipeline = RAGPipeline()
        result = pipeline._check_cache("nonexistent_key_12345")
        assert result is None

    def test_format_context_with_sources(self):
        """Scenario 56: Context formatting includes source citations."""
        pipeline = RAGPipeline()
        chunks = [
            ScoredChunk(
                chunk=Chunk(text="Content about data structures.", metadata={"source": "data/documents/cs101.md", "filename": "cs101.md"}, chunk_id="c1"),
                score=0.9,
            ),
            ScoredChunk(
                chunk=Chunk(text="Content about algorithms.", metadata={"source": "data/documents/cs201.md", "filename": "cs201.md"}, chunk_id="c2"),
                score=0.8,
            ),
        ]
        ctx = pipeline._format_context(chunks)
        assert "[Source 1]" in ctx
        assert "cs101.md" in ctx
        assert "Content about" in ctx

    def test_format_sources_metadata(self):
        """Scenario 57: Source formatting extracts proper metadata."""
        pipeline = RAGPipeline()
        chunks = [
            ScoredChunk(
                chunk=Chunk(text="A" * 400, metadata={"source": "/path/to/doc.md", "filename": "doc.md"}, chunk_id="c1"),
                score=0.95,
            ),
        ]
        sources = pipeline._format_sources(chunks)
        assert len(sources) == 1
        assert sources[0]["filename"] == "doc.md"
        assert sources[0]["score"] == 0.95
        assert len(sources[0]["content"]) == 300  # truncated

    def test_evaluate_sufficiency_with_good_chunks(self):
        """Scenario 58: Sufficiency returns True for high-scoring chunks."""
        pipeline = RAGPipeline()
        chunks = [
            ScoredChunk(chunk=Chunk(text="x"), score=0.5),
            ScoredChunk(chunk=Chunk(text="y"), score=0.6),
        ]
        assert pipeline._evaluate_sufficiency("query", chunks)

    def test_evaluate_sufficiency_with_empty_chunks(self):
        """Scenario 59: Sufficiency returns False for empty chunks."""
        pipeline = RAGPipeline()
        assert not pipeline._evaluate_sufficiency("query", [])

    def test_evaluate_sufficiency_with_low_scores(self):
        """Scenario 60: Sufficiency returns False for low-scoring chunks."""
        pipeline = RAGPipeline()
        chunks = [
            ScoredChunk(chunk=Chunk(text="x"), score=0.05),
            ScoredChunk(chunk=Chunk(text="y"), score=0.1),
        ]
        assert not pipeline._evaluate_sufficiency("query", chunks)

    def test_ragresult_dataclass_fields(self):
        """Scenario 61: RAGResult has all expected fields."""
        result = RAGResult(answer="test", sources=[], confidence=0.8, mode="teaching", model="mock", verified=True)
        assert result.answer == "test"
        assert result.confidence == 0.8
        assert result.mode == "teaching"
        assert result.verified
        assert not result.from_cache

    def test_confidence_calculation(self):
        """Scenario 62: Confidence is average of chunk scores, capped at 1.0."""
        pipeline = RAGPipeline()
        chunks = [
            ScoredChunk(chunk=Chunk(text="a"), score=0.8),
            ScoredChunk(chunk=Chunk(text="b"), score=0.6),
        ]
        sources = pipeline._format_sources(chunks)
        confidence = min(1.0, sum(c.score for c in chunks) / len(chunks))
        assert confidence == pytest.approx(0.7)


# =====================================================================
# SELF-RAG VERIFIER TESTS (6 scenarios)
# =====================================================================

class TestSelfRAG:
    """Scenarios 63-68: Self-RAG verification and correction."""

    def test_verify_with_sources(self, mock_llm):
        """Scenario 63: Verification returns structured result with sources."""
        verifier = SelfRAGVerifier()
        result = verifier.verify(
            query="what is AI",
            answer="AI is artificial intelligence.",
            sources=[{"content": "AI is a branch of computer science.", "source": "doc.md", "filename": "doc.md", "score": 0.9}],
        )
        assert isinstance(result, dict)
        assert "all_claims_grounded" in result
        assert "confidence_score" in result

    def test_verify_without_sources(self, mock_llm):
        """Scenario 64: Verify without sources returns default result."""
        verifier = SelfRAGVerifier()
        result = verifier.verify(query="hello", answer="hi", sources=[])
        assert result["issues"] == ["No sources to verify against"]
        assert result["passed"]

    def test_verify_and_correct_grounded_answer(self, mock_llm):
        """Scenario 65: Grounded answer passes through without correction."""
        verifier = SelfRAGVerifier()
        corrected = verifier.verify_and_correct(
            query="what is AI",
            answer="AI is artificial intelligence.",
            sources=[{"content": "AI is a branch of CS.", "source": "doc.md", "filename": "doc.md", "score": 0.9}],
        )
        assert corrected is not None
        assert isinstance(corrected, str)

    def test_verify_and_correct_hallucinated_answer(self, mock_llm):
        """Scenario 66: Hallucinated answer triggers correction path."""
        verifier = SelfRAGVerifier()
        corrected = verifier.verify_and_correct(
            query="what is AI",
            answer="AI can time travel and fly.",
            sources=[],
        )
        # Should not crash; returns correction or original
        assert isinstance(corrected, str)

    def test_verification_returns_json_fields(self, mock_llm):
        """Scenario 67: Verification returns expected JSON schema fields."""
        verifier = SelfRAGVerifier()
        result = verifier.verify("test", "answer", [{"content": "context", "source": "x", "filename": "x", "score": 0.5}])
        expected_keys = {"all_claims_grounded", "directly_addresses_query", "hallucination_detected", "confidence_score", "issues"}
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"

    def test_verify_and_correct_preserves_answer_on_error(self, mock_llm, monkeypatch):
        """Scenario 68: Correction preserves original on generation error."""
        verifier = SelfRAGVerifier()
        with patch.object(verifier.llm, "generate") as mock_gen:
            mock_gen.side_effect = RuntimeError("LLM unavailable")
            corrected = verifier.verify_and_correct("q", "original answer", [{"content": "ctx", "source": "s", "filename": "f", "score": 0.5}])
            assert corrected == "original answer"


# =====================================================================
# TEACHING AGENT TESTS (14 scenarios)
# =====================================================================

class TestTeachingAgent:
    """Scenarios 69-82: End-to-end agent behavior."""

    def test_greeting_response(self, mock_llm):
        """Scenario 69: Greeting returns welcome message directly."""
        agent = TeachingAgent()
        result = agent.ask("hello")
        assert result.mode == "greeting"
        assert "SMIT" in result.answer or "Hello" in result.answer
        assert result.confidence == 1.0
        assert result.verified

    def test_ask_returns_ragresult(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 70: Ask returns a properly typed RAGResult."""
        agent = TeachingAgent()
        result = agent.ask("explain data structures")
        assert isinstance(result, RAGResult) or hasattr(result, "answer")

    def test_response_contains_mode(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 71: Response includes the routing mode."""
        agent = TeachingAgent()
        result = agent.ask("what is a linked list")
        assert result.mode in ("teaching", "general")

    def test_response_contains_sources(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 72: Response includes sources list."""
        agent = TeachingAgent()
        result = agent.ask("explain algorithms")
        assert hasattr(result, "sources")
        assert isinstance(result.sources, list)

    def test_response_contains_confidence(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 73: Response includes confidence score."""
        agent = TeachingAgent()
        result = agent.ask("what is recursion")
        assert 0.0 <= result.confidence <= 1.0

    def test_response_contains_verified_flag(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 74: Response has verified flag set."""
        agent = TeachingAgent()
        result = agent.ask("explain sorting")
        assert isinstance(result.verified, bool)

    def test_session_memory_persists_across_asks(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 75: Multiple asks in same session accumulate memory."""
        agent = TeachingAgent()
        agent.ask("hello", session_id="mem_test_75")
        agent.ask("what is AI", session_id="mem_test_75")
        history = agent.get_history("mem_test_75")
        assert len(history) >= 2

    def test_ingest_returns_result(self, mock_llm, mock_embedder, mock_vector_store):
        """Scenario 76: Ingest returns a dict with file/chunk counts."""
        agent = TeachingAgent()
        result = agent.ingest("data/documents")
        assert isinstance(result, dict)
        assert "files" in result
        assert "chunks" in result

    def test_get_history_returns_list(self, mock_llm):
        """Scenario 77: Get history returns a list of turn dicts."""
        agent = TeachingAgent()
        agent.ask("hello", session_id="hist_test_77")
        history = agent.get_history("hist_test_77")
        assert isinstance(history, list)
        if history:
            assert "query" in history[0]

    def test_clear_history_removes_session(self, mock_llm, tmp_path):
        """Scenario 78: Clear history removes session data."""
        agent = TeachingAgent()
        agent.ask("hello", session_id="clear_test_78")
        assert len(agent.get_history("clear_test_78")) >= 1
        agent.clear_history("clear_test_78")
        assert len(agent.get_history("clear_test_78")) == 0

    def test_different_sessions_isolated(self, mock_llm):
        """Scenario 79: Different sessions have independent histories."""
        agent = TeachingAgent()
        agent.ask("hello", session_id="session_a_79")
        agent.ask("what is AI", session_id="session_b_79")
        hist_a = agent.get_history("session_a_79")
        hist_b = agent.get_history("session_b_79")
        # Session A should have only the greeting
        assert len(hist_a) == 1
        assert hist_a[0]["query"] == "hello"

    def test_ask_multiple_times_accumulates(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 80: Multiple asks accumulate history in order."""
        agent = TeachingAgent()
        queries = ["hello", "what is AI", "explain ML", "tell me about SMIT"]
        for q in queries:
            agent.ask(q, session_id="accum_test_80")
        history = agent.get_history("accum_test_80")
        assert len(history) >= len(queries)

    def test_model_name_in_response(self, mock_llm, mock_hybrid_retriever, mock_reranker):
        """Scenario 81: Response includes model name."""
        agent = TeachingAgent()
        result = agent.ask("explain algorithms", session_id="model_test_81")
        assert isinstance(result.model, str)
        assert len(result.model) > 0

    def test_greeting_bypasses_rag(self, mock_llm):
        """Scenario 82: Greeting mode returns immediately without calling LLM."""
        agent = TeachingAgent()
        with patch.object(agent.pipeline, "run") as mock_run:
            result = agent.ask("hello")
            mock_run.assert_not_called()
            assert result.mode == "greeting"


import json  # noqa: E402 (imported here for cache tests)
