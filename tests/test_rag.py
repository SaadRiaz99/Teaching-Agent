from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from app.rag.chunking import TextChunker
from app.rag.ingestion import DocumentIngestor
from app.rag.embeddings import EmbeddingManager
from app.models.schemas import ChatRequest, QuizRequest


class TestTextChunker:
    def test_chunk_text_basic(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        text = "Hello world. " * 100
        docs = chunker.chunk_text(text)
        assert len(docs) > 0
        for doc in docs:
            assert "chunk_index" in doc.metadata

    def test_chunk_text_empty(self):
        chunker = TextChunker()
        docs = chunker.chunk_text("")
        assert len(docs) == 0 or (len(docs) == 1 and docs[0].page_content == "")

    def test_chunk_documents_empty_list(self):
        chunker = TextChunker()
        docs = chunker.chunk_documents([])
        assert docs == []


class TestDocumentIngestor:
    def test_ingest_text(self):
        ingestor = DocumentIngestor()
        docs = ingestor.ingest_text("This is a test document for SMIT courses.")
        assert len(docs) == 1
        assert "test" in docs[0].page_content

    def test_ingest_inline_text_metadata(self):
        ingestor = DocumentIngestor()
        docs = ingestor.ingest_text("Python is a programming language.", source_name="test_source")
        assert docs[0].metadata["source"] == "test_source"

    def test_unsupported_extension(self):
        ingestor = DocumentIngestor()
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestor.ingest_file("test.xyz")


class TestChatRequest:
    def test_valid_chat_request(self):
        req = ChatRequest(message="What is Python?")
        assert req.message == "What is Python?"

    def test_chat_request_min_length(self):
        with pytest.raises(ValueError):
            ChatRequest(message="")

    def test_chat_request_max_length(self):
        with pytest.raises(ValueError):
            ChatRequest(message="x" * 4097)


class TestQuizRequest:
    def test_valid_quiz_request(self):
        req = QuizRequest(num_questions=5, difficulty="medium")
        assert req.num_questions == 5
        assert req.difficulty == "medium"

    def test_quiz_request_invalid_difficulty(self):
        with pytest.raises(ValueError):
            QuizRequest(difficulty="extreme")

    def test_quiz_request_num_questions_bounds(self):
        with pytest.raises(ValueError):
            QuizRequest(num_questions=0)
        with pytest.raises(ValueError):
            QuizRequest(num_questions=21)


class TestEmbeddingManager:
    def test_initialization(self):
        mgr = EmbeddingManager()
        assert mgr.embeddings is not None
        assert mgr.persist_directory == "vectordb"

    def test_get_collection_stats(self):
        mgr = EmbeddingManager()
        stats = mgr.get_collection_stats()
        assert "collection" in stats
        assert "total_documents" in stats
