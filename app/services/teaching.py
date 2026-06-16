from __future__ import annotations

import json
import uuid
from typing import Any

from langchain.schema import Document

from app.models.schemas import (
    ChatResponse,
    QuizQuestion,
    QuizResponse,
    SourceCitation,
    SummaryResponse,
)
from app.rag.ingestion import DocumentIngestor
from app.rag.chunking import TextChunker
from app.rag.embeddings import EmbeddingManager
from app.rag.retrieval import RetrievalManager
from app.rag.generator import ResponseGenerator
from app.utils.logger import logger


class TeachingService:
    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.chunker = TextChunker()
        self.embeddings = EmbeddingManager()
        self.retriever = RetrievalManager()
        self.generator = ResponseGenerator()
        self._conversations: dict[str, list[dict[str, str]]] = {}
        logger.info("TeachingService initialized")

    def upload_document(
        self, file_content: bytes, filename: str
    ) -> dict[str, Any]:
        documents = self.ingestor.ingest_uploaded_file(file_content, filename)
        chunked = self.chunker.chunk_documents(documents)
        ids = self.embeddings.add_documents(chunked)
        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}'",
            "filename": filename,
            "document_id": str(uuid.uuid4()),
            "chunks_count": len(chunked),
        }

    def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> ChatResponse:
        conv_id = conversation_id or str(uuid.uuid4())

        if conv_id not in self._conversations:
            self._conversations[conv_id] = []

        self._conversations[conv_id].append({"role": "user", "content": message})

        history = self._conversations[conv_id][-5:]
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )
        augmented_message = (
            f"Conversation history:\n{history_text}\n\n"
            f"Current question: {message}"
        )

        chunks = self.retriever.retrieve(augmented_message, top_k=top_k)

        answer, grounded = self.generator.generate_answer(
            message, chunks, temperature=temperature, max_tokens=max_tokens
        )

        sources = []
        for doc, score in chunks:
            sources.append(
                SourceCitation(
                    content=doc.page_content[:300],
                    document=doc.metadata.get("source", "Unknown"),
                    score=round(score, 4),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                )
            )

        self._conversations[conv_id].append(
            {"role": "assistant", "content": answer}
        )

        return ChatResponse(
            answer=answer,
            conversation_id=conv_id,
            sources=sources,
            grounded=grounded,
        )

    def generate_quiz(
        self,
        document_ids: list[str] | None = None,
        num_questions: int = 5,
        difficulty: str = "medium",
        topic: str | None = None,
    ) -> QuizResponse:
        all_sources = self.retriever.get_all_document_ids()
        docs_used = document_ids or all_sources

        filter_dict = None
        if document_ids:
            filter_dict = {"source": {"$in": document_ids}}

        chunks = self.retriever.retrieve(
            "course content quiz generation", top_k=20
        )

        if not chunks:
            logger.warning("No documents found for quiz generation")
            return QuizResponse(
                questions=[],
                document_ids=docs_used,
                total_questions=0,
            )

        raw = self.generator.generate_quiz(
            chunks,
            num_questions=num_questions,
            difficulty=difficulty,
            topic=topic,
        )

        questions = self._parse_quiz_response(raw, num_questions)

        return QuizResponse(
            questions=questions,
            document_ids=docs_used,
            total_questions=len(questions),
        )

    def generate_summary(
        self,
        document_ids: list[str] | None = None,
        max_length: int = 500,
    ) -> SummaryResponse:
        chunks = self.retriever.retrieve("content summary", top_k=20)
        docs_used = document_ids or self.retriever.get_all_document_ids()

        summary = self.generator.generate_summary(
            chunks, max_length=max_length
        )

        return SummaryResponse(
            summary=summary,
            document_ids=docs_used,
        )

    def get_recommendations(
        self,
        question: str,
        top_k: int | None = None,
    ) -> str:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        return self.generator.generate_recommendations(question, chunks)

    def get_conversation_history(
        self, conversation_id: str
    ) -> list[dict[str, str]]:
        return self._conversations.get(conversation_id, [])

    def get_stats(self) -> dict[str, Any]:
        return self.embeddings.get_collection_stats()

    def _parse_quiz_response(
        self, raw: str, expected: int
    ) -> list[QuizQuestion]:
        try:
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("questions", data.get("quiz", [data]))

            questions = []
            for item in data[:expected]:
                questions.append(
                    QuizQuestion(
                        question=item.get("question", ""),
                        options=item.get("options", []),
                        correct_answer=item.get("correct_answer", ""),
                        explanation=item.get("explanation", ""),
                        difficulty=item.get("difficulty", "medium"),
                    )
                )
            return questions
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse quiz response: {e}")
            return []
