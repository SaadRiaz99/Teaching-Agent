from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "SMIT AI Teaching Agent"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    document_id: str | None = None
    chunks_count: int = 0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    conversation_id: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_k: int | None = None


class SourceCitation(BaseModel):
    content: str
    document: str
    score: float
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[SourceCitation] = []
    grounded: bool = True


class QuizRequest(BaseModel):
    document_ids: list[str] | None = None
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    topic: str | None = None


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    difficulty: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
    document_ids: list[str]
    total_questions: int


class SummaryRequest(BaseModel):
    document_ids: list[str]
    max_length: int = 500


class SummaryResponse(BaseModel):
    summary: str
    document_ids: list[str]


class ChatHistory(BaseModel):
    conversation_id: str
    messages: list[dict[str, str]] = []
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
