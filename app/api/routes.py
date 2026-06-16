from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    QuizRequest,
    QuizResponse,
    SummaryRequest,
    SummaryResponse,
    UploadResponse,
)
from app.services.teaching import TeachingService
from app.utils.logger import logger

router = APIRouter()
service = TeachingService()

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = f".{file.filename.rsplit('.', 1)[-1].lower()}" if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    try:
        result = service.upload_document(content, file.filename)
        return UploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    try:
        response = service.generate_quiz(
            document_ids=request.document_ids,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            topic=request.topic,
        )
        return response
    except Exception as e:
        logger.exception("Quiz generation failed")
        raise HTTPException(
            status_code=500, detail=f"Quiz generation failed: {str(e)}"
        )


@router.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    try:
        response = service.generate_summary(
            document_ids=request.document_ids,
            max_length=request.max_length,
        )
        return response
    except Exception as e:
        logger.exception("Summary generation failed")
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {str(e)}"
        )


@router.get("/recommendations")
async def get_recommendations(
    question: str = Query(..., min_length=1, max_length=1024),
    top_k: int = Query(default=4, ge=1, le=20),
):
    try:
        recommendations = service.get_recommendations(question, top_k=top_k)
        return {"question": question, "recommendations": recommendations}
    except Exception as e:
        logger.exception("Recommendations failed")
        raise HTTPException(
            status_code=500, detail=f"Recommendations failed: {str(e)}"
        )


@router.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    history = service.get_conversation_history(conversation_id)
    return {"conversation_id": conversation_id, "messages": history}


@router.get("/stats")
async def get_stats():
    return service.get_stats()
