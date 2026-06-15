from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.teaching_agent import TeachingAgent
from src.vector_store import get_vector_store

app = FastAPI(
    title="SMIT Teaching Agent API",
    description="RAG-powered educational assistant for Sikkim Manipal Institute of Technology",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = TeachingAgent()


class AskRequest(BaseModel):
    query: str
    session_id: str = "default"
    no_cache: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    mode: str
    model: str
    verified: bool
    from_cache: bool = False


class IngestResponse(BaseModel):
    files: int
    chunks: int
    documents: int


class HistoryResponse(BaseModel):
    turns: list


@app.get("/health")
async def health():
    store = get_vector_store()
    return {
        "status": "healthy",
        "chunks_in_db": store.count(),
        "llm_provider": agent.pipeline.llm.name if hasattr(agent.pipeline, "llm") else "unknown",
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        result = agent.ask(request.query, session_id=request.session_id)
        return AskResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            mode=result.mode,
            model=result.model,
            verified=result.verified,
            from_cache=result.from_cache,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    import tempfile
    import os

    suffix = os.path.splitext(file.filename or ".txt")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = agent.ingest(tmp_path)
        return IngestResponse(**result)
    finally:
        os.unlink(tmp_path)


@app.post("/ingest/directory")
async def ingest_directory(path: str = Query(..., description="Path to directory")):
    from pathlib import Path
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {path}")
    try:
        result = agent.ingest(str(p))
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    turns = agent.get_history(session_id)
    return HistoryResponse(turns=turns)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    agent.clear_history(session_id)
    return {"status": "cleared"}


@app.get("/stats")
async def stats():
    store = get_vector_store()
    return {
        "total_chunks": store.count(),
        "vector_db_type": "ChromaDB",
    }
