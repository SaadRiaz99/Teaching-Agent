from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

dashboard_path = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_path.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")

_agent: Optional[TeachingAgent] = None


def get_agent() -> TeachingAgent:
    global _agent
    if _agent is None:
        _agent = TeachingAgent()
    return _agent


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
    try:
        agent = get_agent()
        llm_name = agent.pipeline.llm.name if hasattr(agent.pipeline, "llm") else "unknown"
    except Exception:
        llm_name = "degraded (agent failed to load)"
    return {
        "status": "healthy",
        "chunks_in_db": store.count(),
        "llm_provider": llm_name,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        result = get_agent().ask(request.query, session_id=request.session_id)
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
        result = get_agent().ingest(tmp_path)
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
        result = get_agent().ingest(str(p))
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    turns = get_agent().get_history(session_id)
    return HistoryResponse(turns=turns)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    get_agent().clear_history(session_id)
    return {"status": "cleared"}


@app.get("/stats")
async def stats():
    store = get_vector_store()
    return {
        "total_chunks": store.count(),
        "vector_db_type": "ChromaDB",
    }
