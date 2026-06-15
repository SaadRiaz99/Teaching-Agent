from typing import Optional

from src.rag.pipeline import RAGPipeline, RAGResult
from src.rag.self_rag import SelfRAGVerifier
from src.generation.memory import ConversationMemory
from .query_router import QueryRouter


class TeachingAgent:
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.verifier = SelfRAGVerifier()
        self.router = QueryRouter()
        self._memories: dict[str, ConversationMemory] = {}

    def _get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(session_id=session_id)
        return self._memories[session_id]

    def ask(self, query: str, session_id: str = "default") -> RAGResult:
        mode = self.router.route(query)

        if mode == "greeting":
            return RAGResult(
                answer="Hello! I'm the SMIT Teaching Assistant. I can help you with course concepts, syllabus information, assignment guidance, and general questions about Sikkim Manipal Institute of Technology. How can I help you today?",
                sources=[],
                confidence=1.0,
                mode="greeting",
                model="",
                verified=True,
            )

        memory = self._get_memory(session_id)
        result = self.pipeline.run(
            query=query,
            session_id=session_id,
            mode=mode,
            memory=memory,
        )

        verified_answer = self.verifier.verify_and_correct(query, result.answer, result.sources)
        result.answer = verified_answer
        result.verified = True

        memory.add_turn(
            query=query,
            answer=result.answer,
            sources=[{k: v for k, v in s.items() if k in ("source", "filename", "score")} for s in result.sources],
            confidence=result.confidence,
            mode=mode,
        )

        return result

    def ingest(self, path: str) -> dict:
        from src.ingestion.processor import IngestionPipeline
        pipeline = IngestionPipeline()
        return pipeline.ingest(path, show_progress=True)

    def get_history(self, session_id: str = "default") -> list:
        memory = self._get_memory(session_id)
        return [t.to_dict() for t in memory.turns]

    def clear_history(self, session_id: str = "default"):
        if session_id in self._memories:
            self._memories[session_id].clear()
            del self._memories[session_id]
