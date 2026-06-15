import hashlib
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.query_expander import QueryExpander
from src.generation.generator import get_llm, LLMResponse
from src.generation.prompts import PromptTemplates
from src.generation.memory import ConversationMemory
from src.vector_store.base import ScoredChunk


CACHE_DIR = Path("data/cache")


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    mode: str
    model: str
    verified: bool = False
    from_cache: bool = False


class RAGPipeline:
    def __init__(self, max_retries: int = 3):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.llm = get_llm()
        self.expander = QueryExpander(llm=self.llm)
        self.max_retries = max_retries
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, query: str, mode: str) -> str:
        return hashlib.sha256(f"{query}:{mode}".encode()).hexdigest()

    def _check_cache(self, key: str) -> Optional[RAGResult]:
        path = CACHE_DIR / f"response_{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return RAGResult(**data, from_cache=True)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, result: RAGResult):
        path = CACHE_DIR / f"response_{key}.json"
        data = {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "mode": result.mode,
            "model": result.model,
            "verified": result.verified,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _expand_query(self, query: str) -> List[str]:
        return self.expander.expand(query)

    def _retrieve(self, query: str, k: int = 20) -> List[ScoredChunk]:
        return self.retriever.search(query, k=k)

    def _rerank(self, query: str, chunks: List[ScoredChunk], top_k: int = 5) -> List[ScoredChunk]:
        return self.reranker.rerank(query, chunks, top_k=top_k)

    def _evaluate_sufficiency(self, query: str, chunks: List[ScoredChunk]) -> bool:
        if not chunks:
            return False
        avg_score = sum(c.score for c in chunks) / len(chunks)
        return avg_score > 0.15

    def _format_context(self, chunks: List[ScoredChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks):
            source = c.chunk.metadata.get("source", "unknown")
            filename = source.split("\\")[-1].split("/")[-1] if source != "unknown" else "unknown"
            parts.append(f"[Source {i+1}] ({filename}, relevance: {c.score:.3f})")
            parts.append(c.chunk.text[:1000])
            parts.append("")
        return "\n".join(parts)

    def _format_sources(self, chunks: List[ScoredChunk]) -> List[Dict[str, Any]]:
        return [
            {
                "content": c.chunk.text[:300],
                "source": c.chunk.metadata.get("source", "unknown"),
                "filename": c.chunk.metadata.get("filename", "unknown"),
                "score": round(c.score, 4),
            }
            for c in chunks
        ]

    def _generate(self, query: str, context: str, history: str, mode: str) -> LLMResponse:
        template_fn = PromptTemplates.get_template(mode)
        system, prompt = template_fn(query, context, history)
        return self.llm.generate(prompt=prompt, system=system)

    def run(
        self, query: str, session_id: str = "default", mode: str = "general",
        use_cache: bool = True, memory: Optional[ConversationMemory] = None
    ) -> RAGResult:
        cache_key = self._cache_key(query, mode)
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                return cached

        expanded_queries = self._expand_query(query)
        all_chunks: List[ScoredChunk] = []
        seen_sources = set()

        for q in expanded_queries:
            chunks = self._retrieve(q)
            for c in chunks:
                src = c.chunk.metadata.get("source", "") + str(c.chunk.metadata.get("chunk_index", ""))
                if src not in seen_sources:
                    seen_sources.add(src)
                    all_chunks.append(c)

        all_chunks.sort(key=lambda x: x.score, reverse=True)
        top_chunks = all_chunks[:20]
        reranked = self._rerank(query, top_chunks)

        for attempt in range(self.max_retries):
            if self._evaluate_sufficiency(query, reranked):
                break
            if attempt < self.max_retries - 1:
                reranked = self._retrieve(query + f" (rephrase {attempt + 1})", k=20)
                reranked = self._rerank(query, reranked)
        else:
            reranked = reranked[:3] if reranked else []

        history = memory.get_history() if memory else ""
        context = self._format_context(reranked)

        response = self._generate(query, context, history, mode)

        sources = self._format_sources(reranked)
        confidence = min(1.0, sum(c.score for c in reranked) / max(len(reranked), 1)) if reranked else 0.0

        result = RAGResult(
            answer=response.text,
            sources=sources,
            confidence=confidence,
            mode=mode,
            model=response.model or self.llm.name,
            verified=False,
        )

        self._save_cache(cache_key, result)
        return result


from dataclasses import dataclass
