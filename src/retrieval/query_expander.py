from typing import List, Optional

from src.embeddings import get_embedder
from src.vector_store import get_vector_store
from src.vector_store.base import ScoredChunk


class QueryExpander:
    def __init__(self, llm=None):
        self.llm = llm
        self.embedder = get_embedder()
        self.store = get_vector_store()

    def set_llm(self, llm):
        self.llm = llm

    def expand(self, query: str) -> List[str]:
        queries = [query]
        if self.llm:
            multi = self._multi_query(query)
            queries.extend(multi)
            hyde = self._hyde_query(query)
            if hyde and hyde != query:
                queries.append(hyde)
        return queries

    def _multi_query(self, query: str) -> List[str]:
        try:
            resp = self.llm.generate(
                prompt=f"Generate 3 alternative phrasings of this question to help search a knowledge base. "
                       f"Return only the queries, one per line, no numbering.\n\nQuestion: {query}",
                system="You are a query reformulation assistant. Output only the reformulated queries, one per line.",
                max_tokens=200,
            )
            lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
            return lines[:3]
        except Exception:
            return []

    def _hyde_query(self, query: str) -> Optional[str]:
        try:
            resp = self.llm.generate(
                prompt=f"Answer this question concisely based on general knowledge:\n\n{query}",
                system="Provide a concise hypothetical answer.",
                max_tokens=300,
            )
            return resp.text.strip()
        except Exception:
            return None

    def search_expanded(self, query: str, retriever, k: int = 20, top_k: int = 5) -> List[ScoredChunk]:
        queries = self.expand(query)
        all_results: List[ScoredChunk] = []
        seen = set()
        for q in queries:
            results = retriever.search(q, k=k)
            for r in results:
                cid = r.chunk.chunk_id or r.chunk.text[:80]
                if cid not in seen:
                    seen.add(cid)
                    all_results.append(r)
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
