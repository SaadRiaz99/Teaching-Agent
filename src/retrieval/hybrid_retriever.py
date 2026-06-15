from typing import List, Optional, Dict, Any

from src.embeddings import get_embedder
from src.vector_store import get_vector_store
from src.vector_store.base import ScoredChunk


class HybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.embedder = get_embedder()
        self.store = get_vector_store()
        self.rrf_k = rrf_k
        self._bm25_corpus: List[str] = []
        self._bm25_index = None

    def _ensure_bm25_index(self):
        if self._bm25_index is not None:
            return
        from rank_bm25 import BM25Okapi
        chunks = self.store.get_all_chunks()
        self._bm25_corpus = [c.text for c in chunks]
        if self._bm25_corpus:
            tokenized = [self._tokenize(t) for t in self._bm25_corpus]
            self._bm25_index = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\w+", text.lower())

    def _dense_search(self, query_embedding: List[float], k: int, filters: Optional[Dict] = None) -> List[ScoredChunk]:
        return self.store.similarity_search(query_embedding, k=k, filters=filters)

    def _sparse_search(self, query: str, k: int) -> List[ScoredChunk]:
        self._ensure_bm25_index()
        if not self._bm25_index:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        chunks = self.store.get_all_chunks()
        results = []
        for idx, score in ranked:
            if score > 0:
                results.append(ScoredChunk(chunk=chunks[idx], score=float(score)))
        return results

    def _rrf_fuse(self, dense_results: List[ScoredChunk], sparse_results: List[ScoredChunk], k: int) -> List[ScoredChunk]:
        rrf_scores: Dict[str, dict] = {}
        for rank, r in enumerate(dense_results):
            cid = r.chunk.chunk_id or r.chunk.text[:50]
            if cid not in rrf_scores:
                rrf_scores[cid] = {"chunk": r.chunk, "score": 0.0}
            rrf_scores[cid]["score"] += 1.0 / (k + rank + 1)
        for rank, r in enumerate(sparse_results):
            cid = r.chunk.chunk_id or r.chunk.text[:50]
            if cid not in rrf_scores:
                rrf_scores[cid] = {"chunk": r.chunk, "score": 0.0}
            rrf_scores[cid]["score"] += 1.0 / (k + rank + 1)
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [ScoredChunk(chunk=item["chunk"], score=item["score"]) for item in sorted_results]

    def search(self, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None) -> List[ScoredChunk]:
        query_embedding = self.embedder.embed_query(query)
        dense_results = self._dense_search(query_embedding, k, filters)
        sparse_results = self._sparse_search(query, k)
        if not sparse_results:
            return dense_results
        fused = self._rrf_fuse(dense_results, sparse_results, self.rrf_k)
        return fused[:k]
