from typing import List

from src.vector_store.base import ScoredChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self._model_name = model_name

    def _lazy_load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, chunks: List[ScoredChunk], top_k: int = 5) -> List[ScoredChunk]:
        if not chunks:
            return []
        if len(chunks) <= 1:
            return chunks
        self._lazy_load()
        pairs = [(query, c.chunk.text) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scored = []
        for i, c in enumerate(chunks):
            scored.append(ScoredChunk(chunk=c.chunk, score=float(scores[i])))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
