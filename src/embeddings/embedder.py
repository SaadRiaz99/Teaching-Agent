from abc import ABC, abstractmethod
from typing import List, Optional
import hashlib
import pickle
from pathlib import Path
from src.config import settings


class BaseEmbeddings(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...


class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, show_progress_bar=False).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class CachedEmbeddings(BaseEmbeddings):
    def __init__(self, inner: BaseEmbeddings, cache_path: str = "data/cache/embeddings.pkl"):
        self._inner = inner
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self):
        if self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def _save_cache(self):
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        uncached = []
        uncached_idx = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self._cache:
                results.append((i, self._cache[k]))
            else:
                uncached.append(t)
                uncached_idx.append(i)
        if uncached:
            embeddings = self._inner.embed(uncached)
            for idx, emb in zip(uncached_idx, embeddings):
                k = self._key(uncached[idx - uncached_idx[0] if uncached_idx else 0])
                self._cache[self._key(uncached[uncached_idx.index(idx)])] = emb
                results.append((idx, emb))
            self._save_cache()
        return [e for _, e in sorted(results, key=lambda x: x[0])]

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)


_embedder: Optional[BaseEmbeddings] = None


def get_embedder() -> BaseEmbeddings:
    global _embedder
    if _embedder is not None:
        return _embedder
    if settings.openai_embeddings:
        inner: BaseEmbeddings = OpenAIEmbeddings()
    else:
        inner = SentenceTransformerEmbeddings(settings.embedding_model)
    _embedder = CachedEmbeddings(inner)
    return _embedder
