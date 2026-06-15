from .base import BaseVectorStore, ScoredChunk
from .chroma_store import ChromaStore

def get_vector_store() -> BaseVectorStore:
    return ChromaStore()

__all__ = ["BaseVectorStore", "ScoredChunk", "ChromaStore", "get_vector_store"]
