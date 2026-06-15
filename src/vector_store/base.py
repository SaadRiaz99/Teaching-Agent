from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        ...

    @abstractmethod
    def similarity_search(
        self, query_embedding: List[float], k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        ...

    @abstractmethod
    def delete_collection(self):
        ...
