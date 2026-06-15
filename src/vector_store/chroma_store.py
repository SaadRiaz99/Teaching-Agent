from typing import List, Optional, Dict, Any
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from .base import BaseVectorStore, Chunk, ScoredChunk


class ChromaStore(BaseVectorStore):
    def __init__(self, collection_name: str = "smit_docs"):
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        ids = [c.chunk_id or str(uuid.uuid4()) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def similarity_search(
        self, query_embedding: List[float], k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        where = None
        if filters:
            where = {}
            for key, val in filters.items():
                if isinstance(val, str):
                    where[key] = val
                elif isinstance(val, list):
                    where[key] = {"$in": val}
                else:
                    where[key] = val

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count() or k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        scored = []
        if results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk = Chunk(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] or {},
                )
                score = 1.0 - (results["distances"][0][i] if results["distances"][0][i] is not None else 0.0)
                scored.append(ScoredChunk(chunk=chunk, score=score))
        return scored

    def count(self) -> int:
        return self._collection.count()

    def get_all_chunks(self) -> List[Chunk]:
        results = self._collection.get(include=["documents", "metadatas"])
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append(Chunk(
                chunk_id=results["ids"][i],
                text=results["documents"][i],
                metadata=results["metadatas"][i] or {},
            ))
        return chunks

    def delete_collection(self):
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
