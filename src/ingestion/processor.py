from typing import List, Optional
from pathlib import Path

from src.embeddings import get_embedder
from src.vector_store import get_vector_store
from src.vector_store.base import Chunk
from .loader import DocumentLoader, Document
from .chunker import RecursiveChunker


class IngestionPipeline:
    def __init__(self, chunker=None):
        self.loader = DocumentLoader()
        self.chunker = chunker or RecursiveChunker(chunk_size=512, chunk_overlap=64)
        self.embedder = get_embedder()
        self.store = get_vector_store()

    def ingest(self, path: str, metadata: Optional[dict] = None, show_progress: bool = False) -> dict:
        docs = self.loader.load(path)
        if not docs:
            return {"files": 0, "chunks": 0, "documents": 0}

        all_chunks: List[Chunk] = []
        for doc in docs:
            chunk_texts = self.chunker.chunk(doc)
            for i, text in enumerate(chunk_texts):
                chunk_meta = dict(doc.metadata)
                if metadata:
                    chunk_meta.update(metadata)
                chunk_meta["chunk_index"] = i
                chunk_meta["total_chunks"] = len(chunk_texts)
                all_chunks.append(Chunk(text=text, metadata=chunk_meta))

        if not all_chunks:
            return {"files": len(docs), "chunks": 0, "documents": 0}

        texts = [c.text for c in all_chunks]

        if show_progress:
            print(f"  Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        if show_progress:
            print(f"  Storing {len(all_chunks)} chunks in vector database...")
        self.store.add_chunks(all_chunks, embeddings)

        return {
            "files": len(docs),
            "chunks": len(all_chunks),
            "documents": len(set(c.metadata.get("source", "") for c in all_chunks)),
        }
