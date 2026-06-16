from __future__ import annotations

from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from app.utils.config import settings
from app.utils.logger import logger


class EmbeddingManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
        )
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.COLLECTION_NAME
        logger.info(
            f"EmbeddingManager initialized: model={settings.EMBEDDING_MODEL}, "
            f"db={settings.VECTOR_DB_TYPE}, persist={self.persist_directory}"
        )

    @property
    def vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            logger.warning("No documents to add to vector store")
            return []

        db = self.vector_store
        ids = []
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_ids = db.add_documents(batch)
            ids.extend(batch_ids)
            logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} docs")

        logger.info(f"Added {len(documents)} documents to vector store")
        return ids

    def delete_documents(self, document_ids: list[str]) -> None:
        if not document_ids:
            return
        db = self.vector_store
        db.delete(document_ids)
        logger.info(f"Deleted {len(document_ids)} documents from vector store")

    def get_collection_stats(self) -> dict[str, Any]:
        db = self.vector_store
        try:
            count = db._collection.count()
        except Exception:
            count = 0
        return {
            "collection": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_directory,
        }
