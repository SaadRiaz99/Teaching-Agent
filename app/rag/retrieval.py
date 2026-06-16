from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from app.utils.config import settings
from app.utils.logger import logger


class RetrievalManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
        )
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.COLLECTION_NAME
        self.top_k = settings.RETRIEVAL_TOP_K
        logger.info(
            f"RetrievalManager initialized: top_k={self.top_k}, "
            f"collection={self.collection_name}"
        )

    @property
    def vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        k = top_k or self.top_k
        db = self.vector_store
        results = db.similarity_search_with_relevance_scores(query, k=k)
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:60]}...")
        for doc, score in results:
            logger.debug(f"  score={score:.4f} | source={doc.metadata.get('source', 'unknown')}")
        return results

    def retrieve_with_filter(
        self,
        query: str,
        filter_dict: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        k = top_k or self.top_k
        db = self.vector_store
        results = db.similarity_search(query, k=k, filter=filter_dict)
        logger.info(
            f"Retrieved {len(results)} filtered chunks for query: {query[:60]}..."
        )
        return results

    def get_all_document_ids(self) -> list[str]:
        db = self.vector_store
        try:
            results = db._collection.get()
            return list(set(m.get("source", "") for m in (results.get("metadatas") or [])))
        except Exception as e:
            logger.error(f"Failed to get document IDs: {e}")
            return []
