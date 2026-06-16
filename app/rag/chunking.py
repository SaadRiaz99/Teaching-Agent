from __future__ import annotations

from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.utils.config import settings
from app.utils.logger import logger


class TextChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"TextChunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def chunk_documents(
        self, documents: list[Document]
    ) -> list[Document]:
        if not documents:
            logger.warning("No documents provided for chunking")
            return []

        chunked = self._splitter.split_documents(documents)
        for i, chunk in enumerate(chunked):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            f"Chunked {len(documents)} documents into {len(chunked)} chunks"
        )
        return chunked

    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        doc = Document(page_content=text, metadata=metadata or {})
        return self.chunk_documents([doc])
