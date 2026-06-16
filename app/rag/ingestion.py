from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)

from app.utils.config import settings
from app.utils.logger import logger


SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}


class DocumentIngestor:
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentIngestor initialized, upload dir: {self.upload_dir}")

    def ingest_file(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        loader_class = SUPPORTED_EXTENSIONS[ext]
        documents = []
        try:
            loader = loader_class(str(path))
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source": path.name,
                    "file_type": ext,
                    "file_path": str(path.absolute()),
                })
            logger.info(
                f"Ingested {path.name}: {len(documents)} page(s)"
            )
        except Exception as e:
            logger.error(f"Failed to ingest {path.name}: {e}")
            raise

        return documents

    def ingest_uploaded_file(
        self, file_content: bytes, filename: str
    ) -> list[Document]:
        sanitized = Path(filename).name
        ext = Path(sanitized).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        temp_path = self.upload_dir / sanitized
        with open(temp_path, "wb") as f:
            f.write(file_content)

        try:
            documents = self.ingest_file(temp_path)
            return documents
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def ingest_text(
        self, text: str, source_name: str = "inline_text"
    ) -> list[Document]:
        doc = Document(
            page_content=text,
            metadata={
                "source": source_name,
                "file_type": ".txt",
            },
        )
        logger.info(f"Ingested inline text as '{source_name}'")
        return [doc]
