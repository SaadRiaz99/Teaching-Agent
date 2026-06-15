from abc import ABC, abstractmethod
from typing import List
import re

from .loader import Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[str]:
        ...


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: Document) -> List[str]:
        text = document.text.strip()
        if not text:
            return []
        return self._split_recursive(text)

    def _split_recursive(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        for sep in self._separators:
            if sep and sep in text:
                chunks = []
                pieces = text.split(sep)
                current = ""
                for piece in pieces:
                    candidate = (current + sep + piece).strip() if current else piece
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = piece
                if current:
                    chunks.append(current)
                if len(chunks) > 1:
                    return self._merge_overlaps(chunks)
                break
        return self._split_by_chars(text)

    def _merge_overlaps(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        merged = [chunks[0]]
        for c in chunks[1:]:
            if len(merged[-1]) < self.chunk_size:
                overlap = self._find_overlap(merged[-1], c)
                merged[-1] += overlap + c
            else:
                merged.append(c)
        return merged

    def _find_overlap(self, a: str, b: str) -> str:
        max_overlap = min(self.chunk_overlap, len(a), len(b))
        for i in range(max_overlap, 0, -1):
            if a[-i:] == b[:i]:
                return ""
        return " "

    def _split_by_chars(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks


class SemanticChunker(BaseChunker):
    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 800):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> List[str]:
        text = document.text.strip()
        if not text:
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) < self.max_chunk_size:
                current = (current + "\n\n" + p).strip() if current else p
            else:
                if current:
                    chunks.append(current)
                current = p
        if current:
            chunks.append(current)
        final = []
        for c in chunks:
            if len(c) < self.min_chunk_size and final:
                final[-1] += "\n\n" + c
            else:
                final.append(c)
        return final
