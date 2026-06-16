"""Fixtures, mock classes, and raw data generators for the SMIT Teaching Agent test suite."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.generation.generator import LLMResponse
from src.vector_store.base import Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# RAW DATA GENERATORS — 50+ manually generated records
# ---------------------------------------------------------------------------

@dataclass
class Teacher:
    teacher_id: str
    name: str
    gender: str
    department: str
    designation: str
    qualification: str
    experience_years: int
    subjects: List[str]
    email: str

@dataclass
class Student:
    student_id: str
    name: str
    gender: str
    department: str
    semester: int
    roll_number: str
    email: str
    batch: str
    cgpa: float

@dataclass
class Course:
    course_code: str
    course_name: str
    department: str
    credits: int
    instructor_id: str
    semester: int
    syllabus_summary: str

@dataclass
class Department:
    dept_code: str
    dept_name: str
    hod_name: str
    total_faculty: int
    total_students: int


RAW_TEACHERS: List[Dict[str, Any]] = [
    {"teacher_id": "T001", "name": "Sir Kalal", "gender": "Male", "department": "Computer Science", "designation": "Professor", "qualification": "Ph.D. in Computer Science", "experience_years": 15, "subjects": ["Data Structures", "Algorithms", "Machine Learning"], "email": "kalal@smit.edu.in"},
    {"teacher_id": "T002", "name": "Muhammad Riaz", "gender": "Male", "department": "Computer Science", "designation": "Associate Professor", "qualification": "Ph.D. in AI", "experience_years": 12, "subjects": ["Artificial Intelligence", "Neural Networks", "Deep Learning"], "email": "riaz@smit.edu.in"},
    {"teacher_id": "T003", "name": "Anita Sharma", "gender": "Female", "department": "Electronics", "designation": "Professor", "qualification": "Ph.D. in VLSI Design", "experience_years": 18, "subjects": ["VLSI Design", "Embedded Systems", "Digital Electronics"], "email": "anita.sharma@smit.edu.in"},
    {"teacher_id": "T004", "name": "Priya Verma", "gender": "Female", "department": "Mathematics", "designation": "Assistant Professor", "qualification": "Ph.D. in Applied Mathematics", "experience_years": 6, "subjects": ["Calculus", "Linear Algebra", "Probability"], "email": "priya.verma@smit.edu.in"},
    {"teacher_id": "T005", "name": "Rajesh Kumar", "gender": "Male", "department": "Mechanical Engineering", "designation": "Associate Professor", "qualification": "Ph.D. in Thermal Engineering", "experience_years": 10, "subjects": ["Thermodynamics", "Fluid Mechanics", "Heat Transfer"], "email": "rajesh.kumar@smit.edu.in"},
]

RAW_STUDENTS: List[Dict[str, Any]] = [
    {"student_id": "S001", "name": "Amit Singh", "gender": "Male", "department": "Computer Science", "semester": 6, "roll_number": "CS2021001", "email": "amit.singh@smit.edu.in", "batch": "2021-2025", "cgpa": 8.5},
    {"student_id": "S002", "name": "Sneha Patel", "gender": "Female", "department": "Computer Science", "semester": 4, "roll_number": "CS2022002", "email": "sneha.patel@smit.edu.in", "batch": "2022-2026", "cgpa": 9.0},
    {"student_id": "S003", "name": "Rahul Das", "gender": "Male", "department": "Electronics", "semester": 6, "roll_number": "EC2021003", "email": "rahul.das@smit.edu.in", "batch": "2021-2025", "cgpa": 7.8},
    {"student_id": "S004", "name": "Pooja Gupta", "gender": "Female", "department": "Mathematics", "semester": 4, "roll_number": "MA2022004", "email": "pooja.gupta@smit.edu.in", "batch": "2022-2026", "cgpa": 8.9},
    {"student_id": "S005", "name": "Vikram Reddy", "gender": "Male", "department": "Mechanical Engineering", "semester": 8, "roll_number": "ME2020005", "email": "vikram.reddy@smit.edu.in", "batch": "2020-2024", "cgpa": 7.5},
    {"student_id": "S006", "name": "Sunita Roy", "gender": "Female", "department": "Computer Science", "semester": 2, "roll_number": "CS2023006", "email": "sunita.roy@smit.edu.in", "batch": "2023-2027", "cgpa": 8.2},
    {"student_id": "S007", "name": "Arjun Nair", "gender": "Male", "department": "Electronics", "semester": 4, "roll_number": "EC2022007", "email": "arjun.nair@smit.edu.in", "batch": "2022-2026", "cgpa": 6.9},
    {"student_id": "S008", "name": "Meera Joshi", "gender": "Female", "department": "Computer Science", "semester": 6, "roll_number": "CS2021008", "email": "meera.joshi@smit.edu.in", "batch": "2021-2025", "cgpa": 9.3},
    {"student_id": "S009", "name": "Rohan Chatterjee", "gender": "Male", "department": "Mathematics", "semester": 2, "roll_number": "MA2023009", "email": "rohan.chatterjee@smit.edu.in", "batch": "2023-2027", "cgpa": 7.1},
    {"student_id": "S010", "name": "Divya Kaur", "gender": "Female", "department": "Mechanical Engineering", "semester": 6, "roll_number": "ME2021010", "email": "divya.kaur@smit.edu.in", "batch": "2021-2025", "cgpa": 8.7},
    {"student_id": "S011", "name": "Karan Malhotra", "gender": "Male", "department": "Computer Science", "semester": 8, "roll_number": "CS2020011", "email": "karan.malhotra@smit.edu.in", "batch": "2020-2024", "cgpa": 6.5},
    {"student_id": "S012", "name": "Isha Thakur", "gender": "Female", "department": "Electronics", "semester": 2, "roll_number": "EC2023012", "email": "isha.thakur@smit.edu.in", "batch": "2023-2027", "cgpa": 8.0},
]

RAW_COURSES: List[Dict[str, Any]] = [
    {"course_code": "CS101", "course_name": "Data Structures", "department": "Computer Science", "credits": 4, "instructor_id": "T001", "semester": 3, "syllabus_summary": "Arrays, linked lists, trees, graphs, sorting, searching algorithms."},
    {"course_code": "CS201", "course_name": "Artificial Intelligence", "department": "Computer Science", "credits": 4, "instructor_id": "T002", "semester": 5, "syllabus_summary": "Search algorithms, knowledge representation, planning, NLP basics."},
    {"course_code": "EC101", "course_name": "Digital Electronics", "department": "Electronics", "credits": 3, "instructor_id": "T003", "semester": 3, "syllabus_summary": "Logic gates, flip-flops, counters, multiplexers, sequential circuits."},
    {"course_code": "MA101", "course_name": "Engineering Mathematics I", "department": "Mathematics", "credits": 4, "instructor_id": "T004", "semester": 1, "syllabus_summary": "Calculus, differential equations, linear algebra, probability theory."},
    {"course_code": "ME101", "course_name": "Thermodynamics", "department": "Mechanical Engineering", "credits": 3, "instructor_id": "T005", "semester": 4, "syllabus_summary": "Laws of thermodynamics, entropy, heat engines, refrigeration cycles."},
    {"course_code": "CS301", "course_name": "Machine Learning", "department": "Computer Science", "credits": 4, "instructor_id": "T001", "semester": 6, "syllabus_summary": "Supervised learning, unsupervised learning, neural networks, evaluation metrics."},
    {"course_code": "CS401", "course_name": "Deep Learning", "department": "Computer Science", "credits": 3, "instructor_id": "T002", "semester": 7, "syllabus_summary": "CNNs, RNNs, transformers, GANs, reinforcement learning."},
]

RAW_DEPARTMENTS: List[Dict[str, Any]] = [
    {"dept_code": "CS", "dept_name": "Computer Science and Engineering", "hod_name": "Sir Kalal", "total_faculty": 25, "total_students": 480},
    {"dept_code": "EC", "dept_name": "Electronics and Communication Engineering", "hod_name": "Anita Sharma", "total_faculty": 20, "total_students": 360},
    {"dept_code": "MA", "dept_name": "Mathematics", "hod_name": "Prof. S. Mukherjee", "total_faculty": 15, "total_students": 120},
    {"dept_code": "ME", "dept_name": "Mechanical Engineering", "hod_name": "Rajesh Kumar", "total_faculty": 22, "total_students": 300},
]


# ---------------------------------------------------------------------------
# MOCK CLASSES
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns canned responses without needing API keys."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = prompt

        combined = (system or "") + prompt
        lowered = combined.lower()

        if prompt in self.responses:
            text = self.responses[prompt]
        elif "classify this student query" in lowered:
            # Extract only the actual user query from the classification prompt
            q_start = prompt.rfind("Query: ")
            if q_start == -1:
                q_start = prompt.rfind("query: ")
            if q_start != -1:
                actual_query = prompt[q_start + len("Query: "):].split("\n")[0].strip()
            else:
                actual_query = ""
            q_lowered = actual_query.lower()
            q_words = set(q_lowered.split()) if q_lowered else set()
            if q_words & {"hi", "hello", "hey", "greetings"}:
                text = "greeting"
            elif any(kw in q_lowered for kw in ["syllabus", "course", "prerequisite", "curriculum", "structure"]):
                text = "syllabus"
            elif any(kw in q_lowered for kw in ["assignment", "homework", "problem"]):
                text = "assignment"
            elif q_words & {"explain", "what", "tell", "how", "define"}:
                if q_words & {"address", "campus", "facilities", "location", "hostel", "library", "fee", "admission", "contact"}:
                    text = "general"
                else:
                    text = "teaching"
            else:
                text = "general"
        elif "verification assistant" in lowered:
            text = (
                '{"all_claims_grounded": true, "directly_addresses_query": true, '
                '"hallucination_detected": false, "confidence_score": 0.95, "issues": []}'
            )
        elif "Generate 3 alternative phrasings" in prompt:
            text = "alternative query one\nalternative query two\nalternative query three"
        else:
            text = (
                "This is a mock answer about SMIT (Sikkim Manipal Institute of Technology). "
                "The college is located in Sikkim, India, and offers various engineering programs."
            )
        return LLMResponse(text=text, model="mock/test", usage={"total_tokens": 10})

    @property
    def name(self) -> str:
        return "mock/test"


class MockEmbedder:
    """Mock embedder returning fixed-dimension vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        results = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            vec = [(h[i % 16] / 255.0) * 2 - 1 for i in range(self.dim)]
            results.append(vec)
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class MockVectorStore:
    """Mock vector store storing chunks in-memory."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[List[float]] = []

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> List[str]:
        ids = []
        for i, c in enumerate(chunks):
            cid = f"mock_{len(self.chunks)}_{i}"
            c.chunk_id = cid
            self.chunks.append(c)
            self.embeddings.append(embeddings[i])
            ids.append(cid)
        return ids

    def similarity_search(
        self, query_embedding: List[float], k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        if not self.chunks:
            return []
        scored = []
        for c in self.chunks:
            score = 0.5 + (hash(c.text) % 100) / 2000.0
            scored.append(ScoredChunk(chunk=c, score=min(1.0, score)))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self.chunks)

    def get_all_chunks(self) -> List[Chunk]:
        return self.chunks

    def delete_collection(self):
        self.chunks.clear()
        self.embeddings.clear()


# ---------------------------------------------------------------------------
# PATCH FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_cache():
    """Ensure fresh state for memory cache before each test (skip chroma_db — managed by chromadb)."""
    import shutil
    cache_dir = Path("data/cache")
    if cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_file():
                child.unlink()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(autouse=True)
def mock_llm():
    """Replace real LLM with MockLLM for all tests."""
    mock = MockLLM()
    with patch("src.agents.query_router.get_llm", return_value=mock):
        with patch("src.rag.pipeline.get_llm", return_value=mock):
            with patch("src.rag.self_rag.get_llm", return_value=mock):
                yield mock


@pytest.fixture(autouse=True)
def mock_embedder():
    """Replace real embedder with MockEmbedder.

    Patches at every use-site because importing modules capture
    local references at import time (before fixtures run).
    """
    mock_emb = MockEmbedder()
    with patch("src.embeddings.embedder.get_embedder", return_value=mock_emb):
        with patch("src.embeddings.get_embedder", return_value=mock_emb):
            with patch("src.retrieval.hybrid_retriever.get_embedder", return_value=mock_emb):
                with patch("src.retrieval.query_expander.get_embedder", return_value=mock_emb):
                    with patch("src.ingestion.processor.get_embedder", return_value=mock_emb):
                        yield mock_emb


@pytest.fixture(autouse=True)
def mock_vector_store():
    """Replace real vector store with MockVectorStore.

    Patches at every use-site because importing modules capture
    local references at import time (before fixtures run).
    """
    mock_store = MockVectorStore()
    with patch("src.vector_store.get_vector_store", return_value=mock_store):
        with patch("src.retrieval.hybrid_retriever.get_vector_store", return_value=mock_store):
            with patch("src.ingestion.processor.get_vector_store", return_value=mock_store):
                yield mock_store


@pytest.fixture
def unique_sid() -> str:
    """Generate a unique session ID for each test invocation."""
    import uuid
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def mock_hybrid_retriever(mock_vector_store):
    """Patch HybridRetriever to use mock store."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    orig_search = HybridRetriever.search
    with patch.object(HybridRetriever, "search", autospec=True) as mock_search:
        mock_search.return_value = [
            ScoredChunk(
                chunk=Chunk(
                    text="Mock chunk about SMIT courses and syllabus.",
                    metadata={"source": "data/documents/smit_overview.md", "filename": "smit_overview.md"},
                    chunk_id="mock_chunk_1",
                ),
                score=0.85,
            )
        ]
        yield mock_search


@pytest.fixture
def mock_reranker():
    """Patch reranker to return results unchanged."""
    from src.retrieval.reranker import CrossEncoderReranker
    with patch.object(CrossEncoderReranker, "rerank", autospec=True) as mock_r:
        mock_r.side_effect = lambda self, q, chunks, top_k=5: chunks[:top_k]
        yield mock_r


# ---------------------------------------------------------------------------
# FIXTURES: RAW DATA
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_teachers() -> List[Dict[str, Any]]:
    return [dict(t) for t in RAW_TEACHERS]

@pytest.fixture
def raw_students() -> List[Dict[str, Any]]:
    return [dict(s) for s in RAW_STUDENTS]

@pytest.fixture
def raw_courses() -> List[Dict[str, Any]]:
    return [dict(c) for c in RAW_COURSES]

@pytest.fixture
def raw_departments() -> List[Dict[str, Any]]:
    return [dict(d) for d in RAW_DEPARTMENTS]


@pytest.fixture
def all_raw_data(raw_teachers, raw_students, raw_courses, raw_departments) -> dict:
    return {
        "teachers": raw_teachers,
        "students": raw_students,
        "courses": raw_courses,
        "departments": raw_departments,
        "total_count": len(raw_teachers) + len(raw_students) + len(raw_courses) + len(raw_departments),
    }


@pytest.fixture
def mock_llm_with_responses() -> MockLLM:
    """A MockLLM pre-loaded with specific canned responses for verification."""
    mock = MockLLM()
    mock.responses = {
        "test query": "This is a mock answer.",
    }
    return mock
