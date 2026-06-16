from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

from app.utils.config import settings
from app.utils.logger import logger

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are SMIT AI Teaching Agent, a helpful tutor for Saylani Mass IT Training (SMIT) students. "
        "Your role is to answer questions based SOLELY on the provided learning materials.\n\n"
        "Guidelines:\n"
        "- Answer in a clear, beginner-friendly manner.\n"
        "- If the context does not contain enough information to answer, say: "
        "'I cannot find sufficient information in the uploaded learning materials to answer this question. "
        "Please upload relevant documents or rephrase your query.'\n"
        "- Always cite the source document names when referencing material.\n"
        "- Use bullet points and short paragraphs for readability.\n"
        "- Do NOT make up information or use external knowledge.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a quiz generator for Saylani Mass IT Training (SMIT). "
        "Based on the provided learning materials, generate {num_questions} multiple-choice questions "
        "at {difficulty} difficulty level.\n\n"
        "{topic_context}"
        "Format each question as a JSON object with keys: "
        "'question', 'options' (list of 4 strings), 'correct_answer', 'explanation', 'difficulty'.\n"
        "Return a JSON array of questions.\n\n"
        "Context:\n{context}",
    ),
    ("human", "Generate {num_questions} quiz questions about the provided material."),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a teaching assistant that summarizes learning materials for SMIT students. "
        "Create a concise summary of the provided content (max {max_length} words). "
        "Focus on key concepts, learning objectives, and main takeaways. "
        "Use simple language suitable for beginners.\n\n"
        "Context:\n{context}",
    ),
    ("human", "Please summarize this learning material."),
])

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a learning advisor for SMIT students. Based on the student's question and the available "
        "learning materials, recommend what topics or documents the student should study next. "
        "Be specific and reference actual document names from the context.\n\n"
        "Available Materials Context:\n{context}",
    ),
    ("human", "Student question: {question}\n\nWhat learning recommendations do you have?"),
])


class ResponseGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
        )
        logger.info(
            f"ResponseGenerator initialized: model={settings.LLM_MODEL}"
        )

    def _format_context(
        self, chunks: list[tuple[Document, float]]
    ) -> str:
        parts = []
        for i, (doc, score) in enumerate(chunks):
            source = doc.metadata.get("source", "Unknown")
            parts.append(
                f"[Source {i+1}: {source} (relevance: {score:.2f})]\n{doc.page_content}\n"
            )
        return "\n---\n".join(parts)

    def generate_answer(
        self,
        question: str,
        chunks: list[tuple[Document, float]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, bool]:
        if not chunks:
            logger.warning("No context chunks provided for generation")
            return (
                "I cannot find sufficient information in the uploaded learning materials "
                "to answer this question. Please upload relevant documents or rephrase your query.",
                False,
            )

        context = self._format_context(chunks)
        prompt = RAG_PROMPT.format(context=context, question=question)

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        llm = self.llm
        if kwargs:
            llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_API_BASE,
                **kwargs,
            )

        response = llm.invoke(prompt)
        answer = response.content.strip()

        not_found_phrases = [
            "cannot find sufficient information",
            "do not have enough information",
            "not available in the provided",
            "not found in the uploaded",
        ]
        grounded = not any(phrase in answer.lower() for phrase in not_found_phrases)

        logger.info(
            f"Generated answer (grounded={grounded}, {len(answer)} chars)"
        )
        return answer, grounded

    def generate_quiz(
        self,
        chunks: list[tuple[Document, float]],
        num_questions: int = 5,
        difficulty: str = "medium",
        topic: str | None = None,
    ) -> str:
        context = self._format_context(chunks)
        topic_context = ""
        if topic:
            topic_context = f"Focus on the topic: {topic}\n"

        prompt = QUIZ_PROMPT.format(
            context=context,
            num_questions=num_questions,
            difficulty=difficulty,
            topic_context=topic_context,
        )

        response = self.llm.invoke(prompt)
        logger.info(f"Generated {num_questions} quiz questions at {difficulty} difficulty")
        return response.content.strip()

    def generate_summary(
        self,
        chunks: list[tuple[Document, float]],
        max_length: int = 500,
    ) -> str:
        context = self._format_context(chunks)
        prompt = SUMMARY_PROMPT.format(
            context=context, max_length=max_length
        )
        response = self.llm.invoke(prompt)
        logger.info(f"Generated summary ({max_length} words max)")
        return response.content.strip()

    def generate_recommendations(
        self,
        question: str,
        chunks: list[tuple[Document, float]],
    ) -> str:
        context = self._format_context(chunks)
        prompt = RECOMMENDATION_PROMPT.format(
            context=context, question=question
        )
        response = self.llm.invoke(prompt)
        logger.info("Generated learning recommendations")
        return response.content.strip()
