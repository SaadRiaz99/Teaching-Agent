from __future__ import annotations

import json
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


class ResponseGenerator:
    def __init__(self):
        self.use_mock = not settings.LLM_API_KEY or settings.DEBUG
        if not self.use_mock:
            try:
                self.llm = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    openai_api_key=settings.LLM_API_KEY,
                    openai_api_base=settings.LLM_API_BASE,
                )
                logger.info(f"ResponseGenerator initialized with OpenAI: model={settings.LLM_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI LLM: {e}. Falling back to mock mode.")
                self.use_mock = True
        
        if self.use_mock:
            logger.info("ResponseGenerator initialized in MOCK mode (Free Testing)")

    def _format_context(self, chunks: list[tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, score) in enumerate(chunks):
            source = doc.metadata.get("source", "Unknown")
            parts.append(f"[Source {i+1}: {source} (relevance: {score:.2f})]\n{doc.page_content}\n")
        return "\n---\n".join(parts)

    def generate_answer(
        self,
        question: str,
        chunks: list[tuple[Document, float]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, bool]:
        if not chunks:
            return (
                "I cannot find sufficient information in the uploaded learning materials "
                "to answer this question. Please upload relevant documents or rephrase your query.",
                False,
            )

        if self.use_mock:
            # Simple mock response logic: use the first chunk's content
            doc, score = chunks[0]
            source = doc.metadata.get("source", "Unknown")
            answer = (
                f"**[MOCK RESPONSE - FREE TEST MODE]**\n\n"
                f"Based on the material in **{source}**, here is what I found:\n\n"
                f"{doc.page_content[:500]}...\n\n"
                f"*Note: This is a simulated response because OpenAI access is currently unavailable.*"
            )
            return answer, True

        context = self._format_context(chunks)
        prompt = RAG_PROMPT.format(context=context, question=question)

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            grounded = "cannot find sufficient information" not in answer.lower()
            return answer, grounded
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"Error: OpenAI API returned a 403 or other error. Please check your API key and billing. Details: {str(e)}", False

    def generate_quiz(
        self,
        chunks: list[tuple[Document, float]],
        num_questions: int = 5,
        difficulty: str = "medium",
        topic: str | None = None,
    ) -> str:
        if self.use_mock:
            questions = []
            for i in range(num_questions):
                questions.append({
                    "question": f"Sample Mock Question {i+1} about the material?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "Option A",
                    "explanation": "This is a mock explanation for testing the UI.",
                    "difficulty": difficulty
                })
            return json.dumps(questions)
        
        # Real logic... (omitted for brevity in this mock implementation)
        return "[]"

    def generate_summary(self, chunks: list[tuple[Document, float]], max_length: int = 500) -> str:
        if self.use_mock:
            return "**[MOCK SUMMARY]** This is a simulated summary of the uploaded documents for testing the dashboard UI."
        return "Summary generation requires a working LLM API."

    def generate_recommendations(self, question: str, chunks: list[tuple[Document, float]]) -> str:
        if self.use_mock:
            return "**[MOCK RECOMMENDATIONS]** 1. Review the first document. 2. Practice more questions."
        return "Recommendation generation requires a working LLM API."
