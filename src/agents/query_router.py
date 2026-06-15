from src.generation.generator import get_llm


class QueryRouter:
    MODES = ["teaching", "syllabus", "assignment", "general", "greeting"]

    def __init__(self):
        self.llm = get_llm()

    def route(self, query: str) -> str:
        lowered = query.lower()
        greeting_words = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}
        if any(g in lowered for g in greeting_words) and len(query.split()) <= 5:
            return "greeting"

        prompt = (
            f"Classify this student query into exactly one category:\n"
            f"- teaching: asking to explain a concept, topic, or subject\n"
            f"- syllabus: asking about course structure, syllabus, prerequisites, curriculum\n"
            f"- assignment: asking for help with an assignment or homework problem\n"
            f"- general: any other SMIT-related question\n\n"
            f"Query: {query}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )

        try:
            resp = self.llm.generate(prompt=prompt, max_tokens=20, temperature=0.0)
            mode = resp.text.strip().lower()
            if mode in self.MODES:
                return mode
        except Exception:
            pass
        return "general"
