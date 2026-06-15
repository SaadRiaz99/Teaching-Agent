from typing import Optional


class PromptTemplates:
    @staticmethod
    def teaching(query: str, context: str, history: Optional[str] = None) -> tuple:
        system = (
            "You are an expert SMIT (Sikkim Manipal Institute of Technology) teaching assistant. "
            "Your role is to explain concepts clearly, provide examples, and help students understand "
            "their course material. Use the provided context to ground your answers. "
            "If the context doesn't contain enough information, acknowledge this and use your general "
            "knowledge while being clear about what comes from the provided sources vs your own knowledge. "
            "Always cite the source filename when using information from the context. "
            "Format your response in a clear, educational manner with bullet points or sections as appropriate."
        )
        prompt = self._build(query, context, history)
        return system, prompt

    @staticmethod
    def syllabus(query: str, context: str, history: Optional[str] = None) -> tuple:
        system = (
            "You are an SMIT academic advisor assistant. Help students understand course structures, "
            "syllabi, prerequisites, and academic programs at Sikkim Manipal Institute of Technology. "
            "Base your answers on the provided context documents. "
            "Cite specific course codes and document sources when possible."
        )
        prompt = self._build(query, context, history)
        return system, prompt

    @staticmethod
    def assignment(query: str, context: str, history: Optional[str] = None) -> tuple:
        system = (
            "You are an SMIT tutor helping a student with their assignment. "
            "Do NOT give direct answers to assignment questions. Instead, explain concepts, "
            "provide guidance on how to approach the problem, and point to relevant course material. "
            "Use the context to ensure your guidance aligns with the course content. "
            "Encourage critical thinking and understanding over memorization."
        )
        prompt = self._build(query, context, history)
        return system, prompt

    @staticmethod
    def general(query: str, context: str, history: Optional[str] = None) -> tuple:
        system = (
            "You are a helpful SMIT assistant answering questions about Sikkim Manipal Institute of Technology. "
            "Use the provided context documents to give accurate, sourced answers. "
            "If the context is insufficient, say so clearly."
        )
        prompt = self._build(query, context, history)
        return system, prompt

    @staticmethod
    def _build(query: str, context: str, history: Optional[str] = None) -> str:
        parts = []
        if history:
            parts.append(f"Previous conversation:\n{history}\n")
        if context:
            parts.append(f"Relevant context documents:\n{context}\n")
        parts.append(f"Question: {query}")
        parts.append("\nAnswer:")
        return "\n\n".join(parts)

    @staticmethod
    def get_template(mode: str):
        templates = {
            "teaching": PromptTemplates.teaching,
            "syllabus": PromptTemplates.syllabus,
            "assignment": PromptTemplates.assignment,
            "general": PromptTemplates.general,
        }
        return templates.get(mode, PromptTemplates.general)
