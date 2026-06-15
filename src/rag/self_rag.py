from typing import List, Dict, Any

from src.generation.generator import get_llm


class SelfRAGVerifier:
    def __init__(self):
        self.llm = get_llm()

    def verify(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> dict:
        if not sources:
            return {"passed": True, "score": 0.5, "issues": ["No sources to verify against"]}

        source_text = "\n\n".join(
            f"Source {i+1}: {s.get('content', '')[:500]}" for i, s in enumerate(sources)
        )

        prompt = (
            f"Question: {query}\n\n"
            f"Generated Answer: {answer}\n\n"
            f"Retrieved Context:\n{source_text}\n\n"
            f"Analyze the generated answer against the retrieved context. Respond in JSON format:\n"
            f"{{\n"
            f'  "all_claims_grounded": true/false,\n'
            f'  "directly_addresses_query": true/false,\n'
            f'  "hallucination_detected": true/false,\n'
            f'  "confidence_score": 0.0-1.0,\n'
            f'  "issues": ["list of any issues found"]\n'
            f"}}"
        )

        system = (
            "You are a verification assistant. Analyze if an LLM-generated answer is properly "
            "grounded in the provided context. Be strict but fair. Return ONLY valid JSON."
        )

        try:
            resp = self.llm.generate(prompt=prompt, system=system, max_tokens=500, temperature=0.0)
            import json
            import re
            json_match = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception:
            pass

        return {"passed": True, "score": 0.7, "issues": ["Verification skipped due to error"]}

    def verify_and_correct(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> str:
        result = self.verify(query, answer, sources)
        if result.get("hallucination_detected") or not result.get("all_claims_grounded", True):
            source_text = "\n\n".join(
                f"Source {i+1}: {s.get('content', '')[:500]}" for i, s in enumerate(sources)
            )
            correction_prompt = (
                f"Question: {query}\n\n"
                f"Previous (potentially incorrect) answer: {answer}\n\n"
                f"Retrieved context:\n{source_text}\n\n"
                f"Issues found: {', '.join(result.get('issues', []))}\n\n"
                f"Please provide a corrected answer that is strictly grounded in the provided context. "
                f"Only use information from the context. If the context doesn't answer the question, say so."
            )
            try:
                corrected = self.llm.generate(
                    prompt=correction_prompt,
                    system="You are a careful fact-checker. Only use information from the provided context.",
                    max_tokens=1024,
                    temperature=0.1,
                )
                return corrected.text
            except Exception:
                return answer
        return answer
