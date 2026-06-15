from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.config import settings


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    usage: Optional[dict] = None


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ClaudeLLM(BaseLLM):
    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        kwargs = dict(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return LLMResponse(
            text=resp.content[0].text,
            model=resp.model,
            usage={"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        )

    @property
    def name(self) -> str:
        return f"claude/{settings.llm_provider}"


class OpenAILLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            text=resp.choices[0].message.content,
            model=resp.model,
            usage={"total_tokens": resp.usage.total_tokens} if resp.usage else None,
        )

    @property
    def name(self) -> str:
        return "openai/gpt-4o"


class OllamaLLM(BaseLLM):
    def __init__(self):
        import httpx
        self._client = httpx.Client(base_url=settings.ollama_base_url)
        self._model = settings.ollama_model

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.3) -> LLMResponse:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if system:
            payload["system"] = system
        resp = self._client.post("/api/generate", json=payload)
        data = resp.json()
        return LLMResponse(
            text=data.get("response", ""),
            model=f"ollama/{self._model}",
            usage={"eval_count": data.get("eval_count", 0)},
        )

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"


_llm: Optional[BaseLLM] = None


def get_llm() -> BaseLLM:
    global _llm
    if _llm is not None:
        return _llm
    provider = settings.llm_provider
    if provider == "claude":
        _llm = ClaudeLLM()
    elif provider == "openai":
        _llm = OpenAILLM()
    elif provider == "ollama":
        _llm = OllamaLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return _llm
