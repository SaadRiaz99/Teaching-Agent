from pydantic_settings import BaseSettings
from typing import Literal
from pathlib import Path
import os


class Settings(BaseSettings):
    llm_provider: Literal["claude", "openai", "ollama"] = "claude"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_embeddings: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "data/chroma_db"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)


settings = Settings()

os.makedirs(settings.chroma_persist_path, exist_ok=True)
