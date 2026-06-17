import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "SMIT AI Teaching Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    ZEN_API_KEY: str = os.getenv("ZEN_API_KEY", "")
    ZEN_API_BASE: str = os.getenv("ZEN_API_BASE", "https://opencode.ai/zen/v1")
    ZEN_MODEL: str = os.getenv("ZEN_MODEL", "opencode/big-pickle")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "vectordb")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "smit_documents")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")
    EMBEDDING_TYPE: str = os.getenv("EMBEDDING_TYPE", "openai")  # openai or local
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")

    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    CONVERSATION_HISTORY_LIMIT: int = int(
        os.getenv("CONVERSATION_HISTORY_LIMIT", "10")
    )

    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
