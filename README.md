# 🎓 SMIT AI Teaching Agent

An intelligent Retrieval-Augmented Generation (RAG) teaching assistant for **Saylani Mass IT Training (SMIT)**. This agent answers student questions based on uploaded course materials using a complete RAG pipeline with LangChain, ChromaDB, FastAPI, and Streamlit.

## Features

- **📄 Document Ingestion** — Upload PDF, DOCX, TXT, and Markdown files
- **🔍 Semantic Search** — Retrieves the most relevant chunks from your knowledge base
- **💬 Chat Interface** — Ask course-related questions and get grounded answers
- **📝 Quiz Generation** — Auto-generate MCQs from your learning materials
- **📋 Summarization** — Get concise summaries of uploaded lessons
- **🎯 Learning Recommendations** — Personalized study path suggestions
- **📚 Source Citations** — Every answer cites the retrieved document chunks
- **💾 Conversational Context** — Maintains chat history within sessions

## Architecture

```
                    ┌─────────────┐
                    │  Streamlit   │
                    │   Frontend   │
                    └──────┬──────┘
                           │ HTTP
                    ┌──────▼──────┐
                    │   FastAPI    │
                    │   Backend    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼────┐ ┌────▼────┐
        │ Ingestion │ │Chunker │ │Retrieval│
        └─────┬─────┘ └───┬────┘ └────┬────┘
              │            │           │
        ┌─────▼────────────▼───────────▼─────┐
        │         ChromaDB Vector DB          │
        │         (or Qdrant/Pinecone)        │
        └────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  OpenAI LLM │
                    │  (GPT-4o)   │
                    └─────────────┘
```

## Project Structure

```
smit-teaching-agent/
├── app/
│   ├── api/
│   │   └── routes.py        # FastAPI endpoint definitions
│   ├── rag/
│   │   ├── ingestion.py     # Document loading (PDF, DOCX, TXT, MD)
│   │   ├── chunking.py      # Text splitting with configurable overlap
│   │   ├── embeddings.py    # OpenAI embedding generation + ChromaDB storage
│   │   ├── retrieval.py     # Semantic similarity search
│   │   └── generator.py     # LLM prompt building + answer generation
│   ├── services/
│   │   └── teaching.py      # Orchestration layer (orchestrates RAG pipeline)
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── utils/
│   │   ├── config.py        # Environment-based configuration
│   │   └── logger.py        # Structured logging
│   └── main.py              # FastAPI application entrypoint
├── frontend/
│   └── streamlit_app.py     # Streamlit chat UI
├── data/
│   ├── sample/              # Sample SMIT course materials
│   └── uploads/             # Uploaded files (auto-created)
├── vectordb/                # ChromaDB persistence (auto-created)
├── tests/
│   ├── test_rag.py          # Unit tests for RAG components
│   └── test_api.py          # API integration tests
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Prerequisites

- Python 3.11+
- OpenAI API key (or compatible LLM API endpoint)
- [uv](https://docs.astral.sh/uv/) (optional, for faster package management)

## Setup

### 1. Clone & Navigate

```bash
git clone <repo-url>
cd smit-teaching-agent
```

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your LLM API key:

```env
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 5. Run the Frontend

In a separate terminal:

```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### Docker (Alternative)

```bash
docker compose up --build
```

This starts both the API (port 8000) and frontend (port 8501).

## API Endpoints

| Method | Endpoint               | Description                    |
|--------|------------------------|--------------------------------|
| GET    | `/api/v1/health`       | Health check                   |
| POST   | `/api/v1/upload`       | Upload learning documents      |
| POST   | `/api/v1/chat`         | Ask a question                 |
| POST   | `/api/v1/quiz`         | Generate quiz questions        |
| POST   | `/api/v1/summary`      | Summarize documents            |
| GET    | `/api/v1/recommendations` | Get learning recommendations |
| GET    | `/api/v1/history/{id}` | Get conversation history       |
| GET    | `/api/v1/stats`        | Vector store statistics        |

### Example: Upload a Document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@data/sample/smit_courses.md"
```

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'
```

### Example: Generate a Quiz

```bash
curl -X POST http://localhost:8000/api/v1/quiz \
  -H "Content-Type: application/json" \
  -d '{"num_questions": 5, "difficulty": "easy"}'
```

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable              | Default                | Description                       |
|-----------------------|------------------------|-----------------------------------|
| `LLM_API_KEY`         | —                      | OpenAI API key                    |
| `LLM_MODEL`           | `gpt-4o-mini`          | LLM model name                    |
| `LLM_TEMPERATURE`     | `0.3`                  | Response creativity (0-1)        |
| `EMBEDDING_MODEL`     | `text-embedding-3-small` | Embedding model                 |
| `CHUNK_SIZE`          | `1000`                 | Document chunk size (chars)       |
| `CHUNK_OVERLAP`       | `200`                  | Chunk overlap (chars)             |
| `RETRIEVAL_TOP_K`     | `4`                    | Number of chunks to retrieve      |
| `VECTOR_DB_TYPE`      | `chroma`               | Vector DB backend                 |
| `CHROMA_PERSIST_DIR`  | `vectordb`             | ChromaDB persistence directory    |

## Sample Data

A sample SMIT course material file is provided at `data/sample/smit_courses.md`. Upload it via the UI or API to start testing immediately.

## License

MIT
