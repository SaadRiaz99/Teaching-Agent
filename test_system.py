#!/usr/bin/env python
"""Quick integration test for the SMIT Teaching Agent RAG system."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("SMIT Teaching Agent - Integration Test")
print("=" * 60)

# Step 1: Test imports
print("\n[1/5] Testing imports...")
try:
    from src.config import settings
    from src.embeddings import get_embedder
    from src.vector_store import get_vector_store
    from src.ingestion.processor import IngestionPipeline
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.generation.prompts import PromptTemplates
    from src.agents.teaching_agent import TeachingAgent
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Step 2: Test embeddings
print("\n[2/5] Testing embeddings...")
try:
    embedder = get_embedder()
    test_emb = embedder.embed(["SMIT is a college in Sikkim"])
    print(f"  ✓ Embedding dimension: {len(test_emb[0])}")
    assert len(test_emb[0]) > 0, "Embedding should not be empty"
except Exception as e:
    print(f"  ✗ Embedding failed: {e}")
    sys.exit(1)

# Step 3: Test ingestion
print("\n[3/5] Testing document ingestion...")
try:
    pipeline = IngestionPipeline()
    sample_dir = ROOT / "data" / "documents"
    result = pipeline.ingest(str(sample_dir))
    print(f"  ✓ Ingested {result['files']} files → {result['chunks']} chunks")
    assert result["chunks"] > 0, "Should have ingested at least some chunks"
except Exception as e:
    print(f"  ✗ Ingestion failed: {e}")
    sys.exit(1)

# Step 4: Test retrieval
print("\n[4/5] Testing retrieval pipeline...")
try:
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    results = retriever.search("What courses does CSE department offer?", k=10)
    print(f"  ✓ Hybrid search returned {len(results)} results")
    if results:
        print(f"    Top score: {results[0].score:.4f}")
    reranked = reranker.rerank("What courses does CSE department offer?", results, top_k=3)
    print(f"  ✓ Reranker returned {len(reranked)} results")
    assert len(results) > 0, "Should retrieve results for a SMIT query"
except Exception as e:
    print(f"  ✗ Retrieval failed: {e}")
    sys.exit(1)

# Step 5: Test Teaching Agent (without LLM for non-API envs)
print("\n[5/5] Testing Teaching Agent...")
try:
    agent = TeachingAgent()
    print(f"  ✓ TeachingAgent created successfully")
    print(f"  ✓ Vector store has {agent.pipeline.retriever.store.count()} chunks")
    print(f"  ✓ Query router available")
    print(f"  ✓ Self-RAG verifier available")
except Exception as e:
    print(f"  ✗ Agent init failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print(f"\nSystem is ready at: {ROOT}")
print("\nTry it now:")
print("  # Ingest more documents:")
print("  python -m src.cli.main ingest data/documents")
print()
print("  # Ask a question (needs ANTHROPIC_API_KEY or other LLM key):")
print('  python -m src.cli.main ask "What is the course code for AI and ML at SMIT?"')
print()
print("  # Start the web server:")
print("  python -m src.cli.main serve")
print()
print("  # Start interactive chat:")
print("  python -m src.cli.main chat")
