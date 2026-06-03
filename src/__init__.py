# src/__init__.py
# Makes src/ a Python package for clean imports across the project.
#
# Phase 2 modules:
#   guardrail        — PII scrubbing (Section 1)
#   query_processor  — Advisory detection, fund normalisation, pre-filter (Section 2)
#   retriever        — Hybrid dense + BM25 + RRF retrieval (Section 3)
#   reranker         — Cross-encoder re-ranking (Section 4)
#   confidence       — Tiered similarity thresholding (Section 5)
#   llm              — Prompt engineering + Ollama LLM orchestration (Section 6)
#   output_validator — Post-generation output validation (Section 7)
#   rag_engine       — End-to-end pipeline orchestrator
