"""
src/rag_engine.py — End-to-End RAG Engine Orchestrator (Phase 2, Full Pipeline).

This module is the single entry point for the entire Phase 2 pipeline.
Every user query flows through these stages in sequence:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Stage 0: PII Scrubbing          (guardrail.py – PIIScrubber)           │
  │ Stage 1: Advisory Detection     (query_processor.py – QueryProcessor)  │
  │          Fund Name Normalisation                                        │
  │          Metadata Pre-filter                                            │
  │ Stage 2: Hybrid Retrieval       (retriever.py – HybridRetriever)       │
  │          Dense (ChromaDB/BGE) + Sparse (BM25) → RRF fusion top-5       │
  │ Stage 3: Cross-Encoder Re-rank  (reranker.py – CrossEncoderReranker)   │
  │          Score all (query, chunk) pairs → select top-3                  │
  │ Stage 4: Confidence Tiering     (confidence.py – ConfidenceThresholder)│
  │          HIGH → generate | MODERATE → generate + hedge | LOW → refuse  │
  │ Stage 5: LLM Generation         (llm.py – LLMOrchestrator)             │
  │          Compile prompt → Ollama Llama 3.1 8B → raw answer             │
  │ Stage 6: Output Validation      (output_validator.py – OutputValidator)│
  │          Sentence cap, citation check, footer check, advisory flag     │
  └─────────────────────────────────────────────────────────────────────────┘

Short-circuit exits (return before reaching Stage 2):
  - Advisory query detected → return ADVISORY_REFUSAL (Stage 1)
  - LOW confidence tier     → return LOW_CONFIDENCE_REFUSAL (Stage 4)
  - LLM unavailable         → return safe fallback (Stage 5 error handler)

Usage:
    from src.rag_engine import RAGEngine

    engine = RAGEngine()
    response = engine.answer("What is the exit load of SBI Small Cap Fund?")
    print(response.final_answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.confidence import ConfidenceTier, ConfidenceThresholder
from src.guardrail import PIIScrubber
from src.llm import LLMOrchestrator
from src.output_validator import OutputValidator, ValidationResult
from src.query_processor import QueryProcessor
from src.reranker import CrossEncoderReranker, RerankedChunk
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ── RAG Response ───────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Complete response object returned by RAGEngine.answer().
    Contains the final answer alongside diagnostic metadata for observability.
    """
    # The final user-facing answer (always populated)
    final_answer: str

    # Diagnostic fields (useful for evaluation, logging, debugging)
    original_query:     str  = ""
    scrubbed_query:     str  = ""
    normalized_query:   str  = ""
    detected_fund:      str  = ""
    is_advisory:        bool = False
    pii_redacted:       list[str] = field(default_factory=list)

    confidence_tier:    str  = ""           # "HIGH" | "MODERATE" | "LOW" | "ADVISORY"
    top_ce_score:       float = 0.0         # Best cross-encoder score
    num_candidates:     int  = 0            # Chunks retrieved before re-ranking
    num_reranked:       int  = 0            # Chunks after re-ranking

    # Source citations from top chunk
    top_source_url:     str  = ""
    top_pub_date:       str  = ""

    # Output validation info
    validation:         ValidationResult | None = None


# ── RAG Engine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    End-to-end Mutual Fund FAQ RAG engine.

    Instantiating this class loads:
      - BAAI/bge-small-en-v1.5 embedding model
      - cross-encoder/ms-marco-MiniLM-L-6-v2 re-ranking model
      - Ollama client (Llama 3.1 8B must be running separately)
      - ChromaDB persistent client

    These are heavy objects — instantiate once and reuse (e.g., at app startup).
    The models are kept in memory for the lifetime of the engine instance.

    Usage:
        engine = RAGEngine()
        response = engine.answer("What is the TER of SBI Contra Fund?")
        print(response.final_answer)
    """

    def __init__(self):
        logger.info("[RAG_ENGINE] Initialising pipeline components...")
        self.scrubber      = PIIScrubber()
        self.query_proc    = QueryProcessor()
        self.retriever     = HybridRetriever()
        self.reranker      = CrossEncoderReranker()
        self.thresholder   = ConfidenceThresholder()
        self.llm           = LLMOrchestrator()
        self.validator     = OutputValidator()
        logger.info("[RAG_ENGINE] All components loaded. Pipeline ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def answer(self, query: str) -> RAGResponse:
        """
        Process a user query end-to-end and return a fully validated RAGResponse.

        Args:
            query: Raw user input (may contain PII, informal fund names, etc.)

        Returns:
        """
        response = RAGResponse(original_query=query, final_answer="")

        # ── Stage 0: PII Scrubbing ─────────────────────────────────────────────
        scrubbed, redactions = self.scrubber.scrub(query)
        response.scrubbed_query = scrubbed
        response.pii_redacted   = redactions

        # ── Stage 1: Query Preprocessing ──────────────────────────────────────
        processed = self.query_proc.process(
            scrubbed_query=scrubbed,
            original_query=query,
        )
        response.normalized_query = processed.cleaned_query
        response.is_advisory      = processed.is_advisory
        response.detected_fund    = processed.detected_fund or ""

        # Short-circuit: advisory query
        if processed.is_advisory:
            logger.info("[RAG_ENGINE] Short-circuit: advisory query.")
            response.final_answer    = processed.refusal_message
            response.confidence_tier = "ADVISORY"
            return response

        # ── Stage 2: Hybrid Retrieval (Dense + BM25 via RRF) ──────────────────
        candidates = self.retriever.retrieve(
            query=processed.cleaned_query,
            fund_filter=processed.chroma_filter,
        )
        response.num_candidates = len(candidates)

        if not candidates:
            logger.warning("[RAG_ENGINE] No candidates retrieved. Returning refusal.")
            response.final_answer    = self.thresholder.get_refusal_message()
            response.confidence_tier = ConfidenceTier.LOW.value
            return response

        # ── Stage 3: Cross-Encoder Re-ranking ─────────────────────────────────
        reranked: list[RerankedChunk] = self.reranker.rerank(
            query=processed.cleaned_query,
            candidates=candidates,
            top_k=3,
        )
        response.num_reranked = len(reranked)

        # Guard: reranker may return [] on total inference failure
        if not reranked:
            logger.error("[RAG_ENGINE] Re-ranker returned no results. Returning refusal.")
            response.final_answer    = self.thresholder.get_refusal_message()
            response.confidence_tier = ConfidenceTier.LOW.value
            return response

        top_chunk    = reranked[0]
        top_ce_score = top_chunk.cross_encoder_score
        response.top_ce_score  = top_ce_score
        response.top_source_url = top_chunk.metadata.get("source_url", "")
        response.top_pub_date   = top_chunk.metadata.get("publication_date", "")

        # ── Stage 4: Confidence Tiering ────────────────────────────────────────
        tier = self.thresholder.classify(top_score=top_ce_score)
        response.confidence_tier = tier.value

        # Short-circuit: low confidence — bypass LLM
        if not self.thresholder.should_call_llm(tier):
            logger.info("[RAG_ENGINE] Short-circuit: LOW confidence. Returning refusal.")
            response.final_answer = self.thresholder.get_refusal_message()
            return response

        hedge_prefix = self.thresholder.get_hedge_prefix(tier)

        # ── Stage 5: LLM Generation ────────────────────────────────────────────
        try:
            raw_answer = self.llm.generate_answer(
                query=processed.cleaned_query,
                chunks=reranked,
                hedge_prefix=hedge_prefix,
            )
        except Exception as e:
            logger.error(f"[RAG_ENGINE] LLM generation error: {e}")
            # Safe fallback: structured refusal with citation from top chunk
            response.final_answer = OutputValidator.build_fallback(top_chunk)
            return response

        # ── Stage 6: Output Validation ─────────────────────────────────────────
        validation = self.validator.validate(
            raw_response=raw_answer,
            top_chunk=top_chunk,
        )
        response.validation   = validation
        response.final_answer = validation.final_response

        logger.info(
            f"[RAG_ENGINE] Done | tier={tier.value} | "
            f"ce_score={top_ce_score:.4f} | "
            f"valid={validation.is_valid} | "
            f"corrections={validation.corrections_applied}"
        )
        return response
