"""
src/reranker.py — Cohere Rerank API (Phase 2, Section 4).
"""

import logging
from dataclasses import dataclass

import cohere

from src.retriever import RetrievedChunk
from config import COHERE_API_KEY

logger = logging.getLogger(__name__)

# ── Model Configuration ────────────────────────────────────────────────────────

COHERE_RERANK_MODEL = "rerank-english-v3.0"
DEFAULT_TOP_K       = 3   

# ── Re-ranked Chunk ────────────────────────────────────────────────────────────

@dataclass
class RerankedChunk:
    chunk_id:            str
    text:                str
    metadata:            dict
    pinecone_score:      float = 0.0          
    cross_encoder_score: float = 0.0          

    @classmethod
    def from_retrieved(cls, chunk: RetrievedChunk, ce_score: float) -> "RerankedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            pinecone_score=chunk.pinecone_score,
            cross_encoder_score=ce_score,
        )

# ── Cohere Re-ranker ────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-ranks using Cohere's Rerank API.
    """

    def __init__(self, model_name: str = COHERE_RERANK_MODEL):
        self.model_name = model_name
        self.client = cohere.Client(api_key=COHERE_API_KEY)
        logger.info(f"[RERANKER] Cohere client initialised | model={model_name}")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RerankedChunk]:
        
        if not candidates:
            logger.warning("[RERANKER] No candidates to re-rank.")
            return []

        docs = [chunk.text for chunk in candidates]

        try:
            results = self.client.rerank(
                query=query,
                documents=docs,
                top_n=top_k,
                model=self.model_name,
            )
            
            reranked_chunks = []
            for result in results.results:
                original_chunk = candidates[result.index]
                reranked_chunks.append(
                    RerankedChunk.from_retrieved(original_chunk, ce_score=result.relevance_score)
                )

            logger.info(
                f"[RERANKER] Top-{top_k} after re-ranking | "
                f"Best score: {reranked_chunks[0].cross_encoder_score:.4f}"
            )
            return reranked_chunks

        except Exception as e:
            logger.error(f"[RERANKER] Cohere API reranking failed: {e}")
            # Graceful degradation
            return [
                RerankedChunk.from_retrieved(c, ce_score=c.pinecone_score)
                for c in candidates[:top_k]
            ]
