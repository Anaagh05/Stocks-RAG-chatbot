"""
src/reranker.py — Cross-Encoder Re-ranking (Phase 2, Section 4).

After hybrid retrieval (dense + BM25 via RRF) produces the top-5 candidate
chunks, this module re-ranks them using a cross-encoder for precision.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Jointly encodes the (query, passage) pair — unlike bi-encoders which
    encode independently — allowing full attention across both sequences.
  - Trained on MS MARCO passage ranking, generalises well to financial Q&A.
  - Compact size (~22 MB) suitable for CPU inference on consumer hardware.

Why re-rank after hybrid retrieval?
  - Bi-encoder retrieval (dense + BM25) optimises for RECALL: surfacing all
    potentially relevant chunks into the candidate set.
  - Cross-encoder optimises for PRECISION: putting the single most relevant
    chunk at position 1, minimising hallucination risk when the top result
    is fed as primary context to the LLM.

Output:
  - A re-ordered list of RetrievedChunk objects, each annotated with a
    `cross_encoder_score` attribute (float, higher = more relevant).
  - Only the top-2 or top-3 are forwarded to the Prompt Compiler.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import CrossEncoder

from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Model Configuration ────────────────────────────────────────────────────────

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K       = 3   # Number of re-ranked candidates forwarded to LLM


# ── Re-ranked Chunk ────────────────────────────────────────────────────────────

@dataclass
class RerankedChunk:
    """
    A retrieved chunk annotated with a cross-encoder relevance score.
    Wraps RetrievedChunk to avoid mutating the upstream dataclass.
    """
    chunk_id:            str
    text:                str
    metadata:            dict
    rrf_score:           float = 0.0          # Inherited from hybrid retriever
    dense_rank:          Optional[int] = None
    bm25_rank:           Optional[int] = None
    cross_encoder_score: float = 0.0          # Score from this re-ranker (higher = better)

    @classmethod
    def from_retrieved(cls, chunk: RetrievedChunk, ce_score: float) -> "RerankedChunk":
        """Construct a RerankedChunk from a RetrievedChunk + cross-encoder score."""
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            rrf_score=chunk.rrf_score,
            dense_rank=chunk.dense_rank,
            bm25_rank=chunk.bm25_rank,
            cross_encoder_score=ce_score,
        )


# ── Cross-Encoder Re-ranker ────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-ranks a list of RetrievedChunk candidates using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        top_chunks = reranker.rerank(query, candidates, top_k=3)

    The cross-encoder jointly encodes each (query, chunk_text) pair and returns
    a relevance logit. Candidates are sorted by this score descending.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        logger.info(f"[RERANKER] Loading cross-encoder: {model_name}")
        # device=None → auto-detects CPU/GPU; use_fp16 saves memory on GPU
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("[RERANKER] Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RerankedChunk]:
        """
        Score all (query, chunk) pairs with the cross-encoder and return
        the top-k candidates sorted by relevance score descending.

        Args:
            query:      The user's query (PII-scrubbed, fund-normalised).
            candidates: Top-5 chunks from hybrid RRF retrieval.
            top_k:      Number of results to forward to the LLM (default 3).

        Returns:
            List of RerankedChunk sorted by cross_encoder_score descending.
        """
        if not candidates:
            logger.warning("[RERANKER] No candidates to re-rank.")
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk in candidates]

        try:
            scores: list[float] = self.model.predict(pairs).tolist()
        except Exception as e:
            logger.error(f"[RERANKER] Cross-encoder inference failed: {e}")
            # Graceful degradation: return candidates in original RRF order
            return [
                RerankedChunk.from_retrieved(c, ce_score=c.rrf_score)
                for c in candidates[:top_k]
            ]

        # Pair each candidate with its score, sort descending
        scored = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        logger.info(
            "[RERANKER] Re-ranking scores: "
            + ", ".join(f"{s:.4f}" for _, s in scored)
        )

        results = [
            RerankedChunk.from_retrieved(chunk, ce_score=score)
            for chunk, score in scored[:top_k]
        ]

        logger.info(
            f"[RERANKER] Top-{top_k} after re-ranking | "
            f"Best score: {results[0].cross_encoder_score:.4f}"
        )
        return results
