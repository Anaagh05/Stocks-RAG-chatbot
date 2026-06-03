"""
src/retriever.py — Semantic Retrieval via Pinecone Serverless.

Retrieves top-10 candidates from Pinecone using BGE embeddings.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.embedder import Embedder

logger = logging.getLogger(__name__)

TOP_K = 10  # Candidates retrieved from Vector DB

# ── Result Dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its text and metadata."""
    chunk_id: str
    text: str
    metadata: dict
    pinecone_score: float = 0.0

# ── Retriever ──────────────────────────────────────────────────────────────────

class Retriever:
    """
    Semantic retrieval using Pinecone Serverless.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def retrieve(
        self,
        query: str,
        fund_filter: Optional[dict] = None,
        top_k: int = TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Run semantic retrieval for a query.

        Args:
            query:       Raw (pre-processed, PII-scrubbed) user query string.
            fund_filter: Optional filter dict (e.g., {"fund_name": {"$eq": "..."}}).
            top_k:       Candidates to retrieve.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        query_embedding = self.embedder.embed_query(query)

        try:
            if fund_filter:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_values=False,
                    include_metadata=True,
                    filter=fund_filter
                )
            else:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_values=False,
                    include_metadata=True,
                )
        except Exception as e:
            logger.error(f"[RETRIEVER] Pinecone query failed: {e}")
            return []

        retrieved_chunks = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            text = meta.get("text", "")
            # Return without 'text' key in metadata if we want it clean
            clean_meta = {k: v for k, v in meta.items() if k != "text"}
            
            chunk = RetrievedChunk(
                chunk_id=match["id"],
                text=text,
                metadata=clean_meta,
                pinecone_score=match.get("score", 0.0)
            )
            retrieved_chunks.append(chunk)

        logger.info(
            f"[RETRIEVER] Query: {query[:60]!r} | "
            f"Retrieved: {len(retrieved_chunks)} chunks"
        )
        return retrieved_chunks
