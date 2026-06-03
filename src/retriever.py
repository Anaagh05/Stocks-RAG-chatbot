"""
src/retriever.py — Hybrid Retrieval Engine: Dense + BM25 via RRF (Phase 2, Section 3).

Two retrievers run in parallel, their ranked result sets are merged using
Reciprocal Rank Fusion (RRF), producing a single unified top-5 candidate list.

Dense Retriever (Semantic):
    Queries ChromaDB with BGE-embedded query vectors (cosine similarity).
    Retrieves top-10 candidates. Understands semantic paraphrasing.

Sparse Retriever (BM25):
    In-memory BM25 index over all chunks fetched from ChromaDB.
    Retrieves top-10 candidates. Excels at exact financial term matching
    (ELSS, NAV, exit load percentages, fund names, riskometer).

RRF Fusion (k=60):
    Merges both ranked lists using the formula:
        RRF_score(d) = Σ 1 / (k + rank_i)
    Selects the top-5 chunks by fused score for downstream re-ranking / LLM use.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_BATCH_SIZE
from src.embedder import Embedder

logger = logging.getLogger(__name__)

# ── RRF Constant ───────────────────────────────────────────────────────────────
RRF_K = 60          # Standard value; higher k reduces the impact of top-rank dominance
TOP_K_EACH = 10     # Candidates retrieved from each retriever
TOP_K_FUSED = 5     # Final fused candidates forwarded to re-ranker / LLM


# ── Result Dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its text, metadata, and fused RRF score."""
    chunk_id: str
    text: str
    metadata: dict
    dense_rank: Optional[int] = None    # Rank from ChromaDB dense retriever (1-indexed)
    bm25_rank: Optional[int] = None     # Rank from BM25 retriever (1-indexed)
    rrf_score: float = 0.0              # Final merged score (higher = more relevant)


# ── ChromaDB Client Helper ─────────────────────────────────────────────────────

def _get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Tokenizer Helper ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing, splitting, and stripping common punctuation."""
    tokens = []
    for word in text.lower().split():
        cleaned = word.strip(".,;:!?()[]\"'")
        if cleaned:
            tokens.append(cleaned)
    return tokens


# ── BM25 Index Builder ─────────────────────────────────────────────────────────

def _build_bm25_index(
    collection,
    fund_filter: Optional[dict] = None,
) -> tuple[BM25Okapi, list[str], list[str], list[dict]]:
    """
    Fetch all stored chunks (optionally filtered by fund_name) from ChromaDB
    and build an in-memory BM25Okapi index over their texts.

    Returns:
        (bm25_index, all_ids, all_texts, all_metadatas)
    """
    try:
        if fund_filter:
            results = collection.get(
                include=["documents", "metadatas"],
                limit=10_000,
                where=fund_filter
            )
        else:
            results = collection.get(
                include=["documents", "metadatas"],
                limit=10_000
            )
    except Exception as e:
        logger.error(f"[BM25] ChromaDB fetch failed: {e}")
        return BM25Okapi([[""]]), [], [], []

    ids_list = results.get("ids")
    texts_list = results.get("documents")
    metadatas_list = results.get("metadatas")

    ids       = ids_list if ids_list is not None else []
    texts     = texts_list if texts_list is not None else []
    metadatas = [dict(m) if m else {} for m in metadatas_list] if metadatas_list is not None else []

    if not texts:
        logger.warning("[BM25] No chunks found in ChromaDB for given filter.")
        return BM25Okapi([[""]]), [], [], []

    # Tokenize: lowercase + split + strip trailing/leading punctuation
    tokenized = [_tokenize(doc) for doc in texts]
    bm25 = BM25Okapi(tokenized)

    logger.debug(f"[BM25] Index built over {len(texts)} chunks.")
    return bm25, ids, texts, metadatas


# ── Hybrid Retriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines dense (ChromaDB/BGE) and sparse (BM25) retrieval via RRF fusion.

    Usage:
        retriever = HybridRetriever()
        chunks = retriever.retrieve("What is the exit load of SBI Small Cap Fund?",
                                    fund_filter={"fund_name": "SBI Small Cap Fund"})
    """

    def __init__(self):
        self.embedder   = Embedder()
        self.collection = _get_collection()

    def retrieve(
        self,
        query: str,
        fund_filter: Optional[dict] = None,
        top_k_each: int = TOP_K_EACH,
        top_k_fused: int = TOP_K_FUSED,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval for a query and return top-k fused chunks.

        Args:
            query:       Raw (pre-processed, PII-scrubbed) user query string.
            fund_filter: Optional ChromaDB `where` dict to restrict search to a fund.
            top_k_each:  Candidates from each retriever (default 10).
            top_k_fused: Final fused candidates to return (default 5).

        Returns:
            List of RetrievedChunk sorted by RRF score descending.
        """
        # ── Dense Retrieval ────────────────────────────────────────────────────
        dense_results = self._dense_retrieve(query, fund_filter, top_k_each)

        # ── Sparse (BM25) Retrieval ────────────────────────────────────────────
        bm25_results = self._bm25_retrieve(query, fund_filter, top_k_each)

        # ── RRF Fusion ─────────────────────────────────────────────────────────
        fused = self._rrf_merge(dense_results, bm25_results, top_k_fused)

        logger.info(
            f"[RETRIEVER] Query: {query[:60]!r} | "
            f"Dense: {len(dense_results)}, BM25: {len(bm25_results)}, "
            f"Fused top-{top_k_fused}: {len(fused)}"
        )
        return fused

    # ── Dense Retriever ────────────────────────────────────────────────────────

    def _dense_retrieve(
        self,
        query: str,
        fund_filter: Optional[dict],
        top_k: int,
    ) -> list[tuple[str, str, dict]]:
        """
        Query ChromaDB using the BGE query embedding (with instruction prefix).
        Returns list of (chunk_id, text, metadata) tuples, ordered by cosine similarity.
        """
        query_embedding = self.embedder.embed_query(query)

        try:
            if fund_filter:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],  # type: ignore
                    where=fund_filter
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]  # type: ignore
                )
        except Exception as e:
            logger.error(f"[DENSE] ChromaDB query failed: {e}")
            return []

        ids_list = results.get("ids")
        texts_list = results.get("documents")
        metadatas_list = results.get("metadatas")

        ids       = ids_list[0] if ids_list is not None else []
        texts     = texts_list[0] if texts_list is not None else []
        metadatas = metadatas_list[0] if metadatas_list is not None else []

        return [
            (cid, text, dict(meta) if meta else {})
            for cid, text, meta in zip(ids, texts, metadatas)
        ]

    # ── BM25 Retriever ─────────────────────────────────────────────────────────

    def _bm25_retrieve(
        self,
        query: str,
        fund_filter: Optional[dict],
        top_k: int,
    ) -> list[tuple[str, str, dict]]:
        """
        Tokenize the query and rank all indexed chunks using BM25Okapi.
        Returns list of (chunk_id, text, metadata) tuples ordered by BM25 score desc.
        """
        bm25, all_ids, all_texts, all_metadatas = _build_bm25_index(
            self.collection, fund_filter
        )

        if not all_texts:
            return []

        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            (all_ids[i], all_texts[i], all_metadatas[i])
            for i in top_indices
            if scores[i] > 0   # Skip zero-score results (no term overlap)
        ]

    # ── RRF Fusion ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        dense_results: list[tuple[str, str, dict]],
        bm25_results: list[tuple[str, str, dict]],
        top_k: int,
        k: int = RRF_K,
    ) -> list[RetrievedChunk]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion:
            RRF_score(d) = Σ 1 / (k + rank_i)

        Chunks appearing in both lists accumulate scores from both.
        Chunks appearing in only one list still contribute their single score.

        Returns top_k RetrievedChunk objects sorted by rrf_score descending.
        """
        # Map chunk_id → RetrievedChunk (accumulate ranks from both retrievers)
        chunk_map: dict[str, RetrievedChunk] = {}

        # Process dense results (rank is 1-indexed)
        for rank, (cid, text, meta) in enumerate(dense_results, start=1):
            if cid not in chunk_map:
                chunk_map[cid] = RetrievedChunk(chunk_id=cid, text=text, metadata=meta)
            chunk_map[cid].dense_rank = rank
            chunk_map[cid].rrf_score += 1.0 / (k + rank)

        # Process BM25 results
        for rank, (cid, text, meta) in enumerate(bm25_results, start=1):
            if cid not in chunk_map:
                chunk_map[cid] = RetrievedChunk(chunk_id=cid, text=text, metadata=meta)
            chunk_map[cid].bm25_rank = rank
            chunk_map[cid].rrf_score += 1.0 / (k + rank)

        # Sort all candidates by RRF score descending, take top-k
        ranked = sorted(chunk_map.values(), key=lambda c: c.rrf_score, reverse=True)
        return ranked[:top_k]
