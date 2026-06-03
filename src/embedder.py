"""
src/embedder.py — Embedding Generation + ChromaDB Storage.

Strategy (Phase 1, Section 5):
- Model:  BAAI/bge-small-en-v1.5  (SOTA free local retrieval model via sentence-transformers)
- Batch processing: 64 chunks per batch for memory efficiency
- Asymmetric embedding: document chunks embedded without prefix;
  query embedding adds a task instruction prefix at search time.
- ChromaDB: local persistent store with metadata filtering support.
- Upsert: old embeddings for a fund are deleted before re-ingesting new ones
  (prevents stale duplicates after the daily scheduler updates a document).
"""

import hashlib
import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    QUERY_INSTRUCTION,
)

logger = logging.getLogger(__name__)


# ── ChromaDB Client (singleton) ────────────────────────────────────────────────

def _get_chroma_collection() -> Any:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


# ── Chunk ID Generator ─────────────────────────────────────────────────────────

def _chunk_id(text: str, metadata: dict) -> str:
    """
    Generate a stable, deterministic ID for a chunk based on its content
    and source metadata. This allows safe upserts without duplicate entries.
    """
    raw = f"{metadata['source_url']}|{metadata['section_name']}|{text[:120]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ── Embedder ───────────────────────────────────────────────────────────────────

class Embedder:
    """
    Loads BAAI/bge-small-en-v1.5, generates embeddings for document chunks,
    and upserts them into ChromaDB with full metadata.
    """

    def __init__(self):
        logger.info(f"[EMBEDDER] Loading model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = _get_chroma_collection()
        logger.info(f"[EMBEDDER] Connected to ChromaDB collection: {CHROMA_COLLECTION}")

    # ── Document Chunk Embedding (no prefix — asymmetric) ──────────────────────

    def embed_and_store(self, chunks: list[dict], fund_name: str) -> int:
        """
        Embed all chunks for a fund and upsert into ChromaDB.

        Steps:
        1. Delete all existing embeddings for `fund_name` (stale data removal).
        2. Batch-embed chunk texts using BAAI/bge-small-en-v1.5.
        3. Upsert (id, embedding, text, metadata) into ChromaDB.

        Returns the number of chunks stored.
        """
        if not chunks:
            logger.warning(f"[EMBEDDER] No chunks to embed for {fund_name}. Skipping.")
            return 0

        # ── Step 1: Delete stale embeddings for this fund ──────────────────────
        self._delete_fund_embeddings(fund_name)

        # ── Step 2: Batch-embed texts ──────────────────────────────────────────
        texts     = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids       = [_chunk_id(c["text"], c["metadata"]) for c in chunks]

        all_embeddings = []
        for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            # Document chunks: NO instruction prefix (asymmetric embedding)
            batch_embeddings = self.model.encode(
                batch,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,   # cosine similarity needs L2-norm
            ).tolist()
            all_embeddings.extend(batch_embeddings)
            logger.debug(
                f"[EMBEDDER] Embedded batch {batch_start}–{batch_start + len(batch)} "
                f"for {fund_name}"
            )

        # ── Step 3: Upsert into ChromaDB ───────────────────────────────────────
        # ChromaDB upsert handles insert-or-replace by ID
        self.collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"[EMBEDDER] Stored {len(chunks)} chunks for {fund_name}.")
        return len(chunks)

    # ── Query Embedding (with instruction prefix) ──────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a user query using the asymmetric instruction prefix.
        This is called at retrieval time, NOT during ingestion.

        BGE instruction prefix: improves retrieval relevance significantly.
        """
        prefixed = QUERY_INSTRUCTION + query
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    # ── Stale Data Removal ─────────────────────────────────────────────────────

    def _delete_fund_embeddings(self, fund_name: str) -> None:
        """
        Delete all existing embeddings tagged with `fund_name` from ChromaDB.
        Called before re-ingesting a fund whose document has changed.
        """
        try:
            results = self.collection.get(
                where={"fund_name": fund_name},
                include=[],   # only need IDs
            )
            stale_ids = results.get("ids", [])
            if stale_ids:
                self.collection.delete(ids=stale_ids)
                logger.info(
                    f"[EMBEDDER] Deleted {len(stale_ids)} stale chunks for {fund_name}."
                )
        except Exception as e:
            logger.warning(f"[EMBEDDER] Could not delete stale chunks for {fund_name}: {e}")

    # ── Collection Stats ───────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of chunks currently in the vector store."""
        return self.collection.count()
