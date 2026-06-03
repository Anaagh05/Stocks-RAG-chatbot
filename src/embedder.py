"""
src/embedder.py — Embedding Generation + Pinecone Storage.
"""

import hashlib
import logging

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    HF_TOKEN,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    QUERY_INSTRUCTION,
)

logger = logging.getLogger(__name__)

# ── Chunk ID Generator ─────────────────────────────────────────────────────────

def _chunk_id(text: str, metadata: dict) -> str:
    raw = f"{metadata['source_url']}|{metadata['section_name']}|{text[:120]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# ── Embedder ───────────────────────────────────────────────────────────────────

class Embedder:
    """
    Generates BAAI/bge-small-en-v1.5 embeddings via HF Inference API
    and upserts them into Pinecone Serverless.
    """

    def __init__(self):
        logger.info(f"[EMBEDDER] Connecting to Pinecone Index: {PINECONE_INDEX_NAME}")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Initialize Pinecone Index
        # Ensure the index exists before running this. (Needs to be created in Pinecone UI or via script)
        try:
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            logger.error(f"[EMBEDDER] Failed to connect to Pinecone index: {e}")
            self.index = None

        logger.info(f"[EMBEDDER] Initializing HuggingFace Embeddings: {EMBEDDING_MODEL}")
        self.embeddings_model = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
        )

    def embed_and_store(self, chunks: list[dict], fund_name: str) -> int:
        """
        Embed all chunks for a fund and upsert into Pinecone.
        """
        if not chunks:
            logger.warning(f"[EMBEDDER] No chunks to embed for {fund_name}. Skipping.")
            return 0
        if self.index is None:
            logger.error("[EMBEDDER] No Pinecone index available. Skipping.")
            return 0

        # Step 1: Delete stale embeddings
        self._delete_fund_embeddings(fund_name)

        # Step 2: Batch-embed texts
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        ids = [_chunk_id(c["text"], c["metadata"]) for c in chunks]

        # Step 3: Embed and Upsert
        # We process in batches to avoid HF API limits or Pinecone payload limits.
        total_upserted = 0
        for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            batch_meta = metadatas[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            batch_ids = ids[batch_start : batch_start + EMBEDDING_BATCH_SIZE]

            # HF Inference API allows passing a list of strings
            try:
                batch_embeddings = self.embeddings_model.embed_documents(batch_texts)
            except Exception as e:
                logger.error(f"[EMBEDDER] HF API Embedding failed: {e}")
                continue

            # Prepare Pinecone vectors format: list of dicts
            vectors = []
            for i in range(len(batch_texts)):
                # Attach the text chunk to the metadata so we can retrieve it
                meta = batch_meta[i].copy()
                meta["text"] = batch_texts[i]
                vectors.append({
                    "id": batch_ids[i],
                    "values": batch_embeddings[i],
                    "metadata": meta
                })

            try:
                self.index.upsert(vectors=vectors)
                total_upserted += len(vectors)
                logger.debug(f"[EMBEDDER] Upserted batch of {len(vectors)} chunks.")
            except Exception as e:
                logger.error(f"[EMBEDDER] Pinecone upsert failed: {e}")

        logger.info(f"[EMBEDDER] Stored {total_upserted} chunks for {fund_name}.")
        return total_upserted

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a user query using the asymmetric instruction prefix.
        """
        prefixed = QUERY_INSTRUCTION + query
        return self.embeddings_model.embed_query(prefixed)

    def _delete_fund_embeddings(self, fund_name: str) -> None:
        """
        Delete all existing embeddings tagged with `fund_name` from Pinecone.
        """
        try:
            # Pinecone serverless supports deleting by metadata
            self.index.delete(filter={"fund_name": {"$eq": fund_name}})
            logger.info(f"[EMBEDDER] Deleted stale chunks for {fund_name}.")
        except Exception as e:
            logger.warning(f"[EMBEDDER] Could not delete stale chunks for {fund_name}: {e}")

    def count(self) -> int:
        """Return total number of chunks currently in the vector store."""
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0
