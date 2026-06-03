"""
src/ingest.py — Main Ingestion Pipeline Orchestrator.

Ties together all Phase 1 components in the correct order:
  1. IncrementalFetcher   → download changed/new documents only
  2. parse_document       → dual-parser (pdfplumber + pymupdf) + HTML parser
  3. chunk_document       → hybrid 3-mode chunker
  4. Embedder             → BAAI/bge-small-en-v1.5 + ChromaDB upsert

Can be run manually:  python -m src.ingest
Or triggered daily by: scheduler.py (APScheduler)
"""

import logging
import sys
from pathlib import Path

# Create data directory relative to project root before configuring logging
ROOT_DIR = Path(__file__).parent.parent.resolve()
(ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(ROOT_DIR / "data" / "ingest.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Local Module Imports ───────────────────────────────────────────────────────
# Adjust sys.path so src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fetcher import IncrementalFetcher
from src.parser import parse_document
from src.chunker import chunk_document
from src.embedder import Embedder


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

def run_ingestion() -> dict:
    """
    Execute the full ingestion pipeline:

    1. Fetch all documents (incremental, ETag-cached).
    2. For each newly fetched document:
       a. Parse it (PDF: pdfplumber + pymupdf; HTML: BeautifulSoup)
       b. Chunk it (mode determined by doc_type)
       c. Embed and upsert into ChromaDB
    3. Return a summary dict.

    Returns:
        {
            "fetched": int,     # documents downloaded
            "skipped": int,     # documents skipped (cache hit)
            "total_chunks": int # total chunks stored in vector DB
        }
    """
    logger.info("=" * 60)
    logger.info("[INGEST] Starting ingestion pipeline...")
    logger.info("=" * 60)

    # ── Step 1: Fetch ──────────────────────────────────────────────────────────
    fetcher = IncrementalFetcher()
    fetched_docs = fetcher.fetch_all()

    total_sources = 15  # from CORPUS_SOURCES in config.py
    skipped = total_sources - len(fetched_docs)
    logger.info(
        f"[INGEST] Fetch complete: {len(fetched_docs)} downloaded, {skipped} cache hits."
    )

    if not fetched_docs:
        logger.info("[INGEST] No new or changed documents. Vector DB is up to date.")
        return {"fetched": 0, "skipped": skipped, "total_chunks": 0}

    # ── Step 2–4: Parse → Chunk → Embed ───────────────────────────────────────
    embedder = Embedder()
    total_chunks = 0

    for doc_meta in fetched_docs:
        url              = doc_meta["url"]
        fund_name        = doc_meta["fund_name"]
        doc_type         = doc_meta["doc_type"]
        local_path       = doc_meta["local_path"]
        publication_date = doc_meta["publication_date"]

        logger.info(f"[INGEST] Processing: {fund_name} ({doc_type})")

        # ── Parse ──────────────────────────────────────────────────────────────
        try:
            parsed = parse_document(local_path, doc_type)
        except Exception as e:
            logger.error(f"[INGEST] Parse failed for {fund_name}: {e}")
            continue

        text_len   = len(parsed.get("text", ""))
        table_count = len(parsed.get("tables", []))
        logger.info(
            f"[INGEST]   Parsed: {text_len} chars, {table_count} table(s)"
        )

        if text_len == 0 and table_count == 0:
            logger.warning(f"[INGEST]   Empty parse result for {fund_name}. Skipping.")
            continue

        # ── Chunk ──────────────────────────────────────────────────────────────
        try:
            chunks = chunk_document(
                parsed=parsed,
                source_url=url,
                fund_name=fund_name,
                doc_type=doc_type,
                publication_date=publication_date,
            )
        except Exception as e:
            logger.error(f"[INGEST] Chunking failed for {fund_name}: {e}")
            continue

        logger.info(f"[INGEST]   Chunks: {len(chunks)}")

        if not chunks:
            logger.warning(f"[INGEST]   No chunks produced for {fund_name}. Skipping.")
            continue

        # ── Embed & Store ──────────────────────────────────────────────────────
        try:
            stored = embedder.embed_and_store(chunks, fund_name)
            total_chunks += stored
        except Exception as e:
            logger.error(f"[INGEST] Embedding failed for {fund_name}: {e}")
            continue

    logger.info("=" * 60)
    logger.info(
        f"[INGEST] Pipeline complete. "
        f"Fetched: {len(fetched_docs)}, "
        f"Skipped: {skipped}, "
        f"Total chunks in DB: {embedder.count()}"
    )
    logger.info("=" * 60)

    return {
        "fetched": len(fetched_docs),
        "skipped": skipped,
        "total_chunks": total_chunks,
    }


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_ingestion()
    print(f"\n✅ Ingestion complete: {result}")
