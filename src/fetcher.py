"""
src/fetcher.py — Incremental Document Fetcher with ETag / Last-Modified Caching.

Strategy (Phase 1, Section 2):
- Issue an HTTP HEAD request to each URL before downloading.
- Compare ETag or Last-Modified headers to values stored in fetch_cache.json.
- Only download if the document has changed (or has never been downloaded).
- Delete stale ChromaDB embeddings for the fund before re-ingesting.
"""

import json
import logging
from datetime import date
from typing import Optional
from urllib.parse import urlparse

import httpx

from config import CORPUS_SOURCES, FETCH_CACHE_PATH, CORPUS_DIR

logger = logging.getLogger(__name__)


# ── Cache Helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    """Load the ETag / Last-Modified cache from disk."""
    if FETCH_CACHE_PATH.exists():
        with open(FETCH_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the updated cache to disk."""
    FETCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FETCH_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _url_to_filename(url: str) -> str:
    """Derive a safe local filename from a URL."""
    parsed = urlparse(url)
    # Use the last path segment; strip slashes
    name = parsed.path.strip("/").replace("/", "_")
    # Detect content type to choose extension (defaulting to .html for Groww pages)
    return name + ".html"


# ── Core Fetcher ───────────────────────────────────────────────────────────────

class IncrementalFetcher:
    """
    Fetches documents from the corpus URL list using ETag/Last-Modified caching.
    Only downloads a document when the remote version has actually changed.
    """

    def __init__(self, timeout: int = 30):
        self.cache = _load_cache()
        self.timeout = timeout
        # Browser-like User-Agent to avoid 403 blocks from Groww
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def fetch_all(self) -> list[dict]:
        """
        Iterate over all corpus sources. For each:
        - Check if the remote document has changed via HEAD + cache comparison.
        - Download if changed or new.
        - Return list of metadata dicts for newly fetched documents.
        """
        fetched = []
        for source in CORPUS_SOURCES:
            result = self._fetch_one(source)
            if result:
                fetched.append(result)

        _save_cache(self.cache)
        logger.info(f"[FETCHER] Done. {len(fetched)} document(s) downloaded / updated.")
        return fetched

    def _fetch_one(self, source: dict) -> Optional[dict]:
        """
        Process a single source entry. Returns metadata dict if the file was
        downloaded (new or changed), or None if it was a cache hit (unchanged).
        """
        url        = source["url"]
        fund_name  = source["fund_name"]
        doc_type   = source["doc_type"]
        filename   = _url_to_filename(url)
        local_path = CORPUS_DIR / filename

        cached_etag     = self.cache.get(url, {}).get("etag")
        cached_modified = self.cache.get(url, {}).get("last_modified")

        remote_etag     = None
        remote_modified = None

        # ── Step 1: HEAD request to check freshness ────────────────────────
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                head_resp = client.head(url, headers=self.headers)
            if head_resp.status_code == 200:
                remote_etag     = head_resp.headers.get("etag")
                remote_modified = head_resp.headers.get("last-modified")
            else:
                logger.warning(
                    f"[FETCHER] HEAD request to {url} returned status {head_resp.status_code}. "
                    "Falling back to direct GET download."
                )
        except Exception as e:
            logger.warning(
                f"[FETCHER] HEAD request to {url} failed: {e}. "
                "Falling back to direct GET download."
            )

        try:
            # ── Step 2: Compare with cache ─────────────────────────────────────
            if local_path.exists():
                if remote_etag and remote_etag == cached_etag:
                    logger.info(f"[CACHE HIT]  {fund_name} — ETag match. Skipping.")
                    return None
                if remote_modified and remote_modified == cached_modified:
                    logger.info(f"[CACHE HIT]  {fund_name} — Last-Modified match. Skipping.")
                    return None

            # ── Step 3: Download the document ─────────────────────────────────
            logger.info(f"[DOWNLOAD]   {fund_name} — Fetching {url}")
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                get_resp = client.get(url, headers=self.headers)
            get_resp.raise_for_status()

            # Save to corpus directory
            local_path.write_bytes(get_resp.content)

            # Detect publication date from headers; fall back to today
            pub_date = (
                remote_modified.split(",")[-1].strip()[:11].strip()
                if remote_modified else str(date.today())
            )

            # ── Step 4: Update cache ───────────────────────────────────────────
            self.cache[url] = {
                "etag": remote_etag,
                "last_modified": remote_modified,
                "local_path": str(local_path),
                "last_fetched": str(date.today()),
            }

            logger.info(f"[SAVED]      {fund_name} → {local_path.name}")

            return {
                "url": url,
                "fund_name": fund_name,
                "doc_type": doc_type,
                "local_path": str(local_path),
                "publication_date": pub_date,
                "ingestion_date": str(date.today()),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"[HTTP ERROR] {fund_name}: {e.response.status_code} — {url}")
            return None
        except httpx.RequestError as e:
            logger.error(f"[NET ERROR]  {fund_name}: {e} — {url}")
            return None
