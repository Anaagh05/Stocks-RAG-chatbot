"""
src/chunker.py — Hybrid Chunking Engine (Three-Mode Strategy).

Strategy (Phase 1, Section 4):

Mode A — Section-Aware Hierarchical Splitting  (SIDs & KIMs)
    Detect section headers via regex → split text at boundaries → one chunk per section.

Mode B — Table-Atomic Chunking  (Factsheets)
    Each extracted Markdown table becomes a single, indivisible chunk.
    Tables are NEVER split across two chunks.

Mode C — Paragraph + Sliding Window  (FAQ HTML Pages)
    Segment at <p>/<li> boundaries → sliding window of 400 chars, 100 chars overlap.

Universal metadata tags are attached to every chunk regardless of mode.
"""

import logging
import re
from datetime import date
from typing import Optional

from config import SLIDE_WINDOW_SIZE, SLIDE_WINDOW_OVERLAP

logger = logging.getLogger(__name__)


# ── Section Header Patterns (Mode A) ──────────────────────────────────────────
# Matches common SID / KIM section titles (case-insensitive, full line anchored).
_SECTION_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"Exit\s+Load"
    r"|Expense\s+Ratio"
    r"|Investment\s+Objective"
    r"|Benchmark"
    r"|Fund\s+Manager"
    r"|Riskometer"
    r"|Asset\s+Allocation"
    r"|Minimum\s+(?:SIP|Investment|Application|Redemption)"
    r"|Lock[\s\-]+in\s+Period"
    r"|Key\s+Information"
    r"|Scheme\s+(?:Type|Category|Objective|Details|Overview)"
    r"|Plans?\s+(?:and|&)\s+Options?"
    r"|Load\s+Structure"
    r"|NAV\s+(?:Applicability|Calculation)"
    r"|Dividend\s+Policy"
    r"|Redemption"
    r"|Taxation"
    r"|Portfolio\s+(?:Details|Turnover|Composition)"
    r"|Risk\s+(?:Factors|Profile|Mitigation)"
    r"|About\s+the\s+Fund"
    r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ── Metadata Builder ───────────────────────────────────────────────────────────

def _make_metadata(
    source_url: str,
    fund_name: str,
    doc_type: str,
    section_name: str,
    publication_date: str,
    ingestion_date: Optional[str] = None,
) -> dict:
    return {
        "source_url": source_url,
        "fund_name": fund_name,
        "doc_type": doc_type,
        "section_name": section_name,
        "publication_date": publication_date,
        "ingestion_date": ingestion_date or str(date.today()),
    }


# ── Mode A: Section-Aware Hierarchical Splitting ───────────────────────────────

def chunk_by_sections(
    text: str,
    source_url: str,
    fund_name: str,
    doc_type: str,
    publication_date: str,
) -> list[dict]:
    """
    Split text at detected section header boundaries.
    Each section's full content becomes one self-contained chunk.
    Returns list of {'text': str, 'metadata': dict}.
    """
    chunks = []
    # Find all header match positions
    matches = list(_SECTION_HEADER_PATTERN.finditer(text))

    if not matches:
        # No headers found: treat entire text as a single general chunk
        logger.debug(f"[CHUNKER-A] No section headers found in {fund_name}. Using single chunk.")
        if text.strip():
            chunks.append({
                "text": text.strip(),
                "metadata": _make_metadata(
                    source_url, fund_name, doc_type, "General", publication_date
                ),
            })
        return chunks

    # Build section boundaries: (header_text, start_pos, end_pos)
    boundaries = []
    for i, match in enumerate(matches):
        section_name = match.group(0).strip().rstrip(":-").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        boundaries.append((section_name, start, end))

    # Add any content before the first header as a "Preamble" chunk
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append({
            "text": preamble,
            "metadata": _make_metadata(
                source_url, fund_name, doc_type, "Preamble", publication_date
            ),
        })

    # Create one chunk per section
    for section_name, start, end in boundaries:
        content = text[start:end].strip()
        if content:
            chunks.append({
                "text": f"{section_name}\n{content}",
                "metadata": _make_metadata(
                    source_url, fund_name, doc_type, section_name, publication_date
                ),
            })

    logger.debug(f"[CHUNKER-A] {fund_name}: {len(chunks)} section chunks created.")
    return chunks


# ── Mode B: Table-Atomic Chunking ─────────────────────────────────────────────

def chunk_tables(
    tables: list[str],
    source_url: str,
    fund_name: str,
    doc_type: str,
    publication_date: str,
) -> list[dict]:
    """
    Convert each pre-extracted Markdown table string into a single indivisible chunk.
    Tables are NEVER split. Header row + all data rows stay together.
    Returns list of {'text': str, 'metadata': dict}.
    """
    chunks = []
    for i, table_md in enumerate(tables):
        if not table_md.strip():
            continue
        # Infer section name from the first data cell of the table (heuristic)
        section_name = f"Table {i + 1}"

        chunks.append({
            "text": table_md.strip(),
            "metadata": _make_metadata(
                source_url, fund_name, doc_type, section_name, publication_date
            ),
        })

    logger.debug(f"[CHUNKER-B] {fund_name}: {len(chunks)} table chunks created.")
    return chunks


# ── Mode C: Paragraph + Sliding Window ────────────────────────────────────────

def chunk_sliding_window(
    text: str,
    source_url: str,
    fund_name: str,
    doc_type: str,
    publication_date: str,
    window_size: int = SLIDE_WINDOW_SIZE,
    overlap: int = SLIDE_WINDOW_OVERLAP,
) -> list[dict]:
    """
    Segment text into overlapping windows of `window_size` characters
    with `overlap` character overlap between adjacent chunks.
    Used for FAQ HTML pages which lack consistent section headers.
    Returns list of {'text': str, 'metadata': dict}.
    """
    chunks = []
    text = text.strip()
    if not text:
        return chunks

    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + window_size
        window = text[start:end].strip()
        if window:
            chunks.append({
                "text": window,
                "metadata": _make_metadata(
                    source_url,
                    fund_name,
                    doc_type,
                    f"Window-{chunk_index}",
                    publication_date,
                ),
            })
        chunk_index += 1
        start += window_size - overlap  # Advance with overlap

    logger.debug(f"[CHUNKER-C] {fund_name}: {len(chunks)} sliding-window chunks created.")
    return chunks


# ── Unified Dispatcher ─────────────────────────────────────────────────────────

def chunk_document(
    parsed: dict,
    source_url: str,
    fund_name: str,
    doc_type: str,
    publication_date: str,
) -> list[dict]:
    """
    Main entry point. Dispatches to the correct chunking mode based on doc_type.

    parsed = {"text": str, "tables": list[str]}

    Returns combined list of chunk dicts:
        [{"text": str, "metadata": {...}}, ...]
    """
    text   = parsed.get("text", "")
    tables = parsed.get("tables", [])
    chunks = []

    if doc_type in ("SID", "KIM"):
        # Mode A: Section-aware hierarchical splitting
        chunks += chunk_by_sections(text, source_url, fund_name, doc_type, publication_date)
        # Also add any tables found in these documents
        chunks += chunk_tables(tables, source_url, fund_name, doc_type, publication_date)

    elif doc_type == "factsheet":
        # Mode B: Table-atomic (primary) + section splitting for remaining text
        chunks += chunk_tables(tables, source_url, fund_name, doc_type, publication_date)
        chunks += chunk_by_sections(text, source_url, fund_name, doc_type, publication_date)

    elif doc_type == "faq_html":
        # Mode C: Paragraph + sliding window
        chunks += chunk_sliding_window(text, source_url, fund_name, doc_type, publication_date)

    else:
        # Unknown type: fall back to sliding window
        logger.warning(f"[CHUNKER] Unknown doc_type '{doc_type}' for {fund_name}. Using sliding window.")
        chunks += chunk_sliding_window(text, source_url, fund_name, doc_type, publication_date)

    logger.info(f"[CHUNKER] {fund_name} ({doc_type}): {len(chunks)} total chunks produced.")
    return chunks
