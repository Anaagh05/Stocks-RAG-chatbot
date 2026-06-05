"""
src/parser.py — Dual-Parser PDF Pipeline + Text Cleaning.

Strategy (Phase 1, Section 3):
- Primary parser:  pdfplumber  (table-aware, correct row-column structure)
- Fallback parser: pymupdf/fitz (fast, for text-heavy sections)
- Text cleaning pipeline applied after extraction:
    1. Strip repeating page headers/footers and legal disclaimers
    2. Normalize PDF ligatures (ﬁ → fi, ﬂ → fl, etc.)
    3. Collapse whitespace and remove null characters
- HTML pages are parsed with BeautifulSoup4.
"""

import logging
import re
from pathlib import Path

import pdfplumber
import fitz  # pymupdf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ── Ligature & Whitespace Normalization Map ────────────────────────────────────
_LIGATURE_MAP = str.maketrans({
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb00": "ff",   # ﬀ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2013": "-",    # en dash
    "\u2014": "--",   # em dash
    "\u00a0": " ",    # non-breaking space
    "\x00": "",       # null character
})

# ── Boilerplate Patterns to Strip ─────────────────────────────────────────────
# These patterns match recurring headers/footers and legal disclaimer blocks
# commonly found in SBI MF documents.
_BOILERPLATE_PATTERNS = [
    re.compile(r"SBI\s+Mutual\s+Fund\s*[|\-–].*?(?=\n|$)", re.IGNORECASE),
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"Mutual\s+fund\s+investments\s+are\s+subject\s+to\s+market\s+risks.*?carefully\.", re.IGNORECASE | re.DOTALL),
    re.compile(r"Regd\.\s+Office.*?(?=\n\n|\Z)", re.IGNORECASE | re.DOTALL),
    re.compile(r"CIN\s*:\s*[A-Z0-9]+", re.IGNORECASE),
    re.compile(r"AMFI\s+Registered\s+Mutual\s+Fund.*?(?=\n|$)", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # Standalone page numbers
]


# ── Text Cleaning Pipeline ─────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Apply the full text cleaning pipeline to raw extracted text:
    1. Normalize ligatures and special Unicode characters
    2. Strip boilerplate headers/footers/disclaimer blocks
    3. Collapse whitespace
    """
    # Step 1: Ligature & character normalization
    text = raw.translate(_LIGATURE_MAP)

    # Step 2: Strip boilerplate patterns
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Step 3: Collapse excessive whitespace (keep single newlines)
    text = re.sub(r"[ \t]+", " ", text)           # multiple spaces → single space
    text = re.sub(r"\n{3,}", "\n\n", text)         # 3+ newlines → double newline
    text = text.strip()

    return text


# ── PDF Parser ─────────────────────────────────────────────────────────────────

class PDFParser:
    """
    Dual-parser for PDF documents:
    - pdfplumber (primary): table-aware, used for factsheets / structured docs
    - pymupdf/fitz (fallback): fast text extraction for text-heavy sections
    """

    def parse(self, local_path: str, doc_type: str) -> dict:
        """
        Parse a PDF file and return:
        {
            "text": str,                 # cleaned full text (non-table content)
            "tables": list[str],         # list of Markdown table strings
        }
        doc_type is used to decide whether to prioritise table extraction.
        """
        path = Path(local_path)
        if not path.exists():
            logger.error(f"[PARSER] File not found: {local_path}")
            return {"text": "", "tables": []}

        text_parts = []
        tables = []

        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # ── Extract tables (Mode B: Table-Atomic) ─────────────────
                    page_tables = page.extract_tables()
                    if page_tables:
                        for raw_table in page_tables:
                            md_table = self._table_to_markdown(raw_table)
                            if md_table:
                                tables.append(md_table)

                    # ── Extract remaining text (excluding table bounding boxes) ─
                    # crop out table regions so text underneath tables is not
                    # double-counted
                    tables_objs = page.find_tables()
                    bboxes = [t.bbox for t in tables_objs]

                    def not_in_table(obj):
                        if obj.get("object_type") != "char":
                            return True
                        x0, top, x1, bottom = obj.get("x0"), obj.get("top"), obj.get("x1"), obj.get("bottom")
                        if x0 is None or top is None or x1 is None or bottom is None:
                            return True
                        for tx0, ttop, tx1, tbottom in bboxes:
                            if (x0 >= tx0 - 1 and x1 <= tx1 + 1 and
                                top >= ttop - 1 and bottom <= tbottom + 1):
                                return False
                        return True

                    filtered_page = page.filter(not_in_table)
                    page_text = filtered_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    text_parts.append(page_text)

            raw_text = "\n".join(text_parts)

        except Exception as e:
            # ── Fallback to pymupdf ────────────────────────────────────────────
            logger.warning(f"[PARSER] pdfplumber failed ({e}). Falling back to pymupdf.")
            raw_text = self._pymupdf_extract(path)
            tables = []  # pymupdf does not support table extraction

        cleaned_text = clean_text(raw_text)
        return {"text": cleaned_text, "tables": tables}

    @staticmethod
    def _table_to_markdown(raw_table: list[list]) -> str:
        """
        Convert a pdfplumber raw table (list of rows, each row a list of cells)
        into a pipe-delimited Markdown string.
        Keeps header row intact. Filters out empty tables.
        """
        if not raw_table or len(raw_table) < 2:
            return ""

        rows = []
        for i, row in enumerate(raw_table):
            # Replace None cells with empty string
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                # Insert Markdown header separator after the first row
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(rows)

    @staticmethod
    def _pymupdf_extract(path: Path) -> str:
        """Extract raw text from a PDF using pymupdf (fast fallback)."""
        doc = fitz.open(str(path))
        pages_text = [str(page.get_text("text")) for page in doc]
        doc.close()
        return "\n".join(pages_text)


# ── HTML Parser ───────────────────────────────────────────────────────────────

class HTMLParser:
    """
    Parses HTML pages downloaded from Groww fund detail pages.
    Extracts meaningful text from <p>, <li>, and key data containers.
    Strips navigation, scripts, styles, and footer boilerplate.
    """

    # CSS selectors and tags to remove (nav, scripts, ads, cookie banners)
    _REMOVE_TAGS = ["script", "style", "nav", "header", "footer", "noscript", "iframe"]
    _REMOVE_SELECTORS = [
        "[class*='cookie']",
        "[class*='banner']",
        "[class*='disclaimer']",
        "[id*='footer']",
    ]

    def parse(self, local_path: str) -> dict:
        """
        Parse an HTML file and return:
        {
            "text": str,    # cleaned extracted text
            "tables": [],   # HTML pages have no table-atomic chunks
        }
        """
        path = Path(local_path)
        if not path.exists():
            logger.error(f"[HTMLPARSER] File not found: {local_path}")
            return {"text": "", "tables": []}

        raw_html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove noise elements
        for tag in self._REMOVE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        for selector in self._REMOVE_SELECTORS:
            for el in soup.select(selector):
                el.decompose()

        # Extract text from the remaining body
        raw_text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)

        cleaned = clean_text(raw_text)
        return {"text": cleaned, "tables": []}


# ── Unified Parser Factory ─────────────────────────────────────────────────────

def parse_document(local_path: str, doc_type: str) -> dict:
    """
    Unified entry point. Routes to the correct parser based on file extension.

    Returns:
        {
            "text": str,         # Full cleaned text
            "tables": list[str], # List of Markdown table strings (PDFs only)
        }
    """
    ext = Path(local_path).suffix.lower()
    if ext == ".pdf":
        return PDFParser().parse(local_path, doc_type)
    elif ext in (".html", ".htm"):
        return HTMLParser().parse(local_path)
    else:
        logger.warning(f"[PARSER] Unsupported file type: {ext}. Attempting HTML parse.")
        return HTMLParser().parse(local_path)
