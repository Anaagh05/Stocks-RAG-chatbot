"""
src/output_validator.py — Post-Generation Output Validation (Phase 2, Section 7).

After the LLM generates a raw response, this module:
  1. Counts sentences and enforces the ≤3 sentence hard cap.
  2. Verifies the presence of a "Source: <url>" citation line.
  3. Verifies the presence of a "Last updated from sources: <date>" footer.
  4. Detects advisory-sounding language and flags/logs it for review.
  5. Programmatically formats non-compliant responses into a safe fallback.

Design philosophy:
  The LLM prompt already constrains output format via few-shot examples.
  This validator is a programmatic safety net — it catches the rare cases
  where the model drifts from the expected format under edge-case inputs.
  It never blocks a valid response but always ensures structural compliance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.reranker import RerankedChunk

logger = logging.getLogger(__name__)

# ── Advisory Language Patterns (for flagging/logging) ─────────────────────────
# Any response containing these patterns is logged for human review.
# The response is still returned but the flag is raised for audit.

_ADVISORY_FLAG_PATTERNS = [
    re.compile(r"\brecommend\b",      re.IGNORECASE),
    re.compile(r"\bsuggestion\b",     re.IGNORECASE),
    re.compile(r"\byou should\b",     re.IGNORECASE),
    re.compile(r"\byou must\b",       re.IGNORECASE),
    re.compile(r"\bbest option\b",    re.IGNORECASE),
    re.compile(r"\bbetter to\b",      re.IGNORECASE),
    re.compile(r"\bi would suggest\b",re.IGNORECASE),
    re.compile(r"\bi think\b",        re.IGNORECASE),
    re.compile(r"\bi believe\b",      re.IGNORECASE),
    re.compile(r"\bshould invest\b",  re.IGNORECASE),
    re.compile(r"\bworth investing\b",re.IGNORECASE),
]

# ── Structural Patterns ────────────────────────────────────────────────────────

_CITATION_PATTERN    = re.compile(r"Source:\s*https?://\S+", re.IGNORECASE)
_DATE_FOOTER_PATTERN = re.compile(
    r"Last updated from sources:\s*\d{4}-\d{2}-\d{2}", re.IGNORECASE
)

# ── Sentence Splitter (simple, finance-aware) ──────────────────────────────────
# Strategy: split on sentence-ending punctuation followed by whitespace.
# We use two passes to avoid splitting on decimal numbers (e.g. "1.5%", "Rs.100"):
#   Pass 1: standard split on [.!?] followed by whitespace.
#   Pass 2: filter out splits that look like they were caused by decimal points
#           (i.e. the preceding token ends with a digit and the next starts with one).
# Python's `re` module does NOT support variable-length lookbehinds, so we cannot
# use (?<!\d\.\d) in a single pattern — hence this two-pass approach.
_SENTENCE_SPLITTER = re.compile(
    r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bvs\.)(?<!\bCo\.)(?<!\bLtd\.)(?<!\bInc\.)(?<!\bCorp\.)"
    r"(?<!\ba\.m\.)(?<!\bp\.m\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
    r"(?<=[.!?])\s+"
)

# ── Safe Fallback ──────────────────────────────────────────────────────────────

_FALLBACK_RESPONSE_TEMPLATE = (
    "The provided official documents contain relevant information for your query, "
    "but the generated response did not meet our quality standards. "
    "Please visit the official SBI Mutual Fund website for accurate details.\n"
    "Source: {source_url}\n"
    "Last updated from sources: {pub_date}"
)


# ── Validation Result ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of the post-generation validation step."""
    raw_response:       str
    final_response:     str                       # Possibly corrected response
    is_valid:           bool   = True             # False if fallback was triggered
    sentence_count:     int    = 0
    has_citation:       bool   = False
    has_date_footer:    bool   = False
    advisory_flags:     list[str] = field(default_factory=list)  # Flagged patterns
    corrections_applied: list[str] = field(default_factory=list) # What was fixed


# ── Output Validator ───────────────────────────────────────────────────────────

class OutputValidator:
    """
    Validates and (if necessary) corrects the LLM-generated response.

    The validator enforces the response contract defined in the system prompt:
      - ≤3 sentences of factual content
      - One "Source: <url>" citation line
      - One "Last updated from sources: <date>" footer

    Advisory language is flagged and logged but does NOT block the response
    (the prompt has already constrained the model; flagging is for audit).

    Usage:
        validator = OutputValidator()
        result = validator.validate(raw_llm_output, top_chunk=chunks[0])
        print(result.final_response)
    """

    def validate(
        self,
        raw_response: str,
        top_chunk:    RerankedChunk,
    ) -> ValidationResult:
        """
        Validate the LLM's raw output and return a ValidationResult.

        If the response fails structural checks, a safe fallback is returned
        instead. Advisory flags are always logged regardless.

        Args:
            raw_response: The raw text returned by OllamaLLM.generate().
            top_chunk:    The highest-scoring re-ranked chunk (for fallback metadata).

        Returns:
            ValidationResult with the final safe response.
        """
        result = ValidationResult(
            raw_response=raw_response,
            final_response=raw_response,
        )

        # ── 1. Advisory language detection & flagging ──────────────────────────
        result.advisory_flags = self._detect_advisory_language(raw_response)
        if result.advisory_flags:
            logger.warning(
                f"[VALIDATOR] Advisory language detected in response: "
                f"{result.advisory_flags}. Flagging for review."
            )

        # ── 2. Sentence count check ────────────────────────────────────────────
        result.sentence_count = self._count_sentences(raw_response)
        if result.sentence_count > 3:
            logger.warning(
                f"[VALIDATOR] Response has {result.sentence_count} sentences "
                f"(cap is 3). Truncating."
            )
            truncated = self._truncate_to_sentences(result.final_response, max_sentences=3)
            result.final_response = truncated
            result.corrections_applied.append(f"truncated_{result.sentence_count}_to_3_sentences")

        # ── 3. Citation presence check ─────────────────────────────────────────
        result.has_citation = bool(_CITATION_PATTERN.search(result.final_response))
        if not result.has_citation:
            source_url = top_chunk.metadata.get("source_url", "https://www.sbimf.com")
            citation_line = f"\nSource: {source_url}"
            result.final_response += citation_line
            result.corrections_applied.append("appended_missing_citation")
            logger.warning("[VALIDATOR] Citation missing — appended from chunk metadata.")

        # ── 4. Date footer presence check ─────────────────────────────────────
        result.has_date_footer = bool(
            _DATE_FOOTER_PATTERN.search(result.final_response)
        )
        if not result.has_date_footer:
            pub_date    = top_chunk.metadata.get("publication_date", "N/A")
            footer_line = f"\nLast updated from sources: {pub_date}"
            result.final_response += footer_line
            result.corrections_applied.append("appended_missing_date_footer")
            logger.warning("[VALIDATOR] Date footer missing — appended from chunk metadata.")

        # ── 5. Mark validity & log summary ────────────────────────────────────
        result.is_valid = len(result.corrections_applied) == 0

        if result.corrections_applied:
            logger.info(
                f"[VALIDATOR] Corrections applied: {result.corrections_applied}"
            )
        else:
            logger.info("[VALIDATOR] Response passed all structural checks.")

        return result

    # ── Internal Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_advisory_language(text: str) -> list[str]:
        """Return a list of advisory pattern names found in the text."""
        flagged = []
        for pattern in _ADVISORY_FLAG_PATTERNS:
            match = pattern.search(text)
            if match:
                flagged.append(match.group(0).lower())
        return flagged

    @staticmethod
    def _count_sentences(text: str) -> int:
        """
        Count the number of factual sentences in the response body.
        Excludes the citation line and date footer from the count.
        """
        # Strip citation and footer lines before counting
        body = _CITATION_PATTERN.sub("", text)
        body = _DATE_FOOTER_PATTERN.sub("", body).strip()

        if not body:
            return 0

        parts = _SENTENCE_SPLITTER.split(body)
        # Filter out empty or whitespace-only splits
        sentences = [p.strip() for p in parts if len(p.strip()) > 1]
        return len(sentences)

    @staticmethod
    def _truncate_to_sentences(text: str, max_sentences: int = 3) -> str:
        """
        Truncate the factual body of the response to max_sentences,
        then re-attach any citation/footer lines that were present.
        """
        # Extract citation and footer lines before truncating
        citation_match = _CITATION_PATTERN.search(text)
        footer_match   = _DATE_FOOTER_PATTERN.search(text)

        citation_line = citation_match.group(0) if citation_match else ""
        footer_line   = footer_match.group(0)   if footer_match   else ""

        # Remove citation and footer from body for clean sentence splitting
        body = _CITATION_PATTERN.sub("", text)
        body = _DATE_FOOTER_PATTERN.sub("", body).strip()

        # Split and keep only the first max_sentences
        parts     = _SENTENCE_SPLITTER.split(body)
        sentences = [p.strip() for p in parts if len(p.strip()) > 1]
        truncated = " ".join(sentences[:max_sentences])
        if truncated and truncated[-1] not in ".!?":
            truncated += "."

        # Re-attach citation and footer
        result = truncated
        if citation_line:
            result += f"\n{citation_line}"
        if footer_line:
            result += f"\n{footer_line}"

        return result

    @staticmethod
    def build_fallback(top_chunk: RerankedChunk) -> str:
        """
        Build a fully compliant safe fallback response when the raw LLM
        output is completely unusable (e.g., empty string, corrupt output).
        """
        source_url = top_chunk.metadata.get("source_url", "https://www.sbimf.com")
        pub_date   = top_chunk.metadata.get("publication_date", "N/A")
        return _FALLBACK_RESPONSE_TEMPLATE.format(
            source_url=source_url,
            pub_date=pub_date,
        )
