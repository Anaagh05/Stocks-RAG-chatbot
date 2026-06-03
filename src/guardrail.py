"""
src/guardrail.py — Safety Guardrail & PII Scrubber (Phase 2, Section 1).

Scrubs and redacts sensitive Personally Identifiable Information (PII)
from user queries BEFORE any retrieval or LLM processing occurs.

PII types handled:
  - PAN card numbers       (e.g., ABCDE1234F)
  - Aadhaar numbers        (e.g., 1234 5678 9012)
  - OTPs                   (4–8 digit standalone codes)
  - Indian mobile numbers  (10 digits, optionally prefixed with +91 / 91)
  - Email addresses
  - Bank/account numbers   (12–18 digit numeric strings)
"""

import logging
import re

logger = logging.getLogger(__name__)


# ── Compiled Regex Patterns ────────────────────────────────────────────────────
# All patterns are compiled once at import time for speed.

_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "[REDACTED_PAN]",
        re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    ),
    (
        "[REDACTED_AADHAAR]",
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b"),
    ),
    (
        "[REDACTED_PHONE]",
        re.compile(r"(?:\+91|91)?[\s\-]?[6-9]\d{9}\b"),
    ),
    (
        "[REDACTED_EMAIL]",
        re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    ),
    (
        "[REDACTED_OTP]",
        # Match standalone 4–8 digit numbers (context: "OTP is 123456")
        re.compile(r"(?i)(?:otp|one[\s-]time[\s-]password)\s*(?:is|:)?\s*(\d{4,8})"),
    ),
    (
        "[REDACTED_ACCOUNT]",
        # Bank account / IFSC-adjacent long numeric strings (12–18 digits)
        re.compile(r"\b\d{12,18}\b"),
    ),
]


# ── Scrubber ───────────────────────────────────────────────────────────────────

class PIIScrubber:
    """
    Applies all PII redaction patterns to a query string.
    Logs each redaction made for audit/monitoring purposes.
    """

    def scrub(self, query: str) -> tuple[str, list[str]]:
        """
        Scrub PII from the input query.

        Returns:
            (scrubbed_query: str, redactions_made: list[str])
            redactions_made is a list of labels for PII types that were removed.
        """
        cleaned = query
        redactions_made: list[str] = []

        for label, pattern in _PATTERNS:
            new_text, count = pattern.subn(label, cleaned)
            if count > 0:
                cleaned = new_text
                redactions_made.append(label)
                logger.warning(
                    f"[GUARDRAIL] PII redacted — {label} ({count} instance(s))"
                )

        if redactions_made:
            logger.info(f"[GUARDRAIL] Original: {query!r}")
            logger.info(f"[GUARDRAIL] Cleaned:  {cleaned!r}")

        return cleaned, redactions_made

    def contains_pii(self, query: str) -> bool:
        """Quick check: returns True if any PII pattern matches."""
        return any(pattern.search(query) for _, pattern in _PATTERNS)
