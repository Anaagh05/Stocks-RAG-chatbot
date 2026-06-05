"""
src/query_processor.py — Query Preprocessing Pipeline (Phase 2, Section 2).

Every user query passes through three steps before retrieval:

  Step A: Advisory Intent Classifier
      Rule-based detection of speculative / advisory / comparative queries.
      If classified as advisory → short-circuit to refusal. No retrieval done.

  Step B: Fund Name Normalization
      Map informal fund references to canonical names using a lookup dict.
      e.g., "SBI bluechip" → "SBI Bluechip Fund"

  Step C: Metadata Pre-filtering
      If a specific fund name is detected, return a ChromaDB `where` filter
      to restrict vector search to embeddings for that fund only.
      Prevents cross-contamination between fund fact sets.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Step A: Advisory Intent Keywords ──────────────────────────────────────────

_ADVISORY_PATTERNS = [
    re.compile(r"\bshould\s+i\b", re.IGNORECASE),
    re.compile(r"\bshould\s+we\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(fund|scheme|sip)\s+is\s+better\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+is\s+better\b", re.IGNORECASE),
    re.compile(r"\bbetter\s+than\b", re.IGNORECASE),
    re.compile(r"\brecommend\b", re.IGNORECASE),
    re.compile(r"\bsuggestion\b", re.IGNORECASE),
    re.compile(r"\badvice\b", re.IGNORECASE),
    re.compile(r"\badvise\b", re.IGNORECASE),
    re.compile(r"\bworth\s+investing\b", re.IGNORECASE),
    re.compile(r"\bbest\s+(fund|scheme|option|choice)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+is\s+best\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+should\s+i\s+invest\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+to\s+invest\b", re.IGNORECASE),
    re.compile(r"\breturn\s+in\s+\d+\s+year", re.IGNORECASE),
    re.compile(r"\bpredict\b", re.IGNORECASE),
    re.compile(r"\bforecast\b", re.IGNORECASE),
    re.compile(r"\boutperform\b", re.IGNORECASE),
    re.compile(r"\bcompare\s+(sbi|fund|scheme)", re.IGNORECASE),
    re.compile(r"\bvs\.?\s*(sbi|hdfc|icici|axis|kotak)", re.IGNORECASE),
    re.compile(r"\bi\s+should\s+invest\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+fund\s+(to|should|can)\s+invest\b", re.IGNORECASE),
    re.compile(r"\bin\s+which\s+fund\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+fund.*\binvest\b", re.IGNORECASE),
    re.compile(r"\bwhere.*\binvest\b", re.IGNORECASE),
]

# Standard refusal message for advisory queries
ADVISORY_REFUSAL = (
    "I am a facts-only mutual fund FAQ assistant and cannot provide investment advice, "
    "recommendations, or performance predictions. "
    "For objective factual information, please refer to the official SBI Mutual Fund "
    "factsheets at https://www.sbimf.com. "
    "For investor Education, visit the AMFI Investor Education page: "
    "https://www.amfiindia.com/investor-corner"
)

# ── Step A2: Meta Intent Keywords ──────────────────────────────────────────────

_META_PATTERNS = [
    re.compile(r"\bhow\s+many\s+funds\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many.*\bfunds?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+can\s+you\s+(tell|do)\b", re.IGNORECASE),
    re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+funds\s+do\s+you\s+know\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+(the\s+)?funds\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+do\s+you\s+know\b", re.IGNORECASE),
]

META_RESPONSE = (
    "I am a facts-only mutual fund FAQ assistant. I have data for 15 SBI Mutual Fund schemes. "
    "I can answer factual questions about expense ratios, exit loads, minimum SIP amounts, "
    "benchmark indices, fund managers, and riskometer classifications. I do not provide investment advice."
)


# ── Step B: Fund Name Normalization Dictionary ─────────────────────────────────
# Maps informal/partial/misspelled fund names → canonical names in the corpus.
# Keys are lowercase for case-insensitive matching.

_FUND_NAME_ALIASES: dict[str, str] = {
    # SBI Bluechip Fund (also renamed to SBI Large Cap Fund)
    "sbi bluechip":              "SBI Bluechip Fund",
    "sbi blue chip":             "SBI Bluechip Fund",
    "sbi large cap":             "SBI Bluechip Fund",
    "sbi largecap":              "SBI Bluechip Fund",
    # SBI Small Cap Fund
    "sbi small cap":             "SBI Small Cap Fund",
    "sbi smallcap":              "SBI Small Cap Fund",
    # SBI Contra Fund
    "sbi contra":                "SBI Contra Fund",
    # SBI Magnum Midcap Fund
    "sbi midcap":                "SBI Magnum Midcap Fund",
    "sbi mid cap":               "SBI Magnum Midcap Fund",
    "sbi magnum midcap":         "SBI Magnum Midcap Fund",
    "sbi magnum mid cap":        "SBI Magnum Midcap Fund",
    # SBI Focused Equity Fund
    "sbi focused":               "SBI Focused Equity Fund",
    "sbi focused equity":        "SBI Focused Equity Fund",
    # SBI Technology Opportunities Fund
    "sbi technology":            "SBI Technology Opportunities Fund",
    "sbi tech fund":             "SBI Technology Opportunities Fund",
    "sbi tech":                  "SBI Technology Opportunities Fund",
    # SBI Healthcare Opportunities Fund
    "sbi healthcare":            "SBI Healthcare Opportunities Fund",
    "sbi health":                "SBI Healthcare Opportunities Fund",
    "sbi pharma":                "SBI Healthcare Opportunities Fund",
    # SBI Arbitrage Opportunities Fund
    "sbi arbitrage":             "SBI Arbitrage Opportunities Fund",
    # SBI Liquid Fund
    "sbi liquid":                "SBI Liquid Fund",
    # SBI Equity Hybrid Fund
    "sbi hybrid":                "SBI Equity Hybrid Fund",
    "sbi equity hybrid":         "SBI Equity Hybrid Fund",
    # SBI PSU Fund
    "sbi psu":                   "SBI PSU Fund",
    # SBI Nifty Index Fund
    "sbi nifty":                 "SBI Nifty Index Fund",
    "sbi nifty index":           "SBI Nifty Index Fund",
    "sbi index fund":            "SBI Nifty Index Fund",
    # SBI Savings Fund
    "sbi savings":               "SBI Savings Fund",
    # SBI Infrastructure Fund
    "sbi infrastructure":        "SBI Infrastructure Fund",
    "sbi infra":                 "SBI Infrastructure Fund",
    # SBI Consumption Opportunities Fund
    "sbi consumption":           "SBI Consumption Opportunities Fund",
}

# All canonical fund names (for detection in Step C)
ALL_FUND_NAMES: list[str] = list(set(_FUND_NAME_ALIASES.values()))


# ── Result Dataclass ───────────────────────────────────────────────────────────

@dataclass
class ProcessedQuery:
    """Output of the query preprocessing pipeline."""
    original_query: str
    cleaned_query: str                          # After PII scrubbing (from guardrail)
    is_advisory: bool = False                   # True → return refusal immediately
    refusal_message: str = ""                   # Populated if is_advisory is True
    detected_fund: Optional[str] = None         # Canonical fund name if detected
    chroma_filter: Optional[dict] = None        # ChromaDB `where` filter if fund detected
    pii_redacted: list[str] = field(default_factory=list)


# ── Pipeline ───────────────────────────────────────────────────────────────────

class QueryProcessor:
    """
    Runs the three-step query preprocessing pipeline:
      A → Advisory intent detection
      B → Fund name normalization
      C → Metadata pre-filter construction
    """

    def process(self, scrubbed_query: str, original_query: str = "") -> ProcessedQuery:
        """
        Process a (PII-scrubbed) user query through the pipeline.

        Args:
            scrubbed_query: Query with PII already redacted by PIIScrubber.
            original_query: Original raw query (for logging only).

        Returns:
            ProcessedQuery dataclass with all preprocessing results.
        """
        result = ProcessedQuery(
            original_query=original_query or scrubbed_query,
            cleaned_query=scrubbed_query,
        )

        # ── Step A: Advisory Intent Classification ─────────────────────────────
        if self._is_advisory(scrubbed_query):
            result.is_advisory = True
            result.refusal_message = ADVISORY_REFUSAL
            logger.info("[QUERY_PROC] Advisory query detected. Short-circuiting to refusal.")
            return result  # Early exit — no retrieval performed

        # ── Step A2: Meta Intent Classification ────────────────────────────────
        if self._is_meta(scrubbed_query):
            result.is_advisory = True  # We reuse is_advisory to trigger a short-circuit return
            result.refusal_message = META_RESPONSE
            logger.info("[QUERY_PROC] Meta query detected. Short-circuiting to meta response.")
            return result

        # ── Step B: Fund Name Normalization ───────────────────────────────────
        normalized_query, detected_fund = self._normalize_fund_name(scrubbed_query)
        result.cleaned_query = normalized_query
        result.detected_fund = detected_fund

        if detected_fund:
            logger.info(f"[QUERY_PROC] Fund detected: '{detected_fund}'")

        # ── Step C: Metadata Pre-filter Construction ───────────────────────────
        if detected_fund:
            result.chroma_filter = {"fund_name": detected_fund}
            logger.info(f"[QUERY_PROC] ChromaDB filter set: {result.chroma_filter}")

        return result

    # ── Step A internals ───────────────────────────────────────────────────────

    @staticmethod
    def _is_advisory(query: str) -> bool:
        """Return True if the query matches any advisory intent pattern."""
        return any(pattern.search(query) for pattern in _ADVISORY_PATTERNS)

    @staticmethod
    def _is_meta(query: str) -> bool:
        """Return True if the query matches any meta/greeting intent pattern."""
        return any(pattern.search(query) for pattern in _META_PATTERNS)

    # ── Step B internals ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_fund_name(query: str) -> tuple[str, Optional[str]]:
        """
        Attempt to match a known fund alias in the query (case-insensitive).
        If found, replace the alias with the canonical name in the query string.

        Returns (normalized_query, canonical_fund_name_or_None).
        """
        lower_query = query.lower()

        # Sort aliases by length descending so longer matches take priority
        # (e.g., "sbi nifty index" is matched before "sbi nifty")
        for alias in sorted(_FUND_NAME_ALIASES.keys(), key=len, reverse=True):
            if alias in lower_query:
                canonical = _FUND_NAME_ALIASES[alias]
                
                replacement_text = canonical
                if canonical == "SBI Bluechip Fund":
                    replacement_text = "SBI Large Cap Fund"
                elif canonical == "SBI Magnum Midcap Fund":
                    replacement_text = "SBI Midcap Fund"

                # Replace alias and optional trailing fund/scheme words (case-insensitive)
                pattern = re.compile(
                    re.escape(alias) + r"\s*(?:funds?|schemes?)?",
                    re.IGNORECASE
                )
                normalized = pattern.sub(replacement_text, query, count=1)
                return normalized, canonical

        # Check for exact canonical name presence (user typed the full name correctly)
        for fund_name in ALL_FUND_NAMES:
            if fund_name.lower() in lower_query:
                return query, fund_name

        return query, None
