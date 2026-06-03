"""
src/confidence.py — Dynamic Tiered Similarity Thresholding (Phase 2, Section 5).

Converts the top cross-encoder score from the re-ranker into one of three
confidence tiers that govern the LLM's response strategy.

Three-tier model:
  ┌─────────────────┬──────────────────┬───────────────────────────────────────────┐
  │ Score Range     │ Confidence Tier  │ Action                                    │
  ├─────────────────┼──────────────────┼───────────────────────────────────────────┤
  │ ≥ 0.80          │ HIGH             │ Generate answer normally via LLM           │
  │ 0.50 – 0.79     │ MODERATE         │ Generate answer + prepend hedge prefix     │
  │ < 0.50          │ LOW              │ Bypass LLM. Return standardised refusal    │
  └─────────────────┴──────────────────┴───────────────────────────────────────────┘

Why tiered vs. fixed threshold?
  A fixed cutoff (e.g., 0.5) either rejects too many valid paraphrased queries
  or lets irrelevant queries through. The tiered model allows graceful degradation:
  a moderately-matched query still gets an answer, but with an epistemic hedge so
  users know to verify. Only clearly off-topic queries are hard-refused.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)

# ── Tier Thresholds ────────────────────────────────────────────────────────────

HIGH_CONFIDENCE_THRESHOLD     = 0.80
MODERATE_CONFIDENCE_THRESHOLD = 0.50


# ── Confidence Tier Enum ───────────────────────────────────────────────────────

class ConfidenceTier(str, Enum):
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"


# ── Refusal / Hedge Strings ────────────────────────────────────────────────────

# Prepended to LLM output in MODERATE confidence cases.
MODERATE_HEDGE_PREFIX = (
    "Based on available official documents, the following information may be relevant, "
    "but please verify directly with SBI Mutual Fund for the most current details: "
)

# Returned directly (LLM bypassed) when confidence is LOW.
LOW_CONFIDENCE_REFUSAL = (
    "I could not find sufficiently relevant information in the official SBI Mutual Fund "
    "documents to answer your question accurately. "
    "For authoritative and up-to-date information, please visit: "
    "https://www.sbimf.com or call the SBI MF investor helpline."
)


# ── Thresholder ────────────────────────────────────────────────────────────────

class ConfidenceThresholder:
    """
    Classifies the top cross-encoder score into a ConfidenceTier and
    provides the appropriate hedge prefix or refusal message.

    Usage:
        thresholder = ConfidenceThresholder()
        tier = thresholder.classify(top_score=0.73)
        # → ConfidenceTier.MODERATE
    """

    def classify(self, top_score: float) -> ConfidenceTier:
        """
        Classify a cross-encoder score into a confidence tier.

        Args:
            top_score: The highest cross-encoder relevance score among
                       re-ranked candidates (float, unbounded but typically
                       in range −10 to +10 for ms-marco models — scaled to
                       0–1 by sigmoid in practice). The thresholds here are
                       calibrated for raw logit outputs of ms-marco-MiniLM-L-6-v2,
                       which maps to interpretable ranges:
                         > 0   → passage is at least somewhat relevant
                         > 0.8 → strong relevance signal
                         < 0   → passage is off-topic (map to LOW via 0.50 check)

        Returns:
            ConfidenceTier enum value.
        """
        if top_score >= HIGH_CONFIDENCE_THRESHOLD:
            tier = ConfidenceTier.HIGH
        elif top_score >= MODERATE_CONFIDENCE_THRESHOLD:
            tier = ConfidenceTier.MODERATE
        else:
            tier = ConfidenceTier.LOW

        logger.info(
            f"[CONFIDENCE] Score: {top_score:.4f} → Tier: {tier.value}"
        )
        return tier

    @staticmethod
    def get_hedge_prefix(tier: ConfidenceTier) -> str:
        """
        Return the hedge prefix string to prepend to LLM output.
        Empty string for HIGH confidence (no hedge needed).
        """
        if tier == ConfidenceTier.MODERATE:
            return MODERATE_HEDGE_PREFIX
        return ""

    @staticmethod
    def get_refusal_message() -> str:
        """Return the standardised refusal message for LOW confidence cases."""
        return LOW_CONFIDENCE_REFUSAL

    @staticmethod
    def should_call_llm(tier: ConfidenceTier) -> bool:
        """Return True if the LLM should be invoked (HIGH or MODERATE tiers)."""
        return tier in (ConfidenceTier.HIGH, ConfidenceTier.MODERATE)
