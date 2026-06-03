"""
src/llm.py — Prompt Engineering & LLM Orchestration via Groq.

LLM: Llama 3.1 8B served via Groq API.
"""

from __future__ import annotations

import logging
from typing import Optional

from groq import Groq

from src.reranker import RerankedChunk
from config import GROQ_API_KEY

logger = logging.getLogger(__name__)

# ── Groq Configuration ─────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.1-8b-instant"

# ── System Prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a facts-only Mutual Fund FAQ assistant for SBI Mutual Fund schemes.

STRICT RULES — follow every rule without exception:
1. Answer ONLY using facts from the provided "Context Passages" below. Do NOT use prior knowledge.
2. Your answer MUST be a maximum of 3 sentences. Shorter is preferred.
3. You MUST end every answer with a citation in this exact format:
   Source: <source_url>
4. You MUST end every answer (after the citation) with a date footer in this exact format:
   Last updated from sources: <publication_date>
5. NEVER provide investment advice, recommendations, predictions, or comparisons.
6. NEVER say "I think", "I believe", "you should", "I recommend", or similar advisory phrases.
7. If the context passages do not contain enough information to answer, say:
   "The provided documents do not contain specific information on this topic. Please visit https://www.sbimf.com for authoritative details."
   Then add the citation and date footer from whichever passage is most relevant.

FEW-SHOT EXAMPLES (exact format expected):

---
Q: What is the exit load of SBI Small Cap Fund?
Context: [passage about SBI Small Cap Fund exit load]
A: SBI Small Cap Fund charges an exit load of 1% if units are redeemed within 1 year of allotment. No exit load is applicable after 1 year. This applies to both regular and direct plan variants.
Source: https://groww.in/mutual-funds/sbi-small-cap-fund-direct-growth
Last updated from sources: 2026-05-31
---

---
Q: Who manages SBI Bluechip Fund?
Context: [passage about SBI Bluechip Fund fund manager]
A: SBI Bluechip Fund is managed by Sohini Andani and Mohit Jain. Sohini Andani has been managing the fund since 2010 and brings extensive expertise in large-cap equities.
Source: https://groww.in/mutual-funds/sbi-bluechip-fund-direct-growth
Last updated from sources: 2026-05-31
---

---
Q: What is the benchmark index for SBI Contra Fund?
Context: [passage about SBI Contra Fund benchmark]
A: SBI Contra Fund benchmarks its performance against the S&P BSE 500 TRI (Total Return Index). This index represents the broader Indian equity market across market capitalizations.
Source: https://groww.in/mutual-funds/sbi-contra-fund-direct-growth
Last updated from sources: 2026-05-31
---
"""

# ── Prompt Compiler ────────────────────────────────────────────────────────────

class PromptCompiler:
    """
    Assembles the full prompt (system + user turn) from re-ranked chunks.
    """

    def compile(
        self,
        query: str,
        chunks: list[RerankedChunk],
        hedge_prefix: str = "",
    ) -> tuple[str, str]:
        context_block = self._format_context(chunks)

        user_message = (
            f"Context Passages:\n{context_block}\n\n"
            f"Question: {query}\n\n"
        )

        if hedge_prefix:
            user_message += (
                f"Important: Begin your answer with this exact phrase:\n"
                f"\"{hedge_prefix}\"\n\n"
            )

        user_message += "Answer (follow ALL rules in the system prompt exactly):"

        return _SYSTEM_PROMPT, user_message

    @staticmethod
    def _format_context(chunks: list[RerankedChunk]) -> str:
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source_url  = chunk.metadata.get("source_url", "N/A")
            pub_date    = chunk.metadata.get("publication_date", "N/A")
            fund_name   = chunk.metadata.get("fund_name", "Unknown Fund")
            section     = chunk.metadata.get("section_name", "")

            header = f"[Passage {i}] {fund_name}"
            if section:
                header += f" — {section}"
            header += f" | Source: {source_url} | Date: {pub_date}"

            lines.append(header)
            lines.append(chunk.text.strip())
            lines.append("")

        return "\n".join(lines)


# ── Groq LLM Wrapper ─────────────────────────────────────────────────────────

class GroqLLM:
    """
    Thin wrapper around the Groq Python client.
    """

    def __init__(self, model: str = GROQ_MODEL):
        self.model  = model
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"[LLM] Groq client initialised | model={model}")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["---", "\n\n\n"],
            )
            generated = response.choices[0].message.content or ""
            generated = generated.strip()
            logger.info(f"[LLM] Generated {len(generated)} chars via Groq.")
            return generated

        except Exception as e:
            logger.error(f"[LLM] Groq generation failed: {e}")
            raise


# ── LLM Orchestrator ───────────────────────────────────────────────────────────

class LLMOrchestrator:
    def __init__(self, model: str = GROQ_MODEL):
        self.compiler = PromptCompiler()
        self.llm      = GroqLLM(model=model)

    def generate_answer(
        self,
        query:        str,
        chunks:       list[RerankedChunk],
        hedge_prefix: str = "",
    ) -> str:
        system_prompt, user_message = self.compiler.compile(
            query=query,
            chunks=chunks,
            hedge_prefix=hedge_prefix,
        )

        logger.debug(f"[LLM_ORCH] User message length: {len(user_message)} chars")

        raw_answer = self.llm.generate(
            system_prompt=system_prompt,
            user_message=user_message,
        )

        logger.info(f"[LLM_ORCH] Raw answer: {raw_answer[:120]!r}...")
        return raw_answer
