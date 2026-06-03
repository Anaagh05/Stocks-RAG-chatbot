"""
src/llm.py — Prompt Engineering & LLM Orchestration (Phase 2, Section 6).

LLM: Llama 3.1 8B served locally via Ollama (free, open-weight, no API key).
  - Install: https://ollama.com
  - Pull model: `ollama pull llama3.1:8b`
  - Ollama REST API runs at http://localhost:11434 by default.

Design:
  - PromptCompiler: Assembles the structured system + user prompt from the
    top re-ranked chunks, enforcing the response contract (facts-only,
    ≤3 sentences, citation link, publication date footer).
  - OllamaLLM: Thin wrapper around the Ollama Python client / REST API.
  - LLMOrchestrator: Composes PromptCompiler + OllamaLLM. Called by the
    main RAGEngine after the confidence tier check passes.

Prompt structure:
  - System prompt:
      * Facts-only mandate
      * 3-sentence hard cap
      * Citation link requirement (from top chunk's source_url)
      * Publication date footer requirement
      * 2–3 few-shot examples illustrating the expected format
  - User turn: the actual question + retrieved context passages
"""

from __future__ import annotations

import logging
from typing import Optional

import ollama

from src.reranker import RerankedChunk

logger = logging.getLogger(__name__)

# ── Ollama Configuration ───────────────────────────────────────────────────────

OLLAMA_MODEL   = "llama3.1:8b"
OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds

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

    The user turn includes:
      - The user's question
      - Up to top_k formatted context passages (text + source URL + date)
    """

    def compile(
        self,
        query: str,
        chunks: list[RerankedChunk],
        hedge_prefix: str = "",
    ) -> tuple[str, str]:
        """
        Build the system prompt and user message for the LLM.

        Args:
            query:        The preprocessed user question.
            chunks:       Re-ranked context chunks (top 2–3 from re-ranker).
            hedge_prefix: Optional hedge string prepended to the LLM's answer
                          instruction (used in MODERATE confidence tier).

        Returns:
            (system_prompt: str, user_message: str)
        """
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
        """
        Format re-ranked chunks into a numbered context block for the prompt.
        Each passage includes its source URL and publication date for grounding.
        """
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
            lines.append("")   # blank line separator

        return "\n".join(lines)


# ── Ollama LLM Wrapper ─────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Thin wrapper around the Ollama Python client for local Llama inference.

    Requires Ollama to be running locally:
        ollama serve            # starts the server
        ollama pull llama3.1:8b # downloads the model once

    Docs: https://github.com/ollama/ollama-python
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host:  str = OLLAMA_HOST,
    ):
        self.model  = model
        self.client = ollama.Client(host=host, timeout=OLLAMA_TIMEOUT)
        logger.info(f"[LLM] Ollama client initialised | model={model} host={host}")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,   # Low temp for factual grounding
        max_tokens: int = 256,
    ) -> str:
        """
        Send a chat completion request to the local Ollama server.

        Args:
            system_prompt: The structured system prompt with constraints & examples.
            user_message:  The user turn (question + context passages).
            temperature:   Sampling temperature (0.1 = mostly deterministic).
            max_tokens:    Maximum tokens in the generated response.

        Returns:
            Generated response string (raw, before output validation).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["---", "\n\n\n"],   # stop on few-shot delimiters
                },
            )
            # NOT a plain dict — use attribute access, not key subscript.
            generated = (response.message.content or "").strip()
            logger.info(f"[LLM] Generated {len(generated)} chars.")
            return generated

        except Exception as e:
            logger.error(f"[LLM] Ollama generation failed: {e}")
            raise


# ── LLM Orchestrator ───────────────────────────────────────────────────────────

class LLMOrchestrator:
    """
    Composes PromptCompiler + OllamaLLM.
    Called by the main RAGEngine after confidence tier check passes.

    Usage:
        orchestrator = LLMOrchestrator()
        answer = orchestrator.generate_answer(
            query="What is the exit load?",
            chunks=top_reranked_chunks,
            hedge_prefix="Based on available official documents, ...",
        )
    """

    def __init__(
        self,
        model:  str = OLLAMA_MODEL,
        host:   str = OLLAMA_HOST,
    ):
        self.compiler = PromptCompiler()
        self.llm      = OllamaLLM(model=model, host=host)

    def generate_answer(
        self,
        query:        str,
        chunks:       list[RerankedChunk],
        hedge_prefix: str = "",
    ) -> str:
        """
        Compile the prompt and generate a grounded factual answer.

        Args:
            query:        Preprocessed user query.
            chunks:       Top re-ranked chunks (2–3) from CrossEncoderReranker.
            hedge_prefix: Optional hedge string for MODERATE confidence tier.

        Returns:
            Raw LLM-generated answer string (passed to OutputValidator next).
        """
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
