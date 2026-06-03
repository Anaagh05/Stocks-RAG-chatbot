"""
api.py — FastAPI Backend (Phase 3, Section 2).

Exposes two endpoints:
  POST /chat    — Main Q&A endpoint; feeds the query into RAGEngine.
  GET  /health  — Liveness check (also reports Ollama connectivity).

Design decisions:
  - RAGEngine is initialised ONCE at startup via FastAPI lifespan context,
    so the heavy model-loading (BGE + cross-encoder) happens only once.
  - Blocking model inference (sentence-transformers, chromadb, ollama) is
    offloaded to a ThreadPoolExecutor so the async event loop stays responsive
    while a query is being processed.
  - StaticFiles mount serves the frontend from ./static/ so the whole stack
    (backend + UI) runs as a single `uvicorn api:app` command.
  - CORS is open for localhost during development. Tighten for production.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import ollama
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag_engine import RAGEngine

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ── Global Engine + Executor ───────────────────────────────────────────────────

_engine: Optional[RAGEngine] = None
_executor = ThreadPoolExecutor(max_workers=2)  # 2 concurrent queries max


# ── Lifespan: load models once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAGEngine (BGE + cross-encoder) once at server startup."""
    global _engine
    logger.info("Starting up — loading RAGEngine models (this may take ~30s)...")
    loop = asyncio.get_running_loop()
    _engine = await loop.run_in_executor(_executor, RAGEngine)
    logger.info("RAGEngine ready. Server is live.")
    yield
    logger.info("Shutting down.")
    _executor.shutdown(wait=False)


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SBI Mutual Fund FAQ Assistant",
    description=(
        "Facts-only RAG-powered Q&A for SBI Mutual Fund schemes. "
        "Answers sourced exclusively from official factsheets, SIDs, KIMs, and FAQ pages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten to specific origins in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static frontend (HTML/CSS/JS) from ./static/
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response Schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="The user's question about SBI Mutual Fund schemes.",
        examples=["What is the exit load of SBI Small Cap Fund?"],
    )


class ChatResponse(BaseModel):
    answer:           str   = Field(..., description="The final validated answer.")
    confidence_tier:  str   = Field("",   description="HIGH | MODERATE | LOW | ADVISORY")
    source_url:       str   = Field("",  description="Citation URL from the top retrieved chunk.")
    publication_date: str   = Field("",  description="Publication date of the source document.")
    is_advisory:      bool  = Field(False, description="True if query was detected as advisory.")
    pii_redacted:     list[str] = Field(default_factory=list, description="PII types redacted from input.")
    num_candidates:   int   = Field(0,  description="Chunks retrieved before re-ranking.")
    latency_ms:       float = Field(0,  description="Total server-side processing time in milliseconds.")


class HealthResponse(BaseModel):
    status:        str  = "ok"
    engine_ready:  bool = False
    ollama_online: bool = False
    ollama_model:  str  = ""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend index page."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness check. Reports:
      - Whether the RAGEngine (BGE + cross-encoder) is loaded.
      - Whether the Ollama server is reachable and which model is available.
    """
    engine_ready = _engine is not None

    # Probe Ollama connectivity
    ollama_online = False
    ollama_model  = ""
    try:
        client = ollama.Client(host="http://localhost:11434", timeout=3)
        loop = asyncio.get_running_loop()
        models_resp = await loop.run_in_executor(_executor, client.list)
        # .models is a list on the ListResponse object; fall back to [] safely
        model_list  = getattr(models_resp, "models", []) or []
        available   = [getattr(m, "model", "") for m in model_list if getattr(m, "model", "")]
        ollama_online = True
        # Prefer the required llama3.1 model; report whatever else is loaded
        for name in available:
            if "llama3.1" in name or "llama3" in name:
                ollama_model = name
                break
        if not ollama_model and available:
            ollama_model = available[0]
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (engine_ready and ollama_online) else "degraded",
        engine_ready=engine_ready,
        ollama_online=ollama_online,
        ollama_model=ollama_model,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main Q&A endpoint.

    Accepts a user query and returns a facts-only answer sourced from
    official SBI Mutual Fund documents. Advisory questions, out-of-scope
    queries, and PII are handled automatically before any LLM call.
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine is still initialising. Please retry in a few seconds.",
        )

    t0 = time.perf_counter()

    # Run blocking RAGEngine.answer() in a thread so the event loop is free
    loop = asyncio.get_running_loop()
    try:
        rag_response = await loop.run_in_executor(
            _executor, _engine.answer, request.query
        )
    except Exception as e:
        logger.error(f"[API] Unhandled error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[API] query={request.query[:60]!r} | "
        f"tier={rag_response.confidence_tier} | "
        f"latency={latency_ms:.0f}ms"
    )

    return ChatResponse(
        answer=rag_response.final_answer,
        confidence_tier=rag_response.confidence_tier,
        source_url=rag_response.top_source_url,
        publication_date=rag_response.top_pub_date,
        is_advisory=rag_response.is_advisory,
        pii_redacted=rag_response.pii_redacted,
        num_candidates=rag_response.num_candidates,
        latency_ms=round(latency_ms, 1),
    )


# ── Dev Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,        # reload=True breaks the ThreadPoolExecutor lifecycle
        log_level="info",
    )
