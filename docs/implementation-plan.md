# Implementation Plan: Mutual Fund FAQ Assistant

This document outlines the step-by-step roadmap for implementing the facts-only RAG-based Mutual Fund FAQ Assistant.

---

## Phase 1: Setup & Data Ingestion (Days 1–2)

### 1. Project Initialization
- Create a virtual environment and install core dependencies:
  - `langchain` / `llama-index` for RAG orchestration
  - `chromadb` for vector storage with metadata filtering
  - `rank-bm25` for sparse retrieval (BM25)
  - `ollama` Python client for local LLM inference (free, no API key required)
  - `pdfplumber` for table-aware PDF extraction (primary parser)
  - `pymupdf` (`fitz`) as a high-speed fallback text parser
  - `beautifulsoup4` + `httpx` for async HTML page scraping
  - `sentence-transformers` for local embedding generation (BGE model)
  - `apscheduler` for the daily ingestion cron scheduler

### 2. Document Fetching Strategy: Incremental + ETag Caching
- **Do not naively re-download** all 15 URLs every day. Financial documents are updated monthly; daily re-ingestion of unchanged PDFs wastes compute and risks polluting the vector DB with duplicate embeddings.
- **Strategy**: On every scheduled run, issue an HTTP `HEAD` request to each URL and compare the `ETag` or `Last-Modified` header value against the cached value stored in `/data/fetch_cache.json`.
  - If unchanged: skip download, log as `[CACHE HIT]`.
  - If changed or new: download the document, store the new `ETag` / `Last-Modified` value, delete old embeddings for that fund from ChromaDB, and re-ingest.
- **Why**: Guarantees data freshness while minimizing redundant daily re-processing.

### 3. PDF Parsing Strategy: Dual-Parser Pipeline
- Use `pdfplumber` as the **primary parser** for its table detection API (`.extract_tables()`), which correctly identifies row-column structure in multi-column financial tables.
- Use `pymupdf` (`fitz`) as a **fallback parser** for text-heavy sections (e.g., risk disclosures, scheme objectives) where `pdfplumber` is slower.
- After extraction, run a **text cleaning pipeline**:
  - Strip repeating page headers/footers (AMC logo text, page numbers, legal disclaimer blocks) using regex pattern matching.
  - Normalize PDF ligatures (`ﬁ` → `fi`, `ﬂ` → `fl`) that corrupt tokenization.
  - Collapse excessive whitespace and remove null characters introduced by PDF encoding.
- **Why**: Raw PDF extraction from financial documents is noisy. Repeated boilerplate content (legal disclaimers, AMC headers) adds noise to embeddings and dilutes retrieval quality.

### 4. Hybrid Chunking Strategy
A single chunking method is insufficient for this corpus. We use a **three-mode hybrid chunking approach** based on document type:

#### Mode A: Section-Aware Hierarchical Splitting (for SIDs & KIMs)
- Detect section headers using regex patterns matching common SID/KIM section titles (e.g., `"Exit Load"`, `"Expense Ratio"`, `"Investment Objective"`, `"Benchmark"`, `"Fund Manager"`).
- Split the document at each detected header boundary, producing **self-contained, topic-focused chunks** (e.g., all exit load details in one chunk).
- Each chunk includes the section name as metadata to improve retrieval specificity.
- **Why**: SIDs and KIMs are section-structured. Splitting by section prevents facts from different scheme properties (e.g., expense ratio and benchmark index) from being mixed in a single retrieved chunk.

#### Mode B: Table-Atomic Chunking (for Factsheets)
- Use `pdfplumber`'s `.extract_tables()` to isolate every table in the document.
- Store each table as a **single, indivisible chunk** — the header row and all data rows are kept together. Tables are never split across two chunks.
- Represent table content as a pipe-separated Markdown-style string for clean embedding.
- **Why**: Factsheets store the most frequently queried data (tiered exit loads, AUM, fund manager names, expense ratios) in tables. Splitting mid-table corrupts the fact structure and leads to hallucinated or merged incorrect values.

#### Mode C: Paragraph + Sliding Window (for FAQ HTML Pages)
- Segment HTML pages at `<p>` and `<li>` tag boundaries.
- Apply a sliding window of **400 characters with 100 characters overlap** between adjacent window chunks.
- **Why**: FAQ pages are prose-structured and lack reliable section headers. Sliding window overlap ensures that critical facts spanning two paragraphs are not lost at a chunk boundary.

#### Universal Metadata Tags per Chunk
Every chunk, regardless of its mode, is stored with the following metadata:
```json
{
  "source_url": "https://groww.in/mutual-funds/...",
  "fund_name": "SBI Small Cap Fund",
  "doc_type": "factsheet" | "SID" | "KIM" | "faq_html",
  "section_name": "Exit Load",
  "publication_date": "2026-05-31",
  "ingestion_date": "2026-06-02"
}
```

### 5. Embedding Strategy: Domain-Optimized Local Model
- **Model**: `BAAI/bge-small-en-v1.5` (via `sentence-transformers`)
- **Why over `all-MiniLM-L6-v2`**: The BGE (BAAI General Embedding) model family is state-of-the-art for English dense retrieval and is specifically tuned for passage Q&A tasks. It significantly outperforms `all-MiniLM` on financial domain benchmarks with comparable inference speed.
- **Instruction-prefixed query embedding**: BGE models benefit from a task instruction prepended to the query at search time:
  ```
  "Represent this question for searching relevant mutual fund passages: {user_query}"
  ```
  Document chunks are embedded without this prefix. This asymmetric embedding approach improves retrieval relevance.
- **Batch Processing**: Embed all chunks in batches of 64 during ingestion for memory efficiency. Do not embed one chunk at a time.

---

## Phase 2: RAG Engine Development (Days 3–4)

### 1. Safety Guardrail & PII Scrubber
- Build a utility to scrub and redact sensitive information (PAN, Aadhaar, OTPs, phone numbers, email addresses) from user input before processing.
- Use compiled regex patterns for speed. PAN format: `[A-Z]{5}[0-9]{4}[A-Z]`; Aadhaar: `\d{4}\s\d{4}\s\d{4}`.

### 2. Query Preprocessing Pipeline
Before hitting the vector database, every user query passes through a preprocessing pipeline:

#### Step A: Advisory Intent Classifier
- Run a lightweight rule-based or LLM-based classifier to detect advisory, speculative, or comparison queries (keywords: "should I", "better than", "recommend", "returns in X years", "which is best").
- If classified as advisory, **short-circuit immediately** to the Refusal Handler. No retrieval is performed.

#### Step B: Fund Name Normalization
- Detect and normalize informal fund references to canonical names using a lookup dictionary:
  - `"SBI bluechip"` → `"SBI Bluechip Fund"`
  - `"SBI small cap"` → `"SBI Small Cap Fund"`
- **Why**: Users rarely type exact scheme names. Normalization ensures the correct metadata pre-filter is applied.

#### Step C: Metadata Pre-filtering
- If a specific fund name is detected in the query, **pre-filter the vector search** to only search embeddings tagged with that `fund_name` in ChromaDB's metadata filter.
- **Why**: Prevents the retriever from confusing SBI Large Cap's expense ratio with SBI Small Cap's expense ratio, which are common cross-contamination errors.

### 3. Hybrid Retrieval Strategy: Dense + BM25 via Reciprocal Rank Fusion (RRF)
A single dense vector search is insufficient for financial Q&A:

#### Dense Retriever (Semantic)
- Queries ChromaDB using BGE embeddings with cosine similarity.
- Retrieves top-10 candidates.
- **Strength**: Understands semantic equivalences ("What does it cost to exit early?" → "Exit Load").

#### Sparse Retriever (BM25)
- Uses `rank-bm25` to index all chunk texts and retrieve by exact term frequency.
- Retrieves top-10 candidates.
- **Strength**: Exact match on critical financial identifiers: specific fund names, numerical values, regulatory terms (`"ELSS"`, `"Riskometer"`, `"NAV"`), and scheme-specific jargon that dense models may miss.

#### Reciprocal Rank Fusion (RRF) Merging
- Combine the two ranked lists into a single merged ranked list using RRF scoring:
  ```
  RRF_score(chunk) = Σ 1 / (k + rank_i)   where k=60
  ```
- Select the **top-5 chunks** from the fused ranked list.
- **Why**: Neither dense-only nor sparse-only works well alone on financial documents. RRF fusion consistently outperforms both on mixed factual/semantic queries without requiring model fine-tuning.

### 4. Cross-Encoder Re-ranking
- After hybrid retrieval produces the top-5 candidates, pass each (query, chunk) pair through a cross-encoder re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Reorder the 5 candidates by the cross-encoder relevance score. Only the top-2 or top-3 are forwarded to the Prompt Compiler.
- **Why**: The bi-encoder retrieval (both dense and BM25) optimizes for **recall** — getting relevant chunks into the candidate set. The cross-encoder optimizes for **precision** — putting the single most relevant chunk at position 1. Sending the LLM the highest-precision context minimizes hallucination.

### 5. Dynamic Tiered Similarity Thresholding
- Replace a fixed similarity cutoff with a **three-tier confidence model** based on the top cross-encoder score:

  | Score Range | Confidence Tier | Action |
  |---|---|---|
  | ≥ 0.80 | **High Confidence** | Generate answer normally via LLM |
  | 0.50 – 0.79 | **Moderate Confidence** | Generate answer but prepend: *"Based on available official documents..."* |
  | < 0.50 | **Low Confidence / No Match** | Bypass LLM. Return standardized refusal with link to official SBI MF site |

- **Why**: A fixed threshold (e.g., 0.5) either rejects too many valid but paraphrased queries or lets too many irrelevant queries through. The tiered model allows graceful degradation.

### 6. Prompt Engineering & LLM Orchestration
- **LLM**: **Llama 3.1 8B** served locally via **Ollama** — completely free, open-weight, no API key required, and runs on consumer hardware (8 GB+ RAM).
  - Install Ollama from [https://ollama.com](https://ollama.com) and pull the model: `ollama pull llama3.1:8b`
  - Invoke via the `ollama` Python library or its REST API at `http://localhost:11434`.
- **Why Llama 3.1 8B over paid models**: Meta's Llama 3.1 8B is instruction-tuned, follows structured system prompts reliably, and produces high-quality factual responses on constrained-format tasks. It requires no API billing and adds zero per-query cost.
- Structure a robust system prompt with the following constraints:
  - Mandate facts-only generation.
  - Strict maximum of 3 sentences.
  - Embed exactly one citation link from the top chunk's `source_url` metadata.
  - Include the `"Last updated from sources: <publication_date>"` footer using the metadata date.
  - Include 2–3 few-shot examples demonstrating the exact expected output format.

### 7. Post-Generation Output Validation
- Implement parser checks to count sentences and verify the presence of the footer and citation link.
- Programmatically format non-compliant responses back to a safe fallback response.
- Flag and log any response containing advisory-sounding words ("recommend", "suggest", "best", "should") for review.

---

## Phase 3: UI & Integration (Days 5–6)

### 1. Minimal UI Interface
- Develop a lightweight front-end with:
  - Chat feed container.
  - Welcome banner with three clickable quick-start questions (e.g., *"What is the exit load of SBI Small Cap Fund?"*).
  - Sticky disclaimer bar: *"Facts-only. No investment advice."*

### 2. API Endpoint / Backend Wrapper
- Build a lightweight `FastAPI` server to expose `/chat` endpoint connecting the UI to the RAG Engine.

---

## Phase 4: Testing & Validation (Day 7)

### 1. Unit & Edge-Case Testing
- Test retrieval accuracy against factual queries.
- Test prompt injection resistance and advisory query refusal.
- Verify automatic PII redaction.
- Validate hybrid retrieval fusion scores vs dense-only baseline.
- Verify cross-encoder re-ranking places the highest-precision chunk at position 1.
