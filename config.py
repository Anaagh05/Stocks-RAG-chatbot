"""
config.py — Central configuration for the Mutual Fund FAQ Assistant.
All tunable constants are defined here. Import from this module throughout the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data Directories ──────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
CORPUS_DIR      = DATA_DIR / "corpus"
FETCH_CACHE_PATH = DATA_DIR / "fetch_cache.json"

# Ensure directories exist
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY", "")

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "mutual-fund-faq"

# ── Embedding Model ───────────────────────────────────────────────────────────
# BAAI/bge-small-en-v1.5 — SOTA free local retrieval model
EMBEDDING_MODEL     = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 64
# Asymmetric instruction prefix applied ONLY to query embeddings at search time
QUERY_INSTRUCTION   = "Represent this question for searching relevant mutual fund passages: "

# ── Chunking ──────────────────────────────────────────────────────────────────
# Mode C — Paragraph sliding window (FAQ HTML pages)
SLIDE_WINDOW_SIZE    = 400   # characters
SLIDE_WINDOW_OVERLAP = 100   # characters

# ── SBI Mutual Fund Corpus URLs ───────────────────────────────────────────────
# 15 official Groww pages for SBI Mutual Fund schemes.
# Each entry: (url, fund_name, doc_type)
# doc_type: "factsheet" | "SID" | "KIM" | "faq_html"
CORPUS_SOURCES = [
    {
        "url": "https://groww.in/mutual-funds/sbi-bluechip-fund-direct-growth",
        "fund_name": "SBI Bluechip Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-small-cap-fund-direct-growth",
        "fund_name": "SBI Small Cap Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-contra-fund-direct-growth",
        "fund_name": "SBI Contra Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-magnum-midcap-fund-direct-growth",
        "fund_name": "SBI Magnum Midcap Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-focused-equity-fund-direct-growth",
        "fund_name": "SBI Focused Equity Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-technology-opportunities-fund-direct-growth",
        "fund_name": "SBI Technology Opportunities Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-healthcare-opportunities-fund-direct-growth",
        "fund_name": "SBI Healthcare Opportunities Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-arbitrage-opportunities-fund-direct-growth",
        "fund_name": "SBI Arbitrage Opportunities Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-liquid-fund-direct-growth",
        "fund_name": "SBI Liquid Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-equity-hybrid-fund-direct-growth",
        "fund_name": "SBI Equity Hybrid Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-psu-fund-direct-growth",
        "fund_name": "SBI PSU Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-nifty-index-fund-direct-growth",
        "fund_name": "SBI Nifty Index Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-savings-fund-direct-growth",
        "fund_name": "SBI Savings Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-infrastructure-fund-direct-growth",
        "fund_name": "SBI Infrastructure Fund",
        "doc_type": "faq_html",
    },
    {
        "url": "https://groww.in/mutual-funds/sbi-consumption-opportunities-fund-direct-growth",
        "fund_name": "SBI Consumption Opportunities Fund",
        "doc_type": "faq_html",
    },
]

# ── Scheduler ─────────────────────────────────────────────────────────────────
# Daily trigger time for the ingestion scheduler (24-hour format, local time)
SCHEDULER_HOUR   = 10   # 10:00 AM
SCHEDULER_MINUTE = 0
