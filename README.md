---
title: MFassistant
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# SBI Mutual Fund FAQ Assistant

A facts-only RAG-powered Q&A assistant for SBI Mutual Fund schemes. Designed to run locally or on Hugging Face Spaces using the Docker SDK.

## 🛠️ Setup Steps (Local Development)

1. **Clone the repository:**
   `git clone https://github.com/Anaagh05/Stocks-RAG-chatbot.git`
2. **Set up Virtual Environment:**
   `python -m venv venv`
   `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
3. **Install Dependencies:**
   `pip install -r requirements.txt`
4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add:
   ```env
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   ```
5. **Run the Application:**
   `uvicorn api:app --host 127.0.0.1 --port 8000`
6. **Access UI:** Open `http://127.0.0.1:8000` in your browser.

## 🎯 Scope
- **Domain:** SBI Mutual Fund
- **Coverage:** This assistant is strictly scoped to 15 specific SBI Mutual Fund schemes (e.g., SBI Large Cap, SBI Contra, SBI Small Cap).
- **Knowledge Base:** Data is ingested directly from Groww Mutual Fund FAQ pages acting as proxies for official fund factsheets.

## ⚠️ Known Limits
- **No Transaction Capabilities:** The assistant cannot execute trades, check portfolios, or process KYC.
- **Facts-Only (No Advice):** Strict guardrails prevent the LLM from providing investment recommendations or forecasting future returns.
- **Static Retrieval:** It relies on cached document chunks embedded via `BAAI/bge-small-en-v1.5`. If the source document updates, the retrieval corpus must be re-ingested.
