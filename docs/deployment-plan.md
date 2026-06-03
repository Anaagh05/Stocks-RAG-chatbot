# Deployment Plan: SBI Mutual Fund FAQ Assistant (Free Cloud Tier)

This document outlines the strategy for deploying the RAG-based chatbot using 100% free cloud services. By leveraging managed APIs (Groq, Pinecone, HuggingFace, Cohere), the backend requires virtually no local RAM, making it perfect for free Serverless and PaaS tiers.

## Architecture

- **Frontend:** Vercel (Static Hosting)
- **Backend (API + Scheduler):** Render.com (Web Service / Background Worker)
- **Vector DB:** Pinecone Serverless (Free Tier)
- **LLM:** Groq API (Free Tier)
- **Embeddings:** HuggingFace Inference API (Free Tier)
- **Re-ranking:** Cohere Rerank API (Free Tier)

---

## 1. Environment Variables Setup

You will need to generate API keys for the managed services. Create a `.env` file locally for testing, and add these exact keys to your Render environment later.

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_key
```

---

## 2. Setting up Pinecone Vector DB

1. Go to [Pinecone](https://www.pinecone.io/) and create a free Serverless index.
2. **Index Name:** `mutual-fund-faq`
3. **Dimensions:** `384` (This matches the `BAAI/bge-small-en-v1.5` embeddings)
4. **Metric:** `cosine`
5. Generate an API Key and save it.

---

## 3. Deploying the Backend on Render.com

Render offers a free tier for Web Services, which is perfect for our FastAPI backend.

1. Go to [Render](https://render.com/) and create a new **Web Service**.
2. Connect your GitHub repository: `Anaagh05/Stocks-RAG-chatbot`.
3. **Configuration:**
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:**
   - Add `GROQ_API_KEY`, `PINECONE_API_KEY`, `HF_TOKEN`, `COHERE_API_KEY`.
5. Deploy the Web Service. Render will provide a URL (e.g., `https://rag-backend.onrender.com`).

### Setting up the Scheduler

Render's free tier spins down the web service after 15 minutes of inactivity. For the ingestion scheduler to run daily at 10:00 AM, you have two options:
- **Option A (Cron Job via Render):** Render offers a "Cron Job" service. You can set the Build Command to `pip install -r requirements.txt` and the Command to `python src/ingest.py` running on a schedule (e.g., `0 10 * * *`).
- **Option B (GitHub Actions):** Create a GitHub action that runs `python src/ingest.py` daily, injecting the API keys as GitHub Secrets.

---

## 4. Deploying the Frontend on Vercel

Vercel is perfect for the static HTML/JS frontend.

1. In your project, update `static/app.js` to point to your new Render Backend URL instead of `http://localhost:8000`.
   ```javascript
   // In static/app.js
   const API_URL = "https://rag-backend.onrender.com/chat";
   ```
2. Go to [Vercel](https://vercel.com/) and create a new project.
3. Import your GitHub repository.
4. **Configuration:**
   - **Framework Preset:** `Other`
   - **Root Directory:** `static`
5. Click **Deploy**. Vercel will instantly host your frontend on a fast global CDN.

---

## 5. First-time Data Ingestion

Before users can ask questions, you must populate Pinecone with the mutual fund data.

Run the ingestion script locally from your computer (ensure your `.env` is set up):

```bash
python src/ingest.py
```

This will download the latest PDFs and HTML pages, generate embeddings via HuggingFace, and push the vectors to Pinecone. Once complete, your live deployment on Vercel/Render is ready to answer questions!
