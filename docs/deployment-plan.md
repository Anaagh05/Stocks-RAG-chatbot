# Deployment Plan: SBI Mutual Fund FAQ Assistant

This document outlines the step-by-step strategy for deploying the RAG-based chatbot to a production environment. Since the project uses a local LLM (Ollama) and local Vector DB (ChromaDB), it requires a single sufficiently resourced Virtual Machine (VM) rather than complex cloud-managed AI services.

---

## 1. Server Requirements & Provisioning

**Recommended Specs:**
- **CPU:** 4+ vCPUs
- **RAM:** 16 GB minimum (8 GB for the OS/Vector DB + 8 GB dedicated to Ollama/Llama 3.1 model weights)
- **Storage:** 50 GB SSD (Model weights are ~4.7GB, ChromaDB will consume ~1-2GB over time).
- **OS:** Ubuntu 22.04 LTS (or equivalent Linux distro)

**Cloud Providers:**
- **AWS:** `t3.xlarge` or `m5.xlarge`
- **DigitalOcean:** 16GB Memory Droplet
- **GCP:** `e2-standard-4`

---

## 2. System Dependencies Installation

SSH into the provisioned server and install the core dependencies:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.14+ (or use pyenv), Git, and Nginx
sudo apt install software-properties-common python3-venv git nginx curl -y

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the required LLM model (This will take a few minutes)
ollama pull llama3.1:8b
```

---

## 3. Application Setup

Clone the repository and set up the Python environment:

```bash
# Clone the repository
git clone https://github.com/Anaagh05/Stocks-RAG-chatbot.git /var/www/rag-chatbot
cd /var/www/rag-chatbot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Initializing the Data

Before starting the server, run the ingestion process once manually to populate the vector database with the latest factsheets.

```bash
# From within the virtual environment
python src/ingest.py
```

---

## 5. Setting up Background Services (Systemd)

We need two continuously running background processes: the **FastAPI Server** and the **Ingestion Scheduler**. 

### A. API Server Service
Create `/etc/systemd/system/rag-api.service`:
```ini
[Unit]
Description=RAG FastAPI Server
After=network.target ollama.service

[Service]
User=root
WorkingDirectory=/var/www/rag-chatbot
ExecStart=/var/www/rag-chatbot/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

### B. Scheduler Service
Create `/etc/systemd/system/rag-scheduler.service`:
```ini
[Unit]
Description=RAG Daily Ingestion Scheduler
After=network.target

[Service]
User=root
WorkingDirectory=/var/www/rag-chatbot
ExecStart=/var/www/rag-chatbot/venv/bin/python scheduler.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start both services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-api rag-scheduler
sudo systemctl start rag-api rag-scheduler
```

---

## 6. Nginx Reverse Proxy & SSL

Configure Nginx to expose the FastAPI server to the web securely on port 80/443, leaving port 8000 blocked from the public internet.

Create `/etc/nginx/sites-available/rag-chatbot`:
```nginx
server {
    listen 80;
    server_name yourdomain.com; # Replace with actual domain/IP

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the configuration:
```bash
sudo ln -s /etc/nginx/sites-available/rag-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**SSL (Optional but recommended):**
Install Let's Encrypt Certbot to secure the domain:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## 7. Security & Firewall

Ensure only HTTP, HTTPS, and SSH traffic are allowed.

```bash
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH
sudo ufw enable
```

---

## 8. Maintenance & Monitoring

- **Logs:** 
  - API Logs: `journalctl -u rag-api -f`
  - Scheduler Logs: `journalctl -u rag-scheduler -f`
  - Ollama Logs: `journalctl -u ollama -f`
- **Data Cleanup:** The `scheduler.py` handles ChromaDB cleanup internally during daily fetch cycles. No manual cron jobs are required.
