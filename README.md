1) README.md
# EmpaRAG

EmpaRAG is an end-to-end emotion-aware Retrieval-Augmented Generation (RAG) system built on top of the `counsel_chat_orijinal.csv` dataset and an auxiliary emotion dataset used for tone-aware LLM fine-tuning.

The project is designed not as a notebook demo, but as a production-minded AI application with:

- DataOps-oriented dataset preparation
- Qdrant-based vector retrieval
- emotion-aware prompt orchestration
- LoRA fine-tuned LLM inference
- FastAPI backend service
- Next.js frontend chat interface
- Dockerized infrastructure
- incremental indexing support
- evaluation and artifact management

---

## 1. Project Scope

This repository covers the following major layers:

### DataOps
- raw counseling dataset cleaning
- topic normalization
- quality scoring
- gold dataset generation
- emotion dataset cleaning
- supervised fine-tuning dataset construction

### RAG
- embedding generation
- Qdrant vector indexing
- retrieval
- reranking / context-aware selection foundations
- prompt construction

### MLOps
- LoRA fine-tuning
- adapter artifact generation
- local offline model loading
- evaluation metrics logging
- inference pipeline testing

### DevOps / Application
- FastAPI backend
- Next.js frontend
- Docker / Compose setup
- local development workflow

---

## 2. High-Level Architecture

```text
User Message
→ Emotion Detection
→ Tone Selection
→ Qdrant Retrieval
→ Context Construction
→ Prompt Building
→ Base LLM + LoRA Adapter
→ Cleaned Final Answer
→ API Response
→ Frontend Rendering
3. Current System Components
Backend
FastAPI
Pydantic request/response schemas
service-oriented orchestration
Vector Database
Qdrant
Dockerized local deployment
Embedding Model
sentence-transformers/all-MiniLM-L6-v2
Base LLM
local base instruct model under models/base_llm/
Fine-Tuning Strategy
LoRA-based fine-tuning
counseling context + emotion-tone examples merged into SFT training data
Frontend
Next.js
chat-style single page UI
4. Repository Structure
EmpaRAG/
├── backend/
│   └── app/
│       ├── main.py
│       ├── core/
│       │   └── config.py
│       ├── api/
│       │   ├── router.py
│       │   └── routes/
│       │       ├── health.py
│       │       └── chat.py
│       ├── schemas/
│       │   └── chat.py
│       └── services/
│           ├── emotion_service.py
│           ├── rag_service.py
│           ├── llm_service.py
│           └── chat_orchestrator.py
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── package.json
│   └── next.config.ts
│
├── configs/
│   └── rag_config.yaml
│
├── data/
│   ├── raw/
│   ├── gold/
│   ├── llm/
│   ├── processed/
│   └── reports/
│
├── models/
│   └── base_llm/
│
├── outputs/
│   └── lora_adapter/
│
├── scripts/
│   ├── build_counsel_gold.py
│   ├── prepare_emotion_tone_dataset.py
│   ├── build_sft_dataset.py
│   ├── train_lora.py
│   ├── test_local_model.py
│   ├── test_lora_inference.py
│   └── run_full_rag_lora_inference.py
│
├── src/
│   ├── common/
│   ├── pipelines/
│   └── rag/
│       ├── rag_preprocess.py
│       ├── rag_corpus_builder.py
│       ├── data_quality.py
│       ├── embedder.py
│       ├── vectordb.py
│       ├── corpus_registry.py
│       ├── incremental_indexer.py
│       ├── id_utils.py
│       ├── evaluation.py
│       ├── retriever.py
│       ├── reranker.py
│       ├── context_builder.py
│       ├── prompt_builder.py
│       └── index_corpus.py
│
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
5. Data Layer
Main Raw Datasets

Place raw files under:

data/raw/

Expected raw files:

counsel_chat_orijinal.csv
emotion-emotion_69k.csv
Generated Gold / Training Data

Produced files include:

data/gold/counsel_full.parquet
data/gold/counsel_train.parquet
data/gold/counsel_val.parquet
data/llm/emotion_tone_train.jsonl
data/llm/emotion_tone_val.jsonl
data/llm/sft_train.jsonl
data/llm/sft_val.jsonl
6. Model Artifacts
Base Model

Stored locally under:

models/base_llm/
Fine-Tuned Adapter

Generated under:

outputs/lora_adapter/

These large artifacts are intentionally excluded from Git.

7. RAG Indexing

Qdrant collection name:

counsel_rag

To build the vector index:

python -m src.rag.index_corpus

To verify Qdrant collections:

Invoke-RestMethod -Uri "http://localhost:6333/collections"
8. Local Development Workflow
8.1 Python environment
python -m venv ilhanragllm

PowerShell:

ilhanragllm\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
8.2 Backend

Run FastAPI locally:

python -m uvicorn backend.app.main:app --reload

Backend URL:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
8.3 Frontend

Inside frontend/:

npm install
npm run dev

Frontend URL:

http://localhost:3000
9. Main Pipeline Scripts
Build gold counseling dataset
python scripts/build_counsel_gold.py
Prepare emotion tone dataset
python scripts/prepare_emotion_tone_dataset.py
Build merged SFT dataset
python scripts/build_sft_dataset.py
Test base model loading
python scripts/test_local_model.py
Train LoRA adapter
python scripts/train_lora.py
Test local adapter inference
python scripts/test_lora_inference.py
Run full RAG + LoRA inference
python scripts/run_full_rag_lora_inference.py
10. API Endpoints
Health
GET /api/v1/health

Response:

{
  "status": "ok"
}
Chat
POST /api/v1/chat/message

Request:

{
  "message": "I feel overwhelmed at work and I keep thinking that I'm not good enough."
}

Response:

{
  "answer": "...",
  "predicted_emotion": "anxiety",
  "tone": "calm, reassuring, grounded",
  "retrieved_topics": ["workplace-relationships"],
  "retrieved_document_count": 4
}
11. Dockerized Infrastructure

Services:

Qdrant
FastAPI backend
Next.js frontend

Run:

docker compose up --build -d

Stop:

docker compose down
12. Why This Project Is More Than a Basic RAG Demo

This repository goes beyond a basic semantic search demo by including:

structured DataOps preprocessing
topic-aware retrieval signals
Qdrant-backed vector search
local fallback retrieval
emotion-aware tone routing
LoRA fine-tuned response generation
backend API layer
frontend interface
deployment-ready project organization

The goal is to demonstrate not only RAG capability, but also:

DataOps thinking
MLOps readiness
reproducibility
artifact separation
production-oriented architecture
13. Notes
Large model artifacts are excluded from Git.
Qdrant must be running for live vector retrieval.
If Qdrant is unavailable, the system may fall back to local semantic retrieval depending on configuration.
First model load and first embedding model load may take longer.
On Windows, Hugging Face cache may warn about symlink limitations; this does not block local development.
14. Current Status
Implemented
counseling dataset preprocessing
emotion dataset preprocessing
SFT dataset generation
Qdrant indexing
LoRA fine-tuning
adapter inference
full RAG + LoRA local inference
FastAPI backend
Next.js frontend scaffold
Next Improvements
stronger emotion detection
richer retrieval evaluation
response formatting improvements
CI/CD polish
monitoring / observability
deployment hardening

---

# 2) `Makefile`

```makefile
install:
	pip install -r requirements.txt

run-backend:
	python -m uvicorn backend.app.main:app --reload

run-frontend:
	cd frontend && npm run dev

run-full-inference:
	python scripts/run_full_rag_lora_inference.py

build-counsel-gold:
	python scripts/build_counsel_gold.py

prepare-emotion:
	python scripts/prepare_emotion_tone_dataset.py

build-sft:
	python scripts/build_sft_dataset.py

train-lora:
	python scripts/train_lora.py

test-local-model:
	python scripts/test_local_model.py

test-lora:
	python scripts/test_lora_inference.py

index-corpus:
	python -m src.rag.index_corpus

run-rag-pipeline:
	python -m src.pipelines.run_rag_pipeline

test-retrieval:
	python -m src.rag.test_retrieval

test-reranking:
	python -m src.rag.test_reranking

test-context:
	python -m src.rag.test_context_builder_v2

test-prompt:
	python -m src.rag.test_prompt_builder

docker-build:
	docker build -t emparag:latest .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

dvc-init:
	dvc init

dvc-track:
	dvc add data/raw
	dvc add data/processed
	dvc add data/reports
3) docker-compose.yml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage

  backend:
    build: .
    container_name: emparag_backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./backend:/app/backend
      - ./src:/app/src
      - ./scripts:/app/scripts
      - ./configs:/app/configs
    environment:
      - APP_ENV=local
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - HF_TOKEN=${HF_TOKEN}
    depends_on:
      - qdrant
    command: python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    container_name: emparag_frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    command: npm run dev

volumes:
  qdrant_storage:
4) .gitignore
# -------------------------
# Python
# -------------------------
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so

# -------------------------
# Virtual environments
# -------------------------
venv/
.venv/
env/
ENV/
ilhanrag/
ilhanragllm/

# -------------------------
# Jupyter
# -------------------------
.ipynb_checkpoints/

# -------------------------
# IDE / Editor
# -------------------------
.vscode/
.idea/
*.swp
*.swo

# -------------------------
# OS
# -------------------------
.DS_Store
Thumbs.db

# -------------------------
# Logs
# -------------------------
*.log

# -------------------------
# Build artifacts
# -------------------------
build/
dist/
*.egg-info/

# -------------------------
# Environment files
# -------------------------
.env

# -------------------------
# Python / HF caches
# -------------------------
.cache/
huggingface/
hf_cache/

# -------------------------
# Models / adapters / weights
# -------------------------
models/base_llm/
outputs/lora_adapter/
*.safetensors
*.bin
*.pt
*.pth

# -------------------------
# Node / Frontend
# -------------------------
frontend/node_modules/
frontend/.next/
frontend/out/
frontend/.vercel/

# -------------------------
# Data artifacts
# -------------------------
data/vector_db/
data/processed/
data/reports/
data/raw/
data/gold/
data/llm/

# -------------------------
# Optional generated artifacts
# -------------------------
artifacts/
mlruns/
mlflow.db

# -------------------------
# Temporary
# -------------------------
tmp/
temp/

# -------------------------
# Misc local files
# -------------------------
*.csv
a.py
data_prep.py
download_financebench.py

# -------------------------
# Git internals
# -------------------------
.git/
5) .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
.vscode/
.idea/
venv/
.venv/
env/
ENV/
ilhanrag/
ilhanragllm/
frontend/node_modules/
frontend/.next/
models/
outputs/
data/raw/
data/processed/
data/reports/
data/gold/
data/llm/
data/vector_db/
artifacts/
.cache/
huggingface/
*.log
tmp/
temp/