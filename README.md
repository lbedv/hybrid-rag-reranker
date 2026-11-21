# Hybrid Search & RAG Service

[![CI](https://github.com/lbedv/hybrid-rag-reranker/actions/workflows/ci.yml/badge.svg)](https://github.com/lbedv/hybrid-rag-reranker/actions)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)
![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)

Production-grade **Hybrid Retrieval & RAG service** combining dense vector search (`pgvector`) and sparse keyword search (PostgreSQL FTS) fused via Reciprocal Rank Fusion (RRF), followed by a deterministic metadata reranker.

---

## Architecture

```text
Query
  │
  ├─► Dense Retrieval   (pgvector cosine distance <=> top-30)
  │
  └─► Sparse Retrieval  (PostgreSQL tsvector / GIN index top-30)
          │
          ▼
     Reciprocal Rank Fusion (RRF, k=60)
          │
          ▼
     Metadata Reranker (Pure function: Time decay + Category match + Severity boost)
          │
          ▼
     Top-K Ranked Results → RAG Generation with Source Citations [1]
```

### Three Search Modes
- `vector`: Dense `pgvector` cosine similarity search only.
- `hybrid`: Dense + Sparse retrieval fused with RRF.
- `hybrid+rerank`: Full two-stage pipeline incorporating metadata business signals (recency, category match, severity).

---

## Tech Stack

- **Backend**: FastAPI, Pydantic v2, SQLAlchemy 2.0 (`psycopg 3`).
- **Database**: PostgreSQL 16 with `pgvector` extension & `GENERATED ALWAYS AS ... STORED` `tsvector` GIN index.
- **LLM & Embeddings**: OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) with deterministic offline fallback.
- **Infrastructure & CI**: Docker Compose, GitHub Actions, `pytest`.

---

## Quickstart

### 1. Clone & Configure Environment
```bash
git clone https://github.com/lbedv/hybrid-rag-reranker.git
cd hybrid-rag-reranker
cp .env.example .env
```
> **Dual-Mode Embeddings:** Leave `OPENAI_API_KEY` empty in `.env` to run in deterministic offline mode (zero API costs, ideal for local dev & CI), or provide a valid OpenAI API key for live semantic embeddings.

### 2. Start PostgreSQL with `pgvector`
```bash
docker compose up -d db
```

### 3. Create Virtual Environment, Install Dependencies & Run
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation (Swagger UI) is available at: **http://localhost:8000/docs**

---

## Usage

### Health Check
Verify API availability, PostgreSQL connectivity, and `pgvector` extension status:
```bash
curl http://localhost:8000/health
# {"status":"ok","database":"connected","pgvector":"installed"}
```

### Ingest Documents (Idempotent)
Ingest the included sample dataset of 100 supply chain incidents:
```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "Content-Type: application/json" \
  -d @data/sample_incidents.json
# {"total":100,"inserted":100,"skipped":0}
```
*Re-running the command skips existing records (`"inserted": 0, "skipped": 100`) atomically via PostgreSQL `ON CONFLICT (id) DO NOTHING`.*

---

## Running Tests

Run the full test suite (unit tests, integration tests, and idempotency checks):
```bash
pytest -v
```
*All unit tests for RRF and Metadata Reranking run 100% offline in milliseconds without requiring database connections or OpenAI API keys.*