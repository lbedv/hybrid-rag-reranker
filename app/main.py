from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database extensions & schemas on startup
    init_db()
    yield


app = FastAPI(
    title="Hybrid Search & RAG Service",
    description="Dense (pgvector) + Sparse (FTS) + RRF + Metadata Reranker",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)
