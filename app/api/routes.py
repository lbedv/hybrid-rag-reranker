from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from uuid import UUID

from app.api.deps import get_db
from app.db.models import Incident
from app.schemas.incident import IncidentCreate, IngestResponse
from app.services.embedding import embedding_service

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check validating service availability, database connection, and pgvector extension."""
    try:
        # Check database connectivity
        db.execute(text("SELECT 1"))

        # Check pgvector extension
        vector_ext = db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if not vector_ext:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="pgvector extension is not installed in database",
            )

        return {
            "status": "ok",
            "database": "connected",
            "pgvector": "installed",
        }
    except HTTPException:
        # Re-raise explicit HTTP exceptions without wrapping
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {exc}",
        ) from exc


@router.post(
    "/api/v1/documents/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_documents(
    documents: list[IncidentCreate],
    db: Session = Depends(get_db),
):
    """
    Idempotent document ingestion endpoint.
    Inserts incident documents with pgvector embeddings and FTS tsvector.
    Handles intra-batch duplicates and skips existing IDs (ON CONFLICT DO NOTHING).
    """
    if not documents:
        return IngestResponse(total=0, inserted=0, skipped=0)

    # 1. Deduplicate within incoming batch to prevent PostgreSQL CardinalityViolation
    unique_documents: list[IncidentCreate] = []
    seen_ids: set[UUID] = set()
    for doc in documents:
        if doc.id in seen_ids:
            continue
        seen_ids.add(doc.id)
        unique_documents.append(doc)

    # 2. Compute missing embeddings in batch with external service error handling
    missing_indices = [i for i, d in enumerate(unique_documents) if d.embedding is None]
    if missing_indices:
        texts_to_embed = [
            f"{unique_documents[i].title}\n{unique_documents[i].content}"
            for i in missing_indices
        ]
        try:
            computed_embeddings = embedding_service.get_embeddings(texts_to_embed)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Embedding generation failed: {exc}",
            ) from exc

        for idx, emb in zip(missing_indices, computed_embeddings):
            unique_documents[idx].embedding = emb

    # 3. Prepare records for bulk insert
    records = [
        {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "category": doc.category,
            "region": doc.region,
            "severity_level": doc.severity_level,
            "occurred_at": doc.occurred_at,
            "embedding": doc.embedding,
        }
        for doc in unique_documents
    ]

    # 4. PostgreSQL Idempotent Bulk Insert (ON CONFLICT (id) DO NOTHING RETURNING id)
    insert_stmt = (
        pg_insert(Incident)
        .values(records)
        .on_conflict_do_nothing(index_elements=["id"])
        .returning(Incident.id)
    )
    try:
        result = db.execute(insert_stmt)
        inserted_ids = result.scalars().all()
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database insertion failed: {exc}",
        ) from exc

    inserted_count = len(inserted_ids)
    skipped_count = len(documents) - inserted_count

    return IngestResponse(
        total=len(documents),
        inserted=inserted_count,
        skipped=skipped_count,
    )
