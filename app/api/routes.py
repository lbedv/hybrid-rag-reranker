from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db

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
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {exc}",
        ) from exc
