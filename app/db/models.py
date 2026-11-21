from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    Index,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID

from app.core.config import settings
from app.db.session import Base


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(64), nullable=False, index=True)
    region = Column(String(64), nullable=True)
    severity_level = Column(
        SmallInteger,
        CheckConstraint("severity_level BETWEEN 1 AND 5", name="ck_incident_severity_range"),
        nullable=False,
    )
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIM), nullable=False)

    # Generated stored tsvector column for fast full-text search
    search_vector = Column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))",
            persisted=True,
        ),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_incidents_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_incidents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
