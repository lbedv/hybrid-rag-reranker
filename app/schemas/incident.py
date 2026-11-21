from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings


class IncidentBase(BaseModel):
    title: str = Field(..., min_length=1, description="Title or short summary of incident")
    content: str = Field(..., min_length=1, description="Detailed incident description")
    category: str = Field(..., min_length=1, description="Category (delay, damage, customs, etc.)")
    region: str | None = Field(default=None, description="Geographic region")
    severity_level: int = Field(..., ge=1, le=5, description="Severity score 1 to 5")
    occurred_at: datetime = Field(..., description="Timestamp of when the incident occurred")


class IncidentCreate(IncidentBase):
    id: UUID = Field(..., description="Stable unique incident UUID")
    embedding: list[float] | None = Field(
        default=None,
        min_length=settings.EMBEDDING_DIM,
        max_length=settings.EMBEDDING_DIM,
        description=f"Optional precomputed {settings.EMBEDDING_DIM}-dim embedding vector. Pass null or omit to auto-compute.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "title": "Port of Rotterdam customs hold on microchips",
                "content": "Customs inspection hold on critical semiconductor shipment at Port of Rotterdam.",
                "category": "customs",
                "region": "EU-West",
                "severity_level": 4,
                "occurred_at": "2026-11-20T12:00:00Z",
                "embedding": None,
            }
        }
    )


class IncidentRead(IncidentBase):
    id: UUID
    model_config = ConfigDict(from_attributes=True)


class IngestResponse(BaseModel):
    total: int
    inserted: int
    skipped: int
