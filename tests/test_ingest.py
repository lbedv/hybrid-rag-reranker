import json
from pathlib import Path
from unittest.mock import patch
import uuid
from uuid import UUID
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.db.models import Incident
from app.db.session import SessionLocal
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tracked_ids():
    """Tracks IDs created during tests and guarantees teardown cleanup."""
    ids: list[str] = []
    yield ids
    if ids:
        with SessionLocal() as db:
            db.execute(delete(Incident).where(Incident.id.in_([UUID(i) for i in ids])))
            db.commit()


def test_ingest_small_batch_and_idempotency(client, tracked_ids):
    """Test ingestion of new documents and verify idempotency (0 inserted on duplicate)."""
    id_1 = str(uuid.uuid4())
    id_2 = str(uuid.uuid4())
    tracked_ids.extend([id_1, id_2])

    payload = [
        {
            "id": id_1,
            "title": "Air freight battery fire alarm at Hong Kong cargo terminal",
            "content": "Lithium cargo shipment emitted smoke during pallet staging. Fire suppression activated.",
            "category": "security",
            "region": "APAC",
            "severity_level": 4,
            "occurred_at": "2026-11-15T10:00:00+00:00",
        },
        {
            "id": id_2,
            "title": "Train derailment blocking freight corridor in Ohio",
            "content": "Intermodal freight cars left tracks; hazmat cars intact but rail link severed.",
            "category": "delay",
            "region": "US-East",
            "severity_level": 5,
            "occurred_at": "2026-11-14T08:30:00+00:00",
        },
    ]

    # First ingest: both should be inserted
    res1 = client.post("/api/v1/documents/ingest", json=payload)
    assert res1.status_code == 201
    data1 = res1.json()
    assert data1["total"] == 2
    assert data1["inserted"] == 2
    assert data1["skipped"] == 0

    # Second ingest (idempotency): 0 inserted, 2 skipped
    res2 = client.post("/api/v1/documents/ingest", json=payload)
    assert res2.status_code == 201
    data2 = res2.json()
    assert data2["total"] == 2
    assert data2["inserted"] == 0
    assert data2["skipped"] == 2

    # Verify search_vector is automatically computed by Postgres
    with SessionLocal() as db:
        item = db.execute(select(Incident).where(Incident.id == UUID(id_1))).scalar_one()
        assert item.search_vector is not None
        assert len(item.embedding) == 1536


def test_ingest_intra_batch_duplicates(client, tracked_ids):
    """Test sending duplicate IDs inside the same request payload (prevents CardinalityViolation)."""
    dup_id = str(uuid.uuid4())
    tracked_ids.append(dup_id)

    payload = [
        {
            "id": dup_id,
            "title": "Customs clearance delay at Dover port",
            "content": "Trucks queued for physical paperwork check.",
            "category": "customs",
            "region": "EU-West",
            "severity_level": 3,
            "occurred_at": "2026-11-16T12:00:00+00:00",
        },
        {
            "id": dup_id,  # Duplicate in same batch
            "title": "Customs clearance delay at Dover port duplicate",
            "content": "Trucks queued for physical paperwork check.",
            "category": "customs",
            "region": "EU-West",
            "severity_level": 3,
            "occurred_at": "2026-11-16T12:00:00+00:00",
        },
    ]

    res = client.post("/api/v1/documents/ingest", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["total"] == 2
    assert data["inserted"] == 1
    assert data["skipped"] == 1


def test_ingest_embedding_service_error(client, tracked_ids):
    """Test graceful 502 Bad Gateway response when external embedding service fails."""
    doc_id = str(uuid.uuid4())
    tracked_ids.append(doc_id)

    payload = [
        {
            "id": doc_id,
            "title": "Test failure handling",
            "content": "Testing embedding failure branch.",
            "category": "network",
            "severity_level": 2,
            "occurred_at": "2026-11-18T10:00:00+00:00",
        }
    ]

    with patch(
        "app.api.routes.embedding_service.get_embeddings",
        side_effect=RuntimeError("OpenAI API rate limit exceeded (HTTP 429)"),
    ):
        res = client.post("/api/v1/documents/ingest", json=payload)
        assert res.status_code == 502
        assert "Embedding generation failed" in res.json()["detail"]


def test_ingest_sample_dataset(client):
    """Ingest the complete sample incidents from data/sample_incidents.json."""
    data_path = Path("data/sample_incidents.json")
    assert data_path.exists(), "Sample dataset file must exist"

    with open(data_path) as f:
        incidents = json.load(f)

    assert len(incidents) >= 80

    # Ingest full dataset
    res = client.post("/api/v1/documents/ingest", json=incidents)
    assert res.status_code == 201
    data = res.json()
    assert data["total"] == len(incidents)
    assert data["inserted"] + data["skipped"] == len(incidents)

    # Re-ingest should skip all
    res_repeat = client.post("/api/v1/documents/ingest", json=incidents)
    assert res_repeat.status_code == 201
    data_repeat = res_repeat.json()
    assert data_repeat["inserted"] == 0
    assert data_repeat["skipped"] == len(incidents)

    # Verify only the sample IDs in DB without global table dependency
    sample_ids = [UUID(item["id"]) for item in incidents]
    with SessionLocal() as db:
        count = db.execute(
            select(func.count(Incident.id)).where(Incident.id.in_(sample_ids))
        ).scalar()
        assert count == len(incidents)


def test_ingest_omitted_or_null_embedding_auto_computes(client, tracked_ids):
    """When embedding is None or omitted, it is auto-computed."""
    test_id = str(uuid.uuid4())
    tracked_ids.append(test_id)

    payload = [
        {
            "id": test_id,
            "title": "Null embedding test",
            "content": "Testing auto-computation when embedding is null.",
            "category": "customs",
            "region": "EU-West",
            "severity_level": 3,
            "occurred_at": "2026-11-20T12:00:00Z",
            "embedding": None,
        }
    ]

    res = client.post("/api/v1/documents/ingest", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["inserted"] == 1

    with SessionLocal() as db:
        item = db.execute(select(Incident).where(Incident.id == UUID(test_id))).scalar_one()
        assert len(item.embedding) == 1536


def test_ingest_invalid_embedding_dimensions(client):
    """Explicitly passing invalid dimensions (e.g. 1 float [0] or 2 floats) returns clean 422 Unprocessable Entity."""
    payload_single = [
        {
            "id": str(uuid.uuid4()),
            "title": "Single element test",
            "content": "Content",
            "category": "customs",
            "severity_level": 1,
            "occurred_at": "2026-11-20T12:00:00Z",
            "embedding": [0],  # Length 1 instead of 1536
        }
    ]
    res1 = client.post("/api/v1/documents/ingest", json=payload_single)
    assert res1.status_code == 422
    assert "at least 1536 items" in res1.text

    payload_double = [
        {
            "id": str(uuid.uuid4()),
            "title": "Two elements test",
            "content": "Content",
            "category": "customs",
            "severity_level": 1,
            "occurred_at": "2026-11-20T12:00:00Z",
            "embedding": [0.1, 0.2],  # Length 2 instead of 1536
        }
    ]
    res2 = client.post("/api/v1/documents/ingest", json=payload_double)
    assert res2.status_code == 422
    assert "at least 1536 items" in res2.text
