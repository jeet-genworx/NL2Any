"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from nl2anyquery.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_schema_endpoint():
    client = TestClient(app)
    response = client.get("/schema/postgres")
    assert response.status_code == 200
    data = response.json()
    assert data["database_type"] == "postgresql"
    assert data["object_count"] > 0


def test_schema_endpoint_invalid_db():
    client = TestClient(app)
    response = client.get("/schema/invalid_db")
    assert response.status_code == 400
