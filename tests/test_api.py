"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from query_processing.api.main import app


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


def test_query_endpoint_basic():
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"database": "postgres", "question": "What database is this?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["database"] == "postgresql"
    assert data["basic_answer"] is not None


def test_query_endpoint_reject():
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"database": "postgres", "question": "DROP TABLE customers;"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["guardrail"]["decision"] == "REJECT"
    assert data["error"] == "Sorry, I can't help with this."

