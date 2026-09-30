"""Tests for FastAPI endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from backend.src.api.rest.app import app


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



def test_table_description_endpoint():
    response = TestClient(app).get("/descriptions/postgres/customers")

    assert response.status_code == 200
    data = response.json()
    assert data["database_type"] == "postgresql"
    assert data["table"] == "customers"
    assert data["kind"] == "table"
    assert data["description"]
    # Every column is listed, described or not, so the shape is stable.
    assert data["column_count"] == len(data["columns"]) > 0
    assert "id" in data["columns"]


def test_table_description_endpoint_mongo_lists_nested_fields_as_dotted_paths():
    response = TestClient(app).get("/descriptions/mongo/customers")

    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "collection"
    assert "address.city" in data["columns"]


def test_table_description_endpoint_unknown_table():
    response = TestClient(app).get("/descriptions/postgres/does_not_exist")

    assert response.status_code == 404
    # The error lists what is available, so a typo is self-correcting.
    assert "customers" in response.json()["detail"]


def test_table_description_endpoint_invalid_db():
    assert TestClient(app).get("/descriptions/invalid_db/customers").status_code == 400


def test_table_description_endpoint_reads_only_the_documentation_toml(monkeypatch):
    """The request is served from the documentation file alone: point the schema
    path at nothing and the endpoint still answers in full."""
    monkeypatch.setattr(
        "backend.src.data.repositories.paths.schema_path",
        lambda db_type: Path("/nonexistent/schema.toml"),
    )

    response = TestClient(app).get("/descriptions/postgres/customers")

    assert response.status_code == 200
    data = response.json()
    assert data["database_name"] == "nl2anyquery_db"
    assert data["kind"] == "table"
    assert data["column_count"] > 0


def test_table_description_endpoint_missing_documentation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.src.data.repositories.paths.descriptions_path",
        lambda db_type: tmp_path / "absent.toml",
    )

    response = TestClient(app).get("/descriptions/postgres/customers")

    assert response.status_code == 404
    assert "describe-schema" in response.json()["detail"]


def test_repeated_requests_parse_the_file_once(monkeypatch):
    """Ten identical requests must not re-read or re-parse the TOML ten times."""
    from backend.src.data.repositories import description_repository, files

    files.clear_cache()
    parses = []
    original = description_repository._parse_documentation
    monkeypatch.setattr(
        description_repository,
        "_parse_documentation",
        lambda path: (parses.append(path), original(path))[1],
    )

    client = TestClient(app)
    for _ in range(10):
        assert client.get("/descriptions/postgres/customers").status_code == 200

    assert len(parses) == 1


def test_databases_endpoint_lists_finops():
    response = TestClient(app).get("/databases")

    assert response.status_code == 200
    rows = {row["key"]: row for row in response.json()["databases"]}
    # A superset, because databases added from a connection string are listed
    # here alongside the configured ones.
    assert {"postgres", "mongo", "finops"} <= set(rows)
    assert rows["finops"]["label"] == "FinOps (PostgreSQL)"
    assert rows["finops"]["database_type"] == "postgresql"
    assert rows["finops"]["source"] == "builtin"
    # The picker needs to know whether a target is usable, whether it has been
    # ingested yet, and whether it is one the user can remove.
    assert set(rows["finops"]) == {
        "key",
        "label",
        "database_type",
        "configured",
        "ingested",
        "source",
    }


def test_schema_endpoint_serves_finops_separately_from_postgres():
    """finops and postgres share an engine but must not share a schema."""
    client = TestClient(app)
    finops = client.get("/schema/finops").json()
    postgres = client.get("/schema/postgres").json()

    assert finops["database_name"] == "finopsiq_be"
    assert postgres["database_name"] == "nl2anyquery_db"
    assert finops["database_type"] == postgres["database_type"] == "postgresql"
    assert finops["object_count"] != postgres["object_count"]


def test_schema_endpoint_accepts_finops_aliases():
    client = TestClient(app)
    for alias in ("finops", "finopsiq", "finopsiq_be"):
        assert client.get(f"/schema/{alias}").status_code == 200


def test_unknown_database_error_lists_finops():
    detail = TestClient(app).get("/schema/nope").json()["detail"]
    assert "finops" in detail
