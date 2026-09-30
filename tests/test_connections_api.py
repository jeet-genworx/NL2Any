"""Tests for adding a database from a connection string.

The behaviour these protect is the one a user notices: a database supplied twice
is not ingested twice. Everything else here -- the probe, the 400/502 split,
keeping credentials out of responses -- exists to make that first experience
safe rather than surprising.
"""

import json

import pytest
from fastapi.testclient import TestClient

from backend.src.api.rest.app import app
from backend.src.api.rest.routes import connections as connections_route
from backend.src.config import settings
from backend.src.data.models.targets import (
    POSTGRES_TARGET,
    clear_dynamic_targets,
    dynamic_targets,
    register_target,
    target_from_connection,
)
from backend.src.data.repositories import connection_repository

SHOP_DSN = "postgresql://user:password@db.internal:5432/shop"

client = TestClient(app)


class FakeAdapter:
    """Stands in for a real adapter so the probe never needs a live database."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.pinged = False

    def ping(self) -> None:
        self.pinged = True
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeAdapter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    """Keep every test's saved connections in its own file and out of the repo."""
    monkeypatch.setattr(settings, "connections_path", str(tmp_path / "connections.json"))
    monkeypatch.setattr(settings, "embeddings_dir", str(tmp_path / "embeddings"))
    clear_dynamic_targets()
    yield
    clear_dynamic_targets()


@pytest.fixture
def reachable(monkeypatch):
    """Make the probe succeed without a database behind it."""
    adapter = FakeAdapter()
    monkeypatch.setattr(connections_route, "get_adapter", lambda target: adapter)
    return adapter


def test_a_connection_string_is_saved_and_returns_a_key(reachable):
    response = client.post("/connections", json={"connection_string": SHOP_DSN, "label": "Shop"})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Shop"
    assert body["database_type"] == "postgresql"
    assert body["database_name"] == "shop"
    assert body["already_saved"] is False
    assert body["key"] in dynamic_targets()
    # The connection was proven before being saved.
    assert reachable.pinged is True


def test_the_saved_database_becomes_selectable(reachable):
    key = client.post("/connections", json={"connection_string": SHOP_DSN}).json()["key"]

    listed = {row["key"]: row for row in client.get("/databases").json()["databases"]}
    assert key in listed
    assert listed[key]["source"] == "user"
    # The built-in databases are still there alongside it.
    assert "postgres" in listed


def test_supplying_the_same_database_again_reuses_the_existing_entry(reachable):
    first = client.post("/connections", json={"connection_string": SHOP_DSN, "label": "Shop"}).json()

    # Same database, different credentials, different label.
    again = client.post(
        "/connections",
        json={
            "connection_string": "postgresql://someone:else@db.internal:5432/shop",
            "label": "Ignored",
        },
    ).json()

    assert again["already_saved"] is True
    assert again["key"] == first["key"]
    assert again["label"] == "Shop"
    assert len(dynamic_targets()) == 1


def test_a_repeat_is_not_probed_again(monkeypatch, reachable):
    client.post("/connections", json={"connection_string": SHOP_DSN})

    def fail(target):
        raise AssertionError("a database already saved must not be probed again")

    monkeypatch.setattr(connections_route, "get_adapter", fail)
    assert client.post("/connections", json={"connection_string": SHOP_DSN}).status_code == 200


def test_pasting_a_configured_databases_dsn_selects_it(monkeypatch, reachable):
    """Otherwise the demo database would be saved a second time under a generated
    key, and ingested again from scratch."""
    monkeypatch.setattr(settings, "postgres_dsn", SHOP_DSN)

    body = client.post("/connections", json={"connection_string": SHOP_DSN}).json()

    assert body["key"] == POSTGRES_TARGET.key
    assert body["already_saved"] is True
    assert body["source"] == "builtin"
    assert dynamic_targets() == {}


def test_an_unsupported_engine_is_rejected_before_any_connection(monkeypatch):
    def fail(target):
        raise AssertionError("an unparseable connection string must not be dialled")

    monkeypatch.setattr(connections_route, "get_adapter", fail)

    response = client.post("/connections", json={"connection_string": "redis://localhost:6379"})
    assert response.status_code == 400
    assert "redis" in response.json()["detail"]
    assert dynamic_targets() == {}


def test_an_unreachable_database_is_not_saved(monkeypatch):
    monkeypatch.setattr(
        connections_route,
        "get_adapter",
        lambda target: FakeAdapter(error=ConnectionError("connection refused")),
    )

    response = client.post("/connections", json={"connection_string": SHOP_DSN})

    assert response.status_code == 502
    assert "connection refused" in response.json()["detail"]
    assert dynamic_targets() == {}


def test_responses_never_carry_the_connection_string(reachable):
    """It is supplied once and stays on the backend; the picker works in keys."""
    created = client.post("/connections", json={"connection_string": SHOP_DSN})
    listed = client.get("/connections")
    databases = client.get("/databases")

    for response in (created, listed, databases):
        assert "password" not in response.text
        assert SHOP_DSN not in response.text


def test_a_saved_connection_can_be_removed(reachable):
    key = client.post("/connections", json={"connection_string": SHOP_DSN}).json()["key"]

    assert client.delete(f"/connections/{key}").status_code == 200
    assert dynamic_targets() == {}
    assert client.delete(f"/connections/{key}").status_code == 404


def test_a_built_in_database_cannot_be_removed(reachable):
    assert client.delete("/connections/postgres").status_code == 400


# --- persistence --------------------------------------------------------------


def test_saved_connections_survive_a_restart(reachable):
    key = client.post("/connections", json={"connection_string": SHOP_DSN, "label": "Shop"}).json()["key"]

    # What a restart does: the process forgets, then reads the file back.
    clear_dynamic_targets()
    assert dynamic_targets() == {}
    restored = connection_repository.load_connections()

    assert [t.key for t in restored] == [key]
    reloaded = dynamic_targets()[key]
    assert reloaded.connection == SHOP_DSN
    assert reloaded.label == "Shop"
    # Same key means the same artifact filenames, so nothing is re-ingested.
    assert reloaded.file_stem == target_from_connection(SHOP_DSN).file_stem


def test_a_mongo_connection_keeps_its_database_name_across_a_restart(reachable):
    client.post("/connections", json={"connection_string": "mongodb://host:27017/analytics"})

    clear_dynamic_targets()
    restored = connection_repository.load_connections()

    assert restored[0].mongo_database == "analytics"


def test_a_missing_connections_file_is_the_normal_first_run():
    assert connection_repository.load_connections() == []


def test_a_corrupt_connections_file_does_not_stop_the_service():
    connection_repository.connections_path().parent.mkdir(parents=True, exist_ok=True)
    connection_repository.connections_path().write_text("{not json", encoding="utf-8")

    assert connection_repository.load_connections() == []


def test_a_malformed_record_is_skipped_and_the_rest_load():
    register_target(target_from_connection(SHOP_DSN))
    connection_repository.save_connections()

    path = connection_repository.connections_path()
    records = json.loads(path.read_text(encoding="utf-8"))
    records.insert(0, {"label": "missing every required field"})
    path.write_text(json.dumps(records), encoding="utf-8")

    restored = connection_repository.load_connections()
    assert [t.connection for t in restored] == [SHOP_DSN]
