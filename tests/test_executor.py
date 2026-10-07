"""Tests that a query is executed against the database it was written for.

Every target carries its own connection string, and a database added from a
connection string has no settings entry to fall back on. If the executor reaches
for a setting instead of what it was handed, the query still succeeds -- against
the wrong server -- and returns rows that look entirely plausible. These tests
pin the routing so that cannot happen silently.
"""

from typing import Any

import pytest

from backend.src.core.query_processing.pipeline import executor as executor_module
from backend.src.core.query_processing.pipeline.executor import QueryExecutor
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import GeneratedQuery, MongoQuery


class FakeCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def sort(self, *_args: Any) -> "FakeCursor":
        return self

    def limit(self, *_args: Any) -> "FakeCursor":
        return self

    def __iter__(self):
        return iter(self.docs)


class FakeCollection:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def find(self, *_args: Any, **_kwargs: Any) -> FakeCursor:
        return FakeCursor(self.docs)


class FakeMongoClient:
    """Records the URI it was built with and the database that was asked for."""

    last: "FakeMongoClient | None" = None

    def __init__(self, uri: str, **_kwargs: Any) -> None:
        self.uri = uri
        self.requested_database: str | None = None
        self.closed = False
        FakeMongoClient.last = self

    def __getitem__(self, name: str) -> dict[str, FakeCollection]:
        self.requested_database = name
        return {"orders": FakeCollection([{"_id": 1, "total": 10}])}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_mongo(monkeypatch):
    FakeMongoClient.last = None
    monkeypatch.setattr(executor_module, "MongoClient", FakeMongoClient)
    return FakeMongoClient


def mongo_query() -> GeneratedQuery:
    raw = MongoQuery(collection="orders", operation="find", filter={})
    return GeneratedQuery(
        database_type=DatabaseType.MONGODB,
        raw_query=raw,
        formatted_query=raw.model_dump_json(),
    )


def test_mongo_uses_the_supplied_uri_and_database(fake_mongo):
    """The heart of it: a target's own connection and database, not settings'."""
    executor = QueryExecutor(
        mongo_uri="mongodb://settings-host:27017",
        mongo_database="settings_db",
    )

    columns, rows = executor.execute(
        mongo_query(),
        connection="mongodb://supplied-host:27017/supplied_db",
        database="supplied_db",
    )

    assert fake_mongo.last.uri == "mongodb://supplied-host:27017/supplied_db"
    assert fake_mongo.last.requested_database == "supplied_db"
    assert columns == ["_id", "total"]
    assert rows == [{"_id": 1, "total": 10}]


def test_mongo_falls_back_to_settings_for_the_built_in_target(fake_mongo):
    """The configured MongoDB URI carries no database in its path, so the
    settings value is still what the built-in target relies on."""
    executor = QueryExecutor(
        mongo_uri="mongodb://settings-host:27017",
        mongo_database="settings_db",
    )

    executor.execute(mongo_query())

    assert fake_mongo.last.uri == "mongodb://settings-host:27017"
    assert fake_mongo.last.requested_database == "settings_db"


def test_mongo_client_is_closed_even_though_the_uri_came_from_a_target(fake_mongo):
    QueryExecutor(mongo_uri="mongodb://a", mongo_database="b").execute(
        mongo_query(), connection="mongodb://c", database="d"
    )
    assert fake_mongo.last.closed is True


def test_mongo_without_any_database_name_says_so(fake_mongo, monkeypatch):
    """A URI with no database in its path and nothing configured has to fail
    loudly; picking an arbitrary database would be worse than stopping."""
    monkeypatch.setattr(executor_module.settings, "mongodb_database", "")
    executor = QueryExecutor(mongo_uri="mongodb://host:27017", mongo_database="")

    with pytest.raises(ValueError, match="database name"):
        executor.execute(mongo_query(), connection="mongodb://host:27017")


def test_postgres_uses_the_supplied_dsn(monkeypatch):
    recorded: dict[str, str] = {}

    class FakeCursorCtx:
        description = [("id",)]

        def execute(self, *_args: Any) -> None:
            pass

        def fetchmany(self, _n: int) -> list[tuple[int]]:
            return [(1,)]

        def __enter__(self): return self
        def __exit__(self, *exc: object) -> None: pass

    class FakeConnection:
        def cursor(self): return FakeCursorCtx()
        def __enter__(self): return self
        def __exit__(self, *exc: object) -> None: pass

    def fake_connect(dsn: str, **_kwargs: Any) -> FakeConnection:
        recorded["dsn"] = dsn
        return FakeConnection()

    monkeypatch.setattr(executor_module.psycopg, "connect", fake_connect)

    executor = QueryExecutor(postgres_dsn="postgresql://settings/demo")
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT 1",
        formatted_query="SELECT 1",
    )

    executor.execute(query, connection="postgresql://supplied/shop")
    assert recorded["dsn"] == "postgresql://supplied/shop"

    executor.execute(query)
    assert recorded["dsn"] == "postgresql://settings/demo"


def test_unconfigured_target_dsn_errors_instead_of_using_the_default_database():
    """An unconfigured target must not silently execute against POSTGRES_DSN.

    `finops` with FINOPS_DSN unset yields connection="" and the old
    `dsn or self.postgres_dsn` ran the query against the demo database instead.
    That does not fail -- it answers from the wrong data or returns no rows,
    which is far worse than an error.
    """
    executor = QueryExecutor(postgres_dsn="postgresql://real:real@localhost:5432/demo")
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT 1;",
        formatted_query="SELECT 1;",
    )
    with pytest.raises(ValueError, match="not configured"):
        executor.execute(query, connection="")
    with pytest.raises(ValueError, match="not configured"):
        executor.execute(query, connection="   ")


def test_unconfigured_mongo_target_uri_errors():
    """Same guard on the MongoDB path."""
    executor = QueryExecutor(mongo_uri="mongodb://localhost:27017", mongo_database="demo")
    query = GeneratedQuery(
        database_type=DatabaseType.MONGODB,
        raw_query=MongoQuery(operation="find", collection="users"),
        formatted_query="{}",
    )
    with pytest.raises(ValueError, match="not configured"):
        executor.execute(query, connection="", database="finops")
