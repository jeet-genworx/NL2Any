"""Tests for deterministic database detection and connection-string parsing."""

import pytest
from backend.src.data.clients.detector import (
    detect_database_type,
    driver_connection,
    identity_of,
    parse_connection,
    redact,
)
from backend.src.data.models.schema import DatabaseType


def test_detect_postgresql_schemes():
    assert detect_database_type("postgresql://user:pass@localhost:5432/mydb") == DatabaseType.POSTGRESQL
    assert detect_database_type("postgres://localhost/mydb") == DatabaseType.POSTGRESQL
    assert detect_database_type("postgresql://localhost:5432/db?sslmode=disable") == DatabaseType.POSTGRESQL


def test_detect_mongodb_schemes():
    assert detect_database_type("mongodb://localhost:27017") == DatabaseType.MONGODB
    assert detect_database_type("mongodb+srv://user:pass@cluster.mongodb.net/test") == DatabaseType.MONGODB


def test_detect_unsupported_scheme():
    with pytest.raises(ValueError, match="Unsupported database scheme: 'mysql'"):
        detect_database_type("mysql://user:pass@localhost:3306/db")

    with pytest.raises(ValueError, match="Unsupported database scheme: 'neo4j'"):
        detect_database_type("neo4j://localhost:7687")


def test_detect_invalid_format():
    with pytest.raises(ValueError, match="Invalid connection string format"):
        detect_database_type("not_a_valid_url")

    with pytest.raises(ValueError, match="Connection string must be a non-empty string"):
        detect_database_type("")


@pytest.mark.parametrize(
    "driver",
    ["asyncpg", "psycopg", "psycopg2", "pg8000", "psycopg2cffi"],
)
def test_detect_sqlalchemy_driver_urls(driver):
    """`postgresql+asyncpg://` is what an application's own config usually holds,
    so it is what gets pasted. The driver suffix names a Python library, not a
    different database."""
    assert (
        detect_database_type(f"postgresql+{driver}://u:p@host:5432/db")
        == DatabaseType.POSTGRESQL
    )


def test_detect_libpq_keyword_form():
    """psql and pgAdmin hand out this form freely, so pasting it should work."""
    assert detect_database_type("host=db.internal dbname=app user=me") == DatabaseType.POSTGRESQL
    assert detect_database_type("dbname=app") == DatabaseType.POSTGRESQL


# --- parse_connection ---------------------------------------------------------


def test_parse_reads_database_host_and_port():
    info = parse_connection("postgresql://u:p@db.internal:5433/shop")
    assert (info.db_type, info.database, info.host, info.port) == (
        DatabaseType.POSTGRESQL,
        "shop",
        "db.internal",
        "5433",
    )


def test_parse_reads_the_mongo_database_from_the_path():
    """Without this the database named in settings would be queried instead."""
    info = parse_connection("mongodb://host:27017/analytics?authSource=admin")
    assert info.database == "analytics"


def test_parse_tolerates_a_mongo_uri_with_no_database():
    info = parse_connection("mongodb://localhost:27017")
    assert info.database == ""
    assert info.host == "localhost"


def test_parse_is_not_confused_by_an_unencoded_at_in_a_password():
    """Both libpq and pymongo split on the last '@'; so must this, or the host
    silently becomes part of the password."""
    info = parse_connection("postgresql://user:pa@ss@realhost:5432/shop")
    assert info.host == "realhost"


def test_parse_reads_the_libpq_keyword_form():
    info = parse_connection("host=db.internal port=5433 dbname=app user=me password=secret")
    assert (info.database, info.host, info.port) == ("app", "db.internal", "5433")


def test_parse_refuses_an_unknown_engine():
    with pytest.raises(ValueError):
        parse_connection("mysql://localhost/db")


def test_parse_reads_a_sqlalchemy_url_like_any_other():
    info = parse_connection("postgresql+asyncpg://u:p@db.internal:5433/shop")
    assert (info.database, info.host, info.port) == ("shop", "db.internal", "5433")


# --- driver_connection --------------------------------------------------------


def test_the_sqlalchemy_driver_suffix_is_dropped_for_the_driver():
    """libpq rejects `postgresql+asyncpg://` outright, so the string handed to
    psycopg cannot be the one the user pasted."""
    assert (
        driver_connection("postgresql+asyncpg://u:p@host:5432/db?sslmode=require")
        == "postgresql://u:p@host:5432/db?sslmode=require"
    )
    assert driver_connection("postgres+psycopg2://host/db") == "postgres://host/db"


def test_mongodb_srv_is_never_rewritten():
    """`+srv` is not a driver suffix -- it means the hosts are resolved by SRV
    record, so stripping it would point pymongo at a different server."""
    assert driver_connection("mongodb+srv://u:p@cluster.mongodb.net/db") == (
        "mongodb+srv://u:p@cluster.mongodb.net/db"
    )


@pytest.mark.parametrize(
    "connection",
    [
        "postgresql://u:p@host:5432/db",
        "mongodb://localhost:27017/db",
        "host=db dbname=app user=me",
    ],
)
def test_an_ordinary_connection_string_is_passed_through_untouched(connection):
    """The drivers are the authority on their own syntax; only the SQLAlchemy
    scheme is rewritten."""
    assert driver_connection(connection) == connection


def test_a_sqlalchemy_url_is_the_same_database_as_the_plain_one():
    """Otherwise the same database pasted in two notations would be ingested
    twice, under two keys."""
    assert identity_of("postgresql+asyncpg://u:p@host:5432/shop") == identity_of(
        "postgresql://u:p@host:5432/shop"
    )


# --- identity_of --------------------------------------------------------------


def test_identity_ignores_credentials_and_host_case():
    baseline = identity_of("postgresql://u:p@localhost:5432/shop")
    assert identity_of("postgresql://other:rotated@LOCALHOST:5432/shop") == baseline
    assert identity_of("postgres://u:p@localhost:5432/shop") == baseline


def test_identity_separates_different_databases():
    assert identity_of("postgresql://h/one") != identity_of("postgresql://h/two")
    assert identity_of("postgresql://h1/db") != identity_of("postgresql://h2/db")
    # Same name, different engine, is still a different database.
    assert identity_of("postgresql://h:1/db") != identity_of("mongodb://h:1/db")


# --- redact -------------------------------------------------------------------


@pytest.mark.parametrize(
    "connection, secret",
    [
        ("postgresql://user:hunter2@host:5432/db", "hunter2"),
        ("mongodb+srv://user:hunter2@cluster.mongodb.net/db", "hunter2"),
        ("host=db dbname=app user=me password=hunter2", "hunter2"),
    ],
)
def test_redact_removes_the_password(connection, secret):
    """Connection strings reach logs and error responses; credentials must not."""
    masked = redact(connection)
    assert secret not in masked
    assert "***" in masked


def test_redact_leaves_a_credential_free_string_alone():
    assert redact("mongodb://localhost:27017/shop") == "mongodb://localhost:27017/shop"
