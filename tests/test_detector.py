"""Tests for deterministic database detection."""

import pytest
from nl2anyquery.databases.detector import detect_database_type
from nl2anyquery.models.schema import DatabaseType


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
