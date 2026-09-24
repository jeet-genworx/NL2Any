"""Canonical schema TOML path resolution, used by both ingestion and query processing."""

from pathlib import Path
from query_processing.models.schema import DatabaseType


def get_default_schema_path(database_type: DatabaseType) -> Path:
    """Return canonical schema TOML path for the database type."""
    filename = "postgres.toml" if database_type == DatabaseType.POSTGRESQL else "mongo.toml"
    return Path("ingestion/schemas") / filename
