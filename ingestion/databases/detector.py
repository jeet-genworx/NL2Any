"""Deterministic database type detection based on connection URI scheme."""

import re
from query_processing.models.schema import DatabaseType


def detect_database_type(connection_string: str) -> DatabaseType:
    """Deterministically detect database type from a connection string.

    Supports:
        postgresql://... -> DatabaseType.POSTGRESQL
        postgres://...   -> DatabaseType.POSTGRESQL
        mongodb://...    -> DatabaseType.MONGODB
        mongodb+srv://.. -> DatabaseType.MONGODB

    Does not use the SLM.
    """
    if not connection_string or not isinstance(connection_string, str):
        raise ValueError("Connection string must be a non-empty string.")

    cleaned = connection_string.strip()

    # Match scheme prefix
    match = re.match(r"^([a-zA-Z0-9\+\-]+)://", cleaned)
    if not match:
        raise ValueError(f"Invalid connection string format: '{connection_string}'")

    scheme = match.group(1).lower()

    if scheme in ("postgresql", "postgres"):
        return DatabaseType.POSTGRESQL
    elif scheme in ("mongodb", "mongodb+srv"):
        return DatabaseType.MONGODB
    else:
        raise ValueError(
            f"Unsupported database scheme: '{scheme}'. Only PostgreSQL and MongoDB are supported."
        )
