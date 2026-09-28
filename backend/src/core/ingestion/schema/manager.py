"""Canonical schema TOML path resolution.

The import site query processing has always used. Resolution itself lives in
`data.repositories.paths`, so there is one definition of where the file is;
this module exists so `query_processing` keeps the import it was written
against and does not have to follow the ingestion layering.
"""

from pathlib import Path

from backend.src.data.models.schema import DatabaseType
from backend.src.data.repositories import paths


def get_default_schema_path(database_type: DatabaseType) -> Path:
    """Return canonical schema TOML path for the database type."""
    return paths.schema_path(database_type)
