"""TOML storage and serialization for DatabaseSchema.

The import site query processing has always used. The implementation lives in
`data.repositories.schema_repository`; this module exists so
`query_processing` keeps the import it was written against.
"""

from pathlib import Path

from backend.src.data.models.schema import DatabaseSchema
from backend.src.data.repositories import schema_repository


def load_schema_file(file_path: Path | str) -> DatabaseSchema:
    """Load DatabaseSchema from a TOML file on disk."""
    return schema_repository.load_schema(file_path)


def save_schema_file(schema: DatabaseSchema, file_path: Path | str) -> Path:
    """Save DatabaseSchema to a TOML file on disk."""
    return schema_repository.save_schema(schema, file_path)
