"""Reads and writes the generated documentation TOML.

One file per database, and the single place anything asks for a table's
description or its column descriptions. Ingestion writes it once; readers -- the
embedding stage, the API -- parse it and never recompute what it already holds.

    [database]
    type = "postgresql"
    name = "nl2anyquery_db"
    generated_at = "2026-09-28T09:12:44+00:00"

    [tables.customers]
    description = "Customer records, one row per person."

    [tables.customers.columns]
    id = "Surrogate primary key."
    email = "Contact email address."

When written with the schema in hand, every column of every object is listed --
including MongoDB subdocument fields, flattened to dotted keys that TOML quotes
and reads back verbatim -- so a reader gets the full shape from this file alone
and never has to consult the schema TOML. Columns still awaiting a description
are present with an empty value.

PostgreSQL objects live under `[tables.*]` and MongoDB's under `[collections.*]`,
matching the canonical schema TOML; reading accepts either.
"""

import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.src.data.models.schema import DatabaseSchema, DatabaseType, SchemaObjectKind
from backend.src.data.repositories import files
from backend.src.schemas.ingestion import Documentation, TableDescription

_MISSING_HINT = "Run 'uv run describe-schema' first."
_SECTIONS: tuple[tuple[str, SchemaObjectKind], ...] = (
    ("tables", SchemaObjectKind.TABLE),
    ("collections", SchemaObjectKind.COLLECTION),
)


def _section_for(database_type: DatabaseType) -> str:
    """The section this database type's documentation is stored under."""
    return "tables" if database_type == DatabaseType.POSTGRESQL else "collections"


def _columns_from_schema(
    schema: DatabaseSchema,
    name: str,
    described: TableDescription | None,
) -> dict[str, str] | None:
    """Every column of `name`, mapped to its description.

    Text comes from the generated descriptions, falling back to whatever the
    schema field already carries. Returns None when the schema has no such
    object, leaving the caller to store what was generated as-is.
    """
    obj = schema.get_object(name)
    if obj is None:
        return None

    generated = described.columns if described else {}
    columns: dict[str, str] = {}
    for path in obj.all_field_paths():
        field = obj.get_field(path)
        columns[path] = generated.get(path) or (field.description if field else "")
    return columns


def build_documentation(
    descriptions: dict[str, TableDescription],
    *,
    database_type: DatabaseType,
    database_name: str = "",
    schema: DatabaseSchema | None = None,
) -> dict[str, Any]:
    """Build the documentation document for a set of generated descriptions.

    Passing `schema` expands each object to its full column list; without it the
    file records exactly what was generated.
    """
    entries: dict[str, Any] = {}
    for name, table in descriptions.items():
        columns = _columns_from_schema(schema, name, table) if schema else None
        entries[name] = {
            "description": table.description,
            "columns": dict(table.columns) if columns is None else columns,
        }

    return {
        "database": {
            "type": database_type.value,
            "name": database_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        _section_for(database_type): entries,
    }


def save_descriptions(
    descriptions: dict[str, TableDescription],
    file_path: Path | str,
    *,
    database_type: DatabaseType,
    database_name: str = "",
    schema: DatabaseSchema | None = None,
) -> Path:
    """Write the table and column descriptions to a TOML file on disk."""
    return files.write_toml(
        file_path,
        build_documentation(
            descriptions,
            database_type=database_type,
            database_name=database_name,
            schema=schema,
        ),
    )


def _parse_documentation(path: Path) -> Documentation:
    """Parse a documentation TOML file into a Documentation model."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    database = document.get("database", {})

    section, kind = next(
        ((name, kind) for name, kind in _SECTIONS if document.get(name)),
        _SECTIONS[0],
    )

    return Documentation(
        database_type=DatabaseType(database.get("type", "postgresql")),
        database_name=database.get("name", ""),
        generated_at=database.get("generated_at", ""),
        object_kind=kind,
        objects={
            name: TableDescription.from_raw(entry, keep_empty_columns=True)
            for name, entry in document.get(section, {}).items()
        },
    )


def load_documentation(file_path: Path | str) -> Documentation:
    """Read a database's full documentation from its TOML file.

    The whole parsed model is cached against the file's modification time: a
    repeated request reuses it without re-reading or rebuilding anything, and a
    fresh ingestion run -- which rewrites the file -- is picked up on the next
    call. Treat the result as read-only; it is shared between callers.

    Undescribed columns are kept, so the caller sees the whole column list.
    """
    return files.read_cached(file_path, _parse_documentation, hint=_MISSING_HINT)


def load_descriptions(file_path: Path | str) -> dict[str, TableDescription]:
    """Read the generated descriptions, dropping columns with no description yet.

    The narrow view used by the embedding stage and by `apply_descriptions`,
    where an empty string must never be mistaken for a real description.
    """
    return {
        name: TableDescription(
            description=table.description,
            columns={col: text for col, text in table.columns.items() if text},
        )
        for name, table in load_documentation(file_path).objects.items()
    }
