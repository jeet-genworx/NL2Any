"""Pydantic models for the ingestion pipeline."""

from typing import Any

from pydantic import BaseModel, Field

from backend.src.data.models.schema import DatabaseType, SchemaObjectKind


class TableDescription(BaseModel):
    """One table's generated documentation: prose for the table, a line per column.

    Only `description` is embedded. `columns` is written into the schema TOML as
    documentation and deliberately kept out of the retrieval vectors.
    """

    description: str = ""
    columns: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, entry: Any, *, keep_empty_columns: bool = False) -> "TableDescription":
        """Build a TableDescription from loosely-shaped input.

        Two callers need this. The description stage parses model output, where
        the prompt asks for {"description": ..., "columns": {...}} but a smaller
        model sometimes collapses that to a bare string -- which should still
        cost us the columns, not the table description as well. The
        documentation repository reads files whose column slots may not have
        been filled in yet. Anything else yields an empty description rather
        than raising, so one malformed table cannot fail a whole batch.

        Undescribed columns are dropped by default, so an empty string never
        reaches the schema TOML as a description. `keep_empty_columns` retains
        them, for callers that want the full column list.
        """
        if isinstance(entry, str):
            return cls(description=entry.strip())
        if not isinstance(entry, dict):
            return cls()

        raw_columns = entry.get("columns")
        columns: dict[str, str] = {}
        if isinstance(raw_columns, dict):
            for name, text in raw_columns.items():
                cleaned = str(text).strip()
                if cleaned or keep_empty_columns:
                    columns[str(name)] = cleaned

        return cls(
            description=str(entry.get("description", "")).strip(),
            columns=columns,
        )


class Documentation(BaseModel):
    """A whole database's generated documentation, as read from its TOML file.

    Self-contained: it carries the database metadata and the object kind
    alongside the per-table text, so a reader never has to consult the schema
    TOML to answer a question about it.
    """

    database_type: DatabaseType
    database_name: str = ""
    generated_at: str = ""
    object_kind: SchemaObjectKind
    objects: dict[str, TableDescription] = Field(default_factory=dict)
