"""Reads and writes the canonical schema TOML.

This is the authoritative description of a database -- its tables or
collections, their fields, and the relationships between them -- and the file
query processing loads at query time. Ingestion writes it twice: once from the
extracted metadata, and again after the description stage has folded generated
prose into it.

PostgreSQL tables and MongoDB collections differ only in which key their fields
are stored under (`columns` vs `fields`), so both directions are driven by
`_OBJECT_SECTIONS` rather than by two parallel code paths.
"""

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.data.repositories import files

# (TOML section, object kind, key its fields are stored under)
_OBJECT_SECTIONS: tuple[tuple[str, SchemaObjectKind, str], ...] = (
    ("tables", SchemaObjectKind.TABLE, "columns"),
    ("collections", SchemaObjectKind.COLLECTION, "fields"),
)


def _serialize_field(field: Field) -> dict[str, Any]:
    """Serialize one field, recursing into nested fields (MongoDB subdocuments)."""
    data: dict[str, Any] = {
        "type": field.type,
        "description": field.description or "",
        "nullable": field.nullable,
    }
    if field.sample_values:
        data["sample_values"] = field.sample_values
    if field.nested:
        data["fields"] = {child.name: _serialize_field(child) for child in field.nested}
    return data


def _deserialize_field(name: str, data: dict[str, Any]) -> Field:
    """Rebuild one field from its serialized form, recursing into nested fields."""
    nested_data = data.get("fields")
    return Field(
        name=name,
        type=data.get("type", "unknown"),
        description=data.get("description", ""),
        nullable=data.get("nullable", True),
        sample_values=data.get("sample_values", []),
        nested=[
            _deserialize_field(child_name, child_data)
            for child_name, child_data in (nested_data or {}).items()
        ]
        if isinstance(nested_data, dict)
        else [],
    )


def dump_schema_to_toml(schema: DatabaseSchema) -> str:
    """Serialize a DatabaseSchema into canonical TOML, validating it first."""
    schema.validate_consistency()

    doc: dict[str, Any] = {
        "database": {
            "type": schema.database_type.value,
            "name": schema.database_name,
            "schema_version": schema.schema_version,
            "last_updated": schema.last_updated or "",
            "description_generated_at": schema.description_generated_at or "",
        }
    }

    for section, kind, field_key in _OBJECT_SECTIONS:
        entries = {
            obj.name: {
                "kind": obj.kind.value,
                "description": obj.description or "",
                field_key: {field.name: _serialize_field(field) for field in obj.fields},
            }
            for obj in schema.objects
            if obj.kind == kind
        }
        if entries:
            doc[section] = entries

    if schema.relationships:
        doc["relationships"] = [
            {
                "from_table": rel.from_object,
                "from_column": rel.from_field,
                "to_table": rel.to_object,
                "to_column": rel.to_field,
                "type": rel.relationship_type,
            }
            for rel in schema.relationships
        ]

    return tomli_w.dumps(doc)


def load_schema_from_toml(toml_str: str) -> DatabaseSchema:
    """Deserialize canonical TOML back into a DatabaseSchema, validating it."""
    doc = tomllib.loads(toml_str)
    db_info = doc.get("database", {})

    objects = [
        SchemaObject(
            name=name,
            kind=kind,
            description=entry.get("description", ""),
            fields=[
                _deserialize_field(field_name, field_data)
                for field_name, field_data in entry.get(field_key, {}).items()
            ],
        )
        for section, kind, field_key in _OBJECT_SECTIONS
        for name, entry in doc.get(section, {}).items()
    ]

    relationships = [
        Relationship(
            from_object=rel.get("from_table", ""),
            from_field=rel.get("from_column", ""),
            to_object=rel.get("to_table", ""),
            to_field=rel.get("to_column", ""),
            relationship_type=rel.get("type", "many_to_one"),
        )
        for rel in doc.get("relationships", [])
    ]

    schema = DatabaseSchema(
        database_type=DatabaseType(db_info.get("type", "postgresql")),
        database_name=db_info.get("name", "database"),
        schema_version=db_info.get("schema_version", "1.0"),
        last_updated=db_info.get("last_updated") or None,
        description_generated_at=db_info.get("description_generated_at") or None,
        objects=objects,
        relationships=relationships,
    )
    schema.validate_consistency()
    return schema


def save_schema(schema: DatabaseSchema, file_path: Path | str) -> Path:
    """Write a DatabaseSchema to a TOML file on disk."""
    return files.write_text(file_path, dump_schema_to_toml(schema))


def load_schema(file_path: Path | str) -> DatabaseSchema:
    """Read a DatabaseSchema from a TOML file on disk."""
    return load_schema_from_toml(
        files.read_text(file_path, hint="Run 'uv run init-schema' first.")
    )
