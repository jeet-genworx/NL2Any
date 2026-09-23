"""TOML storage and serialization for DatabaseSchema."""

import os
from pathlib import Path
import tomllib
from typing import Any
import tomli_w

from nl2anyquery.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)


def _serialize_field(field: Field) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": field.type,
        "description": field.description or "",
        "nullable": field.nullable,
    }
    if field.sample_values:
        data["sample_values"] = field.sample_values
    if field.nested:
        nested_dict: dict[str, Any] = {}
        for child in field.nested:
            nested_dict[child.name] = _serialize_field(child)
        data["fields"] = nested_dict
    return data


def _deserialize_field(name: str, data: dict[str, Any]) -> Field:
    nested: list[Field] = []
    if "fields" in data and isinstance(data["fields"], dict):
        for child_name, child_data in data["fields"].items():
            nested.append(_deserialize_field(child_name, child_data))

    return Field(
        name=name,
        type=data.get("type", "unknown"),
        description=data.get("description", ""),
        nullable=data.get("nullable", True),
        sample_values=data.get("sample_values", []),
        nested=nested,
    )


def dump_schema_to_toml(schema: DatabaseSchema) -> str:
    """Serialize DatabaseSchema into a clean, canonical TOML string.

    Performs consistency validation prior to serialization.
    """
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

    tables: dict[str, Any] = {}
    collections: dict[str, Any] = {}

    for obj in schema.objects:
        obj_dict: dict[str, Any] = {
            "kind": obj.kind.value,
            "description": obj.description or "",
        }
        field_group_key = "columns" if obj.kind == SchemaObjectKind.TABLE else "fields"
        fields_dict: dict[str, Any] = {}
        for field in obj.fields:
            fields_dict[field.name] = _serialize_field(field)
        obj_dict[field_group_key] = fields_dict

        if obj.kind == SchemaObjectKind.TABLE:
            tables[obj.name] = obj_dict
        else:
            collections[obj.name] = obj_dict

    if tables:
        doc["tables"] = tables
    if collections:
        doc["collections"] = collections

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
    """Deserialize TOML string into a DatabaseSchema model."""
    doc = tomllib.loads(toml_str)

    db_info = doc.get("database", {})
    db_type_str = db_info.get("type", "postgresql")
    db_type = DatabaseType(db_type_str)
    db_name = db_info.get("name", "database")
    schema_version = db_info.get("schema_version", "1.0")
    last_updated = db_info.get("last_updated") or None
    desc_gen_at = db_info.get("description_generated_at") or None

    objects: list[SchemaObject] = []

    # Tables
    tables_dict = doc.get("tables", {})
    for tbl_name, tbl_data in tables_dict.items():
        cols_dict = tbl_data.get("columns", {})
        fields = [_deserialize_field(c_name, c_data) for c_name, c_data in cols_dict.items()]
        objects.append(
            SchemaObject(
                name=tbl_name,
                kind=SchemaObjectKind.TABLE,
                description=tbl_data.get("description", ""),
                fields=fields,
            )
        )

    # Collections
    colls_dict = doc.get("collections", {})
    for coll_name, coll_data in colls_dict.items():
        flds_dict = coll_data.get("fields", {})
        fields = [_deserialize_field(f_name, f_data) for f_name, f_data in flds_dict.items()]
        objects.append(
            SchemaObject(
                name=coll_name,
                kind=SchemaObjectKind.COLLECTION,
                description=coll_data.get("description", ""),
                fields=fields,
            )
        )

    # Relationships
    relationships: list[Relationship] = []
    for rel_data in doc.get("relationships", []):
        relationships.append(
            Relationship(
                from_object=rel_data.get("from_table", rel_data.get("from_object", "")),
                from_field=rel_data.get("from_column", rel_data.get("from_field", "")),
                to_object=rel_data.get("to_table", rel_data.get("to_object", "")),
                to_field=rel_data.get("to_column", rel_data.get("to_field", "")),
                relationship_type=rel_data.get("type", "many_to_one"),
            )
        )

    schema = DatabaseSchema(
        database_type=db_type,
        database_name=db_name,
        schema_version=schema_version,
        last_updated=last_updated,
        description_generated_at=desc_gen_at,
        objects=objects,
        relationships=relationships,
    )
    schema.validate_consistency()
    return schema


def save_schema_file(schema: DatabaseSchema, file_path: Path | str) -> None:
    """Save DatabaseSchema to a TOML file on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dump_schema_to_toml(schema)
    path.write_text(content, encoding="utf-8")


def load_schema_file(file_path: Path | str) -> DatabaseSchema:
    """Load DatabaseSchema from a TOML file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found at: {path}")
    content = path.read_text(encoding="utf-8")
    return load_schema_from_toml(content)
