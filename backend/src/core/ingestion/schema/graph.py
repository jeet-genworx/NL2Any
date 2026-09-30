"""Builds a nodes/edges graph view of a DatabaseSchema.

An additive representation alongside the canonical schema TOML: nodes are
tables or collections carrying their fields, edges are the schema's
relationships (foreign keys for PostgreSQL, none for MongoDB). The description
stage walks this view -- or the MST reduced from it -- to decide which tables
are described together, and to tell the model which columns exist.

Pure: building the structure is here, writing it to disk is the graph
repository's job.
"""

from datetime import datetime, timezone
from typing import Any

from backend.src.data.models.schema import DatabaseSchema, SchemaObject


def _columns(obj: SchemaObject) -> list[dict[str, Any]]:
    """Every field of an object, subdocument fields included.

    Nested fields are flattened to dotted paths (`address.city`), because this
    list is what the description stage shows the model -- and the prompt forbids
    inventing columns, so a field that does not appear here can never get a
    description. The containing field is listed too (`address` alongside
    `address.city`), so the subdocument itself can be described.

    `apply_descriptions` resolves the same dotted paths back into nested fields,
    so descriptions land where they belong.
    """
    columns = []
    for path in obj.all_field_paths():
        field = obj.get_field(path)
        if field is not None:
            columns.append({"name": path, "type": field.type, "nullable": field.nullable})
    return columns


def build_schema_graph(schema: DatabaseSchema) -> dict[str, Any]:
    """Convert a DatabaseSchema into a nodes/edges graph structure."""
    nodes = [
        {
            "id": obj.name,
            "kind": obj.kind.value,
            "columns": _columns(obj),
        }
        for obj in schema.objects
    ]

    edges = [
        {
            "from": rel.from_object,
            "from_column": rel.from_field,
            "to": rel.to_object,
            "to_column": rel.to_field,
            "type": rel.relationship_type,
        }
        for rel in schema.relationships
    ]

    return {
        "graph": {
            "database_type": schema.database_type.value,
            "database_name": schema.database_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
