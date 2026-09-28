"""Builds a nodes/edges graph view of a DatabaseSchema.

An additive representation alongside the canonical schema TOML: nodes are
tables or collections carrying their fields, edges are the schema's
relationships (foreign keys for PostgreSQL, none for MongoDB). The description
stage walks this view -- or the MST reduced from it -- to decide which tables
are described together.

Pure: building the structure is here, writing it to disk is the graph
repository's job.
"""

from datetime import datetime, timezone
from typing import Any

from backend.src.data.models.schema import DatabaseSchema


def build_schema_graph(schema: DatabaseSchema) -> dict[str, Any]:
    """Convert a DatabaseSchema into a nodes/edges graph structure."""
    nodes = [
        {
            "id": obj.name,
            "kind": obj.kind.value,
            "columns": [
                {"name": field.name, "type": field.type, "nullable": field.nullable}
                for field in obj.fields
            ],
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
