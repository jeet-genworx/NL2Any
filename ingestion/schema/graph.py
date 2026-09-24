"""Builds a nodes/edges graph view of a DatabaseSchema, for inspection/visualization.

This is a separate, additive representation alongside the canonical schema TOML
(toml_store.py) that query processing actually reads from. Nodes are tables or
collections, carrying their columns/fields as attributes. Edges are the schema's
relationships (foreign keys for PostgreSQL; empty for MongoDB, which has none).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import tomli_w

from query_processing.models.schema import DatabaseSchema, DatabaseType


def get_default_graph_path(database_type: DatabaseType) -> Path:
    """Return canonical graph TOML path for the database type."""
    filename = "postgres_graph.toml" if database_type == DatabaseType.POSTGRESQL else "mongo_graph.toml"
    return Path("ingestion/schemas") / filename


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


def save_schema_graph(schema: DatabaseSchema, file_path: Path | str) -> None:
    """Build and write the schema graph to a TOML file on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_schema_graph(schema)
    path.write_text(tomli_w.dumps(graph), encoding="utf-8")
