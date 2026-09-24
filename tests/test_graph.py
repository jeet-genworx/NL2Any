"""Tests for the nodes/edges schema graph builder."""

from pathlib import Path
import tomllib

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from ingestion.schema.graph import build_schema_graph, save_schema_graph, get_default_graph_path


def _sample_schema() -> DatabaseSchema:
    customers = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        fields=[
            Field(name="id", type="integer", nullable=False),
            Field(name="name", type="varchar(100)", nullable=False),
        ],
    )
    orders = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[
            Field(name="id", type="integer", nullable=False),
            Field(name="customer_id", type="integer", nullable=False),
        ],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[customers, orders],
        relationships=[
            Relationship(
                from_object="orders",
                from_field="customer_id",
                to_object="customers",
                to_field="id",
                relationship_type="many_to_one",
            )
        ],
    )


def test_build_schema_graph_nodes_and_edges():
    graph = build_schema_graph(_sample_schema())

    assert graph["graph"]["database_type"] == "postgresql"
    assert graph["graph"]["node_count"] == 2
    assert graph["graph"]["edge_count"] == 1

    node_ids = {node["id"] for node in graph["nodes"]}
    assert node_ids == {"customers", "orders"}

    customers_node = next(n for n in graph["nodes"] if n["id"] == "customers")
    column_names = {col["name"] for col in customers_node["columns"]}
    assert column_names == {"id", "name"}

    edge = graph["edges"][0]
    assert edge["from"] == "orders"
    assert edge["from_column"] == "customer_id"
    assert edge["to"] == "customers"
    assert edge["to_column"] == "id"
    assert edge["type"] == "many_to_one"


def test_save_schema_graph_writes_valid_toml(tmp_path):
    out_path = tmp_path / "postgres_graph.toml"
    save_schema_graph(_sample_schema(), out_path)

    assert out_path.exists()
    parsed = tomllib.loads(out_path.read_text())
    assert len(parsed["nodes"]) == 2
    assert len(parsed["edges"]) == 1


def test_get_default_graph_path():
    assert get_default_graph_path(DatabaseType.POSTGRESQL) == Path("ingestion/schemas/postgres_graph.toml")
    assert get_default_graph_path(DatabaseType.MONGODB) == Path("ingestion/schemas/mongo_graph.toml")
