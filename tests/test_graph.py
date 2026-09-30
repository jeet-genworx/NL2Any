"""Tests for the nodes/edges schema graph builder."""

from pathlib import Path
import tomllib

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.core.ingestion.schema.graph import build_schema_graph
from backend.src.data.repositories import graph_repository, paths


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


def test_save_graph_document_writes_valid_toml(tmp_path):
    out_path = tmp_path / "postgres_graph.toml"
    graph_repository.save_graph_document(build_schema_graph(_sample_schema()), out_path)

    assert out_path.exists()
    parsed = tomllib.loads(out_path.read_text())
    assert len(parsed["nodes"]) == 2
    assert len(parsed["edges"]) == 1


def test_graph_path():
    assert paths.graph_path(DatabaseType.POSTGRESQL) == Path("backend/src/data/schemas/postgres_graph.toml")
    assert paths.graph_path(DatabaseType.MONGODB) == Path("backend/src/data/schemas/mongo_graph.toml")


def _nested_schema() -> DatabaseSchema:
    """A MongoDB-shaped collection with a subdocument."""
    return DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop_demo",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.COLLECTION,
                fields=[
                    Field(name="_id", type="string"),
                    Field(
                        name="address",
                        type="object",
                        nested=[
                            Field(name="city", type="string"),
                            Field(name="zip_code", type="string"),
                        ],
                    ),
                ],
            )
        ],
    )


def test_graph_lists_nested_fields_as_dotted_paths():
    """The description stage can only describe columns this list names, and the
    prompt forbids inventing any -- so subdocument fields must appear here or
    they can never get a description."""
    node = build_schema_graph(_nested_schema())["nodes"][0]
    names = [col["name"] for col in node["columns"]]

    assert names == ["_id", "address", "address.city", "address.zip_code"]
    # The containing field is kept alongside its children, so the subdocument
    # itself can be described too.
    assert {"name": "address", "type": "object", "nullable": True} in node["columns"]


def test_graph_columns_cover_every_schema_field_path():
    for schema in (_nested_schema(), _sample_schema()):
        node_columns = {
            col["name"] for node in build_schema_graph(schema)["nodes"] for col in node["columns"]
        }
        expected = {path for obj in schema.objects for path in obj.all_field_paths()}
        assert node_columns == expected
