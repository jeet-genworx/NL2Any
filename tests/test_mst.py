"""Tests for the minimum spanning tree/forest reduction of the schema graph."""

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
from ingestion.schema.mst import (
    compute_minimum_spanning_tree,
    save_minimum_spanning_tree,
    get_default_mst_path,
)


def _table(name: str) -> SchemaObject:
    return SchemaObject(name=name, kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="integer")])


def test_mst_removes_redundant_cycle_edge():
    # Triangle: customers <-> orders <-> support_tickets <-> customers.
    # All three edges connect the same three nodes, so one must be dropped.
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[_table("customers"), _table("orders"), _table("support_tickets")],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="support_tickets", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="support_tickets", from_field="order_id", to_object="orders", to_field="id"),
        ],
    )

    mst = compute_minimum_spanning_tree(schema)

    assert mst["graph"]["node_count"] == 3
    assert mst["graph"]["total_edge_count"] == 3
    # A spanning tree over 3 connected nodes has exactly 2 edges (no cycle).
    assert mst["graph"]["mst_edge_count"] == 2
    assert len(mst["mst_edges"]) == 2
    assert mst["graph"]["component_count"] == 1


def test_mst_produces_forest_for_disconnected_graph():
    # customers<->orders is one component; products is isolated (no FKs).
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[_table("customers"), _table("orders"), _table("products")],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
        ],
    )

    mst = compute_minimum_spanning_tree(schema)

    assert mst["graph"]["node_count"] == 3
    assert mst["graph"]["mst_edge_count"] == 1
    assert mst["graph"]["component_count"] == 2  # {customers, orders} and {products}


def test_save_minimum_spanning_tree_writes_valid_toml(tmp_path):
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[_table("customers"), _table("orders")],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
        ],
    )
    out_path = tmp_path / "postgres_mst.toml"
    save_minimum_spanning_tree(schema, out_path)

    assert out_path.exists()
    parsed = tomllib.loads(out_path.read_text())
    assert len(parsed["mst_edges"]) == 1
    assert len(parsed["nodes"]) == 2


def test_mst_nodes_ordered_by_tree_traversal():
    # customers("A") -- orders("B") -- support_tickets("C"), a simple chain.
    # Objects are deliberately listed out of order to prove the output isn't
    # just echoing extraction order.
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[_table("support_tickets"), _table("customers"), _table("orders")],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="support_tickets", from_field="order_id", to_object="orders", to_field="id"),
        ],
    )

    mst = compute_minimum_spanning_tree(schema)

    assert [node["id"] for node in mst["nodes"]] == ["customers", "orders", "support_tickets"]


def test_mst_node_order_groups_each_component_together():
    # {customers, orders} is one connected component; {products} is isolated.
    # Nodes from the same component must stay adjacent in the output.
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[_table("products"), _table("orders"), _table("customers")],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
        ],
    )

    mst = compute_minimum_spanning_tree(schema)

    assert [node["id"] for node in mst["nodes"]] == ["customers", "orders", "products"]


def test_get_default_mst_path():
    assert get_default_mst_path(DatabaseType.POSTGRESQL) == Path("ingestion/schemas/postgres_mst.toml")
    assert get_default_mst_path(DatabaseType.MONGODB) == Path("ingestion/schemas/mongo_mst.toml")
