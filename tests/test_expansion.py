"""Tests for SchemaExpander."""

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.core.query_processing.pipeline.expansion import SchemaExpander


def _shop_schema() -> DatabaseSchema:
    orders = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="int"), Field(name="total", type="numeric")],
    )
    products = SchemaObject(
        name="products", kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="int")]
    )
    order_items = SchemaObject(
        name="order_items",
        kind=SchemaObjectKind.TABLE,
        fields=[
            Field(name="id", type="int"),
            Field(name="order_id", type="int"),
            Field(name="product_id", type="int"),
        ],
    )
    audit_log = SchemaObject(
        name="audit_log", kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="int")]
    )

    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[orders, products, order_items, audit_log],
        relationships=[
            Relationship(from_object="order_items", from_field="order_id", to_object="orders", to_field="id"),
            Relationship(from_object="order_items", from_field="product_id", to_object="products", to_field="id"),
        ],
    )


def test_materialize_fills_in_the_columns_for_the_selected_tables():
    schema = _shop_schema()
    expander = SchemaExpander()

    relevant = expander.materialize(
        ["orders", "order_items", "products"], schema.relationships, schema
    )

    assert {obj.name for obj in relevant.objects} == {"orders", "order_items", "products"}
    # The selector saw names and descriptions; the slice carries the bodies.
    orders = next(obj for obj in relevant.objects if obj.name == "orders")
    assert {f.name for f in orders.fields} == {"id", "total"}
    assert len(relevant.relationships) == 2


def test_materialize_adds_no_table_the_selector_did_not_choose():
    """The selector already chose from the whole neighborhood: do not overrule it."""
    schema = _shop_schema()

    relevant = SchemaExpander().materialize(["orders", "products"], schema.relationships, schema)

    assert {obj.name for obj in relevant.objects} == {"orders", "products"}
    # order_items is not in the slice, so neither are the edges that reference it.
    assert relevant.relationships == []


def test_materialize_drops_edges_whose_endpoints_are_not_in_the_slice():
    schema = _shop_schema()

    relevant = SchemaExpander().materialize(
        ["order_items", "orders"], schema.relationships, schema
    )

    assert len(relevant.relationships) == 1
    assert relevant.relationships[0].to_object == "orders"


def test_materialize_is_case_insensitive_and_de_duplicates_edges():
    schema = _shop_schema()
    duplicated = [schema.relationships[0], schema.relationships[0]]

    relevant = SchemaExpander().materialize(["ORDERS", "Order_Items"], duplicated, schema)

    assert {obj.name for obj in relevant.objects} == {"orders", "order_items"}
    assert len(relevant.relationships) == 1


def test_materialize_with_no_selection_is_empty():
    relevant = SchemaExpander().materialize([], [], _shop_schema())

    assert relevant.objects == []
    assert relevant.relationships == []
