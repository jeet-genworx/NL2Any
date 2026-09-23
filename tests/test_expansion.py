"""Tests for SchemaExpander."""

from nl2anyquery.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from nl2anyquery.pipeline.expansion import SchemaExpander


def test_schema_expander_with_bridge_table():
    orders = SchemaObject(name="orders", kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="int")])
    products = SchemaObject(name="products", kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="int")])
    order_items = SchemaObject(
        name="order_items",
        kind=SchemaObjectKind.TABLE,
        fields=[
            Field(name="id", type="int"),
            Field(name="order_id", type="int"),
            Field(name="product_id", type="int"),
        ],
    )

    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[orders, products, order_items],
        relationships=[
            Relationship(from_object="order_items", from_field="order_id", to_object="orders", to_field="id"),
            Relationship(from_object="order_items", from_field="product_id", to_object="products", to_field="id"),
        ],
    )

    expander = SchemaExpander()
    # Selected only orders and products
    relevant = expander.expand(["orders", "products"], schema)

    # Bridge table order_items should automatically be included!
    rel_names = {obj.name for obj in relevant.objects}
    assert "orders" in rel_names
    assert "products" in rel_names
    assert "order_items" in rel_names
    assert len(relevant.relationships) == 2
