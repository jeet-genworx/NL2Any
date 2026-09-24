"""Tests for TOML serialization and deserialization."""

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from ingestion.schema.toml_store import (
    dump_schema_to_toml,
    load_schema_from_toml,
)


def test_postgresql_toml_roundtrip(tmp_path):
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        schema_version="1.0",
        last_updated="2026-09-23T18:00:00Z",
        description_generated_at="2026-09-23T18:05:00Z",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                description="Registered customer directory",
                fields=[
                    Field(name="id", type="integer", nullable=False, description="Primary key"),
                    Field(name="name", type="varchar(100)", nullable=False, description="Customer name"),
                    Field(name="city", type="varchar(50)", nullable=True, sample_values=["NYC", "Chicago"]),
                ],
            ),
            SchemaObject(
                name="orders",
                kind=SchemaObjectKind.TABLE,
                description="Customer orders",
                fields=[
                    Field(name="id", type="integer", nullable=False, description="Order identifier"),
                    Field(name="customer_id", type="integer", nullable=False, description="Customer FK"),
                    Field(name="total", type="numeric(10,2)", nullable=False),
                ],
            ),
        ],
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

    toml_str = dump_schema_to_toml(schema)
    assert "[database]" in toml_str
    assert 'type = "postgresql"' in toml_str
    assert "[tables.customers]" in toml_str
    assert 'description = "Registered customer directory"' in toml_str
    assert "[[relationships]]" in toml_str

    # Deserialize back
    loaded = load_schema_from_toml(toml_str)
    assert loaded.database_type == DatabaseType.POSTGRESQL
    assert loaded.database_name == "shop_db"
    assert len(loaded.objects) == 2
    assert len(loaded.relationships) == 1

    cust = loaded.get_object("customers")
    assert cust is not None
    assert cust.description == "Registered customer directory"
    city = cust.get_field("city")
    assert city is not None
    assert city.sample_values == ["NYC", "Chicago"]

    rel = loaded.relationships[0]
    assert rel.from_object == "orders"
    assert rel.from_field == "customer_id"
    assert rel.to_object == "customers"
    assert rel.to_field == "id"


def test_mongodb_toml_roundtrip_with_nested_fields():
    item_pid = Field(name="product_id", type="string", description="ID of product")
    item_qty = Field(name="quantity", type="integer", description="Quantity ordered")
    items_field = Field(name="items", type="array[object]", nested=[item_pid, item_qty])

    orders_coll = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.COLLECTION,
        description="Orders collection with embedded items",
        fields=[
            Field(name="_id", type="string", description="Order ID"),
            Field(name="customer_id", type="string"),
            items_field,
        ],
    )

    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop_mongo",
        objects=[orders_coll],
        relationships=[],
    )

    toml_str = dump_schema_to_toml(schema)
    assert "[collections.orders]" in toml_str
    assert "fields.items" in toml_str

    loaded = load_schema_from_toml(toml_str)
    assert loaded.database_type == DatabaseType.MONGODB
    obj = loaded.get_object("orders")
    assert obj is not None
    items = obj.get_field("items")
    assert items is not None
    assert len(items.nested) == 2
    assert obj.get_field("items.product_id") is not None
