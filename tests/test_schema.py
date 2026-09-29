"""Tests for normalized schema data models and consistency validation."""

import pytest
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)


def test_schema_model_creation():
    field1 = Field(name="id", type="integer", nullable=False, sample_values=[1, 2, 3])
    field2 = Field(name="name", type="varchar(100)", nullable=False)
    obj = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        description="Customer directory",
        fields=[field1, field2],
    )
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="test_db",
        objects=[obj],
        relationships=[],
    )

    assert schema.database_type == DatabaseType.POSTGRESQL
    assert len(schema.objects) == 1
    assert schema.get_object("customers") == obj
    assert obj.get_field("id") == field1
    assert obj.all_field_paths() == ["id", "name"]


def test_nested_field_paths():
    subfield1 = Field(name="city", type="string")
    subfield2 = Field(name="zip", type="string")
    parent = Field(name="address", type="object", nested=[subfield1, subfield2])

    obj = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.COLLECTION,
        fields=[parent],
    )

    assert obj.all_field_paths() == ["address", "address.city", "address.zip"]
    assert obj.get_field("address.city") == subfield1
    assert obj.get_field("address.zip") == subfield2
    assert obj.get_field("non_existent") is None


def test_consistency_validation_success():
    customers = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="name", type="text")],
    )
    orders = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="customer_id", type="integer")],
    )
    rel = Relationship(
        from_object="orders",
        from_field="customer_id",
        to_object="customers",
        to_field="id",
    )
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[customers, orders],
        relationships=[rel],
    )
    # Should not raise
    schema.validate_consistency()


def test_consistency_validation_duplicate_objects():
    obj1 = SchemaObject(name="customers", fields=[Field(name="id", type="int")])
    obj2 = SchemaObject(name="customers", fields=[Field(name="id", type="int")])
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[obj1, obj2],
    )
    with pytest.raises(ValueError, match="duplicate object name 'customers'"):
        schema.validate_consistency()


def test_consistency_validation_duplicate_fields():
    f1 = Field(name="id", type="int")
    f2 = Field(name="id", type="varchar")
    obj = SchemaObject(name="customers", fields=[f1, f2])
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[obj],
    )
    with pytest.raises(ValueError, match="duplicate field 'id'"):
        schema.validate_consistency()


def test_consistency_validation_invalid_relationship_source():
    obj1 = SchemaObject(name="customers", fields=[Field(name="id", type="int")])
    rel = Relationship(
        from_object="non_existent",
        from_field="id",
        to_object="customers",
        to_field="id",
    )
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[obj1],
        relationships=[rel],
    )
    with pytest.raises(ValueError, match="relationship source object 'non_existent' does not exist"):
        schema.validate_consistency()


def test_consistency_validation_invalid_relationship_field():
    cust = SchemaObject(name="customers", fields=[Field(name="id", type="int")])
    orders = SchemaObject(name="orders", fields=[Field(name="order_id", type="int")])
    rel = Relationship(
        from_object="orders",
        from_field="customer_id",  # does not exist in orders
        to_object="customers",
        to_field="id",
    )
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[cust, orders],
        relationships=[rel],
    )
    with pytest.raises(ValueError, match="relationship source field 'customer_id' does not exist"):
        schema.validate_consistency()


def test_postgres_metadata_extractor_multi_schema_naming():
    from unittest.mock import MagicMock
    from backend.src.data.clients.postgres.metadata import PostgreSQLMetadataExtractor, _format_table_name

    assert _format_table_name("public", "customers") == "customers"
    assert _format_table_name("finopsiq", "email_context") == "finopsiq.email_context"
    assert _format_table_name("connector_service", "email_context") == "connector_service.email_context"

    # Mock cursor responses for multi-schema scenario
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_cursor.fetchone.return_value = ("test_finops",)
    mock_cursor.fetchall.side_effect = [
        # 1. tables
        [
            ("connector_service", "email_context"),
            ("finopsiq", "email_context"),
            ("public", "customers"),
        ],
        # 2. columns
        [
            ("connector_service", "email_context", "id", "uuid", "NO"),
            ("connector_service", "email_context", "sender", "varchar", "YES"),
            ("finopsiq", "email_context", "id", "bigint", "NO"),
            ("finopsiq", "email_context", "subject", "text", "YES"),
            ("public", "customers", "id", "integer", "NO"),
        ],
        # 3. foreign keys
        [],
    ]

    extractor = PostgreSQLMetadataExtractor(mock_conn)
    schema = extractor.extract_schema()

    assert len(schema.objects) == 3
    obj_names = {obj.name for obj in schema.objects}
    assert obj_names == {
        "connector_service.email_context",
        "finopsiq.email_context",
        "customers",
    }
    # Validate consistency passed without duplicate field 'id' error
    schema.validate_consistency()

