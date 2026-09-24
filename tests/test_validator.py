"""Tests for QueryValidator and deterministic schema AST verification."""

import pytest
from query_processing.models.pipeline import GeneratedQuery, QueryPlan, RelevantSchema
from query_processing.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from query_processing.pipeline.validator import QueryValidator, validate_sql_schema_references


def test_deterministic_sql_schema_validation_success():
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text"), Field(name="city", type="text")],
            )
        ]
    )
    sql = "SELECT id, name FROM customers WHERE city = 'Boston';"
    issues = validate_sql_schema_references(sql, schema)
    assert issues == []


def test_deterministic_sql_schema_validation_unknown_table():
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int")],
            )
        ]
    )
    sql = "SELECT id FROM orders;"
    issues = validate_sql_schema_references(sql, schema)
    assert any("orders" in i and "does not exist in relevant schema" in i for i in issues)


def test_deterministic_sql_schema_validation_unknown_column():
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text")],
            )
        ]
    )
    sql = "SELECT id, fake_column FROM customers;"
    issues = validate_sql_schema_references(sql, schema)
    assert any("fake_column" in i and "does not exist" in i for i in issues)


class MockValidatorProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return '{"valid": true, "issues": [], "suggestion": null}'


@pytest.mark.asyncio
async def test_query_validator_full_flow():
    validator = QueryValidator(provider=MockValidatorProvider())
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text")],
            )
        ]
    )
    plan = QueryPlan(sources=["customers"], projections=["id", "name"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT id, name FROM customers;",
        formatted_query="SELECT id, name FROM customers;",
    )

    result = await validator.validate("Show customers", plan, query, schema)
    assert result.valid is True
    assert result.issues == []
