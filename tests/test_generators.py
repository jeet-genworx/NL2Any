"""Tests for database-specific query generators."""

import pytest
from query_processing.models.pipeline import MongoQuery, QueryPlan, RelevantSchema
from query_processing.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from query_processing.pipeline.generators.mongo import MongoQueryGenerator
from query_processing.pipeline.generators.postgres import PostgresQueryGenerator


class MockPostgresProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return """
        <think>Generating SQL</think>
        ```sql
        SELECT id, name, city FROM customers WHERE city = 'Chennai';
        ```
        """


class MockMongoProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return """
        <think>Generating Mongo</think>
        ```json
        {
          "operation": "find",
          "collection": "customers",
          "filter": {"address.city": "Chennai"},
          "limit": 50
        }
        ```
        """


@pytest.mark.asyncio
async def test_postgres_generator():
    generator = PostgresQueryGenerator(provider=MockPostgresProvider())
    plan = QueryPlan(sources=["customers"], projections=["id", "name", "city"])
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text"), Field(name="city", type="text")],
            )
        ]
    )

    result = await generator.generate_query("Show customers", plan, schema)
    assert result.database_type == DatabaseType.POSTGRESQL
    assert "SELECT id, name, city FROM customers WHERE city = 'Chennai';" in result.raw_query


@pytest.mark.asyncio
async def test_mongo_generator_typed_output():
    generator = MongoQueryGenerator(provider=MockMongoProvider())
    plan = QueryPlan(sources=["customers"], projections=["*"])
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.COLLECTION,
                fields=[Field(name="_id", type="string"), Field(name="name", type="string")],
            )
        ]
    )

    result = await generator.generate_query("Show customers in Chennai", plan, schema)
    assert result.database_type == DatabaseType.MONGODB
    assert isinstance(result.raw_query, MongoQuery)
    assert result.raw_query.operation == "find"
    assert result.raw_query.collection == "customers"
    assert result.raw_query.filter == {"address.city": "Chennai"}
    assert result.raw_query.limit == 50
