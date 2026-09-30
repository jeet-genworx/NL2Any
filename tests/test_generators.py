"""Tests for database-specific query generators."""

import pytest
from backend.src.schemas.pipeline import MongoQuery, QueryPlan, RelevantSchema
from backend.src.data.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from backend.src.core.query_processing.pipeline.generators.mongo import MongoQueryGenerator
from backend.src.core.query_processing.pipeline.generators.postgres import PostgresQueryGenerator


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


class RecordingFeedbackProvider:
    def __init__(self) -> None:
        self.recorded_prompt: str = ""

    async def generate(self, prompt: str, **kwargs) -> str:
        self.recorded_prompt = prompt
        return "\x1b[4m```sql\nSELECT * FROM \x1b[0mcustomers;"


@pytest.mark.asyncio
async def test_postgres_generator_strips_ansi_escapes():
    """Generator strips ANSI escape codes from feedback in prompt and from output SQL."""
    provider = RecordingFeedbackProvider()
    generator = PostgresQueryGenerator(provider=provider)
    plan = QueryPlan(sources=["customers"], projections=["*"])
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int")],
            )
        ]
    )

    dirty_feedback = "Syntax error near \x1b[4m.\x1b[0m.. with [4mcode[0m"
    result = await generator.generate_query("Show customers", plan, schema, feedback=dirty_feedback)

    # Prompt sent to model must have cleaned feedback
    assert "\x1b" not in provider.recorded_prompt
    assert "[4m" not in provider.recorded_prompt
    assert "[0m" not in provider.recorded_prompt

    # Output SQL must be clean
    assert "\x1b" not in result.raw_query
    assert "[4m" not in result.raw_query
    assert "SELECT * FROM customers;" in result.raw_query
