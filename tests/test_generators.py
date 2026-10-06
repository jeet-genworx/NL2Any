"""Tests for database-specific query generators in V2."""

import pytest
from backend.src.schemas.pipeline import GeneratorMissingInfo, MongoQuery, QueryPlan, RelevantSchema
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


class MockPostgresMissingInfoProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return """
        ```json
        {
          "status": "missing_information",
          "missing_tables": ["order_items"],
          "reason": "Need line item table to calculate total."
        }
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
async def test_postgres_generator_direct():
    generator = PostgresQueryGenerator(provider=MockPostgresProvider())
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text"), Field(name="city", type="text")],
            )
        ]
    )

    result = await generator.generate_query("Show customers", schema)
    assert result.database_type == DatabaseType.POSTGRESQL
    assert "SELECT id, name, city FROM customers WHERE city = 'Chennai';" in result.raw_query


@pytest.mark.asyncio
async def test_postgres_generator_missing_information():
    generator = PostgresQueryGenerator(provider=MockPostgresMissingInfoProvider())
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int")],
            )
        ]
    )

    result = await generator.generate_query("Show customer order total", schema)
    assert isinstance(result, GeneratorMissingInfo)
    assert result.status == "missing_information"
    assert "order_items" in result.missing_tables


@pytest.mark.asyncio
async def test_mongo_generator_typed_output():
    generator = MongoQueryGenerator(provider=MockMongoProvider())
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.COLLECTION,
                fields=[Field(name="_id", type="string"), Field(name="name", type="string")],
            )
        ]
    )

    result = await generator.generate_query("Show customers in Chennai", schema)
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
    result = await generator.generate_query("Show customers", schema, feedback=dirty_feedback)

    # Prompt sent to model must have cleaned feedback
    assert "\x1b" not in provider.recorded_prompt
    assert "[4m" not in provider.recorded_prompt
    assert "[0m" not in provider.recorded_prompt

    # Output SQL must be clean
    assert "\x1b" not in result.raw_query


class MockMongoPipelineProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return """```json
        {
          "operation": "aggregate",
          "collection": "agents",
          "pipeline": [
            {
              "$match": {"team": "AB"}
            },
            {
              "$lookup": {
                "from": "agent_health_logs",
                "localField": "agent_id",
                "foreignField": "agent_id",
                "pipeline": [
                  { "$match": { "is_healthy": false } }
                ],
                "as": "unhealthy_logs"
              }
            }
          ]
        }
        ```"""


@pytest.mark.asyncio
async def test_mongo_generator_aggregate_output():
    """Mongo generator produces typed MongoQuery aggregate pipeline directly from SLM output."""
    generator = MongoQueryGenerator(provider=MockMongoPipelineProvider())
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="agents",
                kind=SchemaObjectKind.COLLECTION,
                fields=[Field(name="team", type="string"), Field(name="agent_id", type="string")],
            ),
            SchemaObject(
                name="agent_health_logs",
                kind=SchemaObjectKind.COLLECTION,
                fields=[Field(name="agent_id", type="string"), Field(name="is_healthy", type="bool")],
            ),
        ]
    )
    result = await generator.generate_query("Find agents in team AB", schema)
    assert result.database_type == DatabaseType.MONGODB
    pipeline = result.raw_query.pipeline
    assert len(pipeline) == 2
    assert "$match" in pipeline[0]
    lookup_stage = pipeline[1]["$lookup"]
    sub_pipeline = lookup_stage["pipeline"]
    assert len(sub_pipeline) == 1
    assert "$match" in sub_pipeline[0]



