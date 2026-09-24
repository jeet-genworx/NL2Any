"""Tests for QueryValidator and error type classification."""

import pytest
from query_processing.models.pipeline import (
    GeneratedQuery,
    QueryPlan,
    RelevantSchema,
    ValidationErrorType,
)
from query_processing.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from query_processing.pipeline.validator import QueryValidator, validate_sql_schema_references


class MockValidatorProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response


@pytest.fixture
def sample_schema() -> RelevantSchema:
    return RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[
                    Field(name="id", type="int"),
                    Field(name="name", type="text"),
                    Field(name="city", type="text"),
                ],
            )
        ]
    )


def test_deterministic_sql_schema_validation_success(sample_schema):
    sql = "SELECT id, name FROM customers WHERE city = 'Boston';"
    issues = validate_sql_schema_references(sql, sample_schema)
    assert issues == []


def test_deterministic_sql_schema_validation_unknown_table(sample_schema):
    sql = "SELECT id FROM orders;"
    issues = validate_sql_schema_references(sql, sample_schema)
    assert any("orders" in i and "does not exist in relevant schema" in i for i in issues)


def test_deterministic_sql_schema_validation_unknown_column(sample_schema):
    sql = "SELECT id, fake_column FROM customers;"
    issues = validate_sql_schema_references(sql, sample_schema)
    assert any("fake_column" in i and "does not exist" in i for i in issues)


@pytest.mark.asyncio
async def test_query_validator_valid_query(sample_schema):
    provider = MockValidatorProvider('{"valid": true, "error_type": "VALID", "issues": [], "suggestion": null}')
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["customers"], projections=["id", "name"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT id, name FROM customers WHERE city = 'Chennai';",
        formatted_query="SELECT id, name FROM customers WHERE city = 'Chennai';",
    )

    result = await validator.validate("Show customers in Chennai", plan, query, sample_schema)
    assert result.valid is True
    assert result.error_type == ValidationErrorType.VALID
    assert result.issues == []


@pytest.mark.asyncio
async def test_query_validator_syntax_error(sample_schema):
    # Malformed SQL that fails SQL parsing
    validator = QueryValidator(provider=None)
    plan = QueryPlan(sources=["customers"], projections=["id"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT FROM WHERE customers;;; SELECT",
        formatted_query="SELECT FROM WHERE customers;;; SELECT",
    )

    result = await validator.validate("Show customers", plan, query, sample_schema)
    assert result.valid is False
    assert result.error_type == ValidationErrorType.SYNTAX_ERROR
    assert len(result.issues) > 0


@pytest.mark.asyncio
async def test_query_validator_planner_error(sample_schema):
    # SLM validator diagnoses that the planner selected the wrong source or aggregation
    resp = '{"valid": false, "error_type": "PLANNER_ERROR", "issues": ["Query plan should aggregate orders rather than customers."], "suggestion": "Re-plan to source from orders."}'
    provider = MockValidatorProvider(resp)
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["customers"], projections=["id"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT id FROM customers;",
        formatted_query="SELECT id FROM customers;",
    )

    result = await validator.validate("Show orders count", plan, query, sample_schema)
    assert result.valid is False
    assert result.error_type == ValidationErrorType.PLANNER_ERROR
    assert "Query plan should aggregate orders" in result.issues[0]


@pytest.mark.asyncio
async def test_query_validator_unsafe_query(sample_schema):
    # Query contains destructive command
    validator = QueryValidator(provider=None)
    plan = QueryPlan(sources=["customers"], projections=["id"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="DROP TABLE customers;",
        formatted_query="DROP TABLE customers;",
    )

    result = await validator.validate("Drop all customers", plan, query, sample_schema)
    assert result.valid is False
    assert result.error_type == ValidationErrorType.UNSAFE
    assert any("Unsafe" in i or "destructive" in i for i in result.issues)


@pytest.mark.asyncio
async def test_query_validator_generation_error(sample_schema):
    resp = '{"valid": false, "error_type": "GENERATION_ERROR", "issues": ["SELECT clause omitted required alias."], "suggestion": "Add alias count_id."}'
    provider = MockValidatorProvider(resp)
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["customers"], projections=["id"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT id FROM customers;",
        formatted_query="SELECT id FROM customers;",
    )

    result = await validator.validate("Show customer ids", plan, query, sample_schema)
    assert result.valid is False
    assert result.error_type == ValidationErrorType.GENERATION_ERROR
    assert "alias" in result.issues[0]
