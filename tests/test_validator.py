"""Tests for QueryValidator and error type classification."""

import pytest
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    QueryPlan,
    RelevantSchema,
    ValidationErrorType,
)
from backend.src.data.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from backend.src.core.query_processing.pipeline.validator import QueryValidator, validate_sql_schema_references


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
    # SLM validator identifies malformed SQL syntax
    resp = '{"valid": false, "error_type": "SYNTAX_ERROR", "issues": ["PostgreSQL syntax error near WHERE"], "suggestion": "Correct the SQL syntax."}'
    provider = MockValidatorProvider(resp)
    validator = QueryValidator(provider=provider)
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


def test_deterministic_sql_schema_validation_projected_alias():
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="products",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text")],
            ),
            SchemaObject(
                name="order_items",
                kind=SchemaObjectKind.TABLE,
                fields=[
                    Field(name="id", type="int"),
                    Field(name="product_id", type="int"),
                    Field(name="quantity", type="int"),
                    Field(name="unit_price", type="numeric"),
                ],
            ),
        ]
    )
    sql = """
    SELECT p.id, p.name, SUM(oi.quantity * oi.unit_price) AS total_sales
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    GROUP BY p.id, p.name
    ORDER BY total_sales DESC
    LIMIT 10;
    """
    issues = validate_sql_schema_references(sql, schema)
    assert issues == []


def test_deterministic_sql_schema_validation_qualified_table():
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="finopsiq.customers",
                kind=SchemaObjectKind.TABLE,
                fields=[
                    Field(name="customer_id", type="uuid"),
                    Field(name="name", type="text"),
                ],
            )
        ]
    )
    # Qualified table name in SQL
    sql1 = "SELECT COUNT(customer_id) AS total_customers FROM finopsiq.customers;"
    assert validate_sql_schema_references(sql1, schema) == []

    # Unqualified table name in SQL
    sql2 = "SELECT customer_id, name FROM customers;"
    assert validate_sql_schema_references(sql2, schema) == []

    # Aliased qualified table name in SQL
    sql3 = "SELECT c.customer_id FROM finopsiq.customers AS c;"
    assert validate_sql_schema_references(sql3, schema) == []

    # Unknown table
    sql4 = "SELECT id FROM orders;"
    assert len(validate_sql_schema_references(sql4, schema)) > 0


@pytest.mark.asyncio
async def test_query_validator_nested_query_equal_operator():
    """Validator properly evaluates nested queries that return multiple results with '=' instead of 'IN'."""
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="finopsiq.invoices",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="invoice_id", type="uuid"), Field(name="invoice_number", type="text")],
            ),
            SchemaObject(
                name="finopsiq.invoice_line_items",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="invoice_id", type="uuid"), Field(name="line_total", type="numeric")],
            ),
        ]
    )
    slm_resp = """```json
    {
      "valid": false,
      "error_type": "GENERATION_ERROR",
      "issues": [
        "Nested subquery returns a group of invoice IDs but uses '=' instead of 'IN' or a JOIN."
      ],
      "suggestion": "Replace '=' with 'IN' or rewrite the subquery using an explicit JOIN between finopsiq.invoices and finopsiq.invoice_line_items on invoice_id."
    }
    ```"""
    provider = MockValidatorProvider(slm_resp)
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["finopsiq.invoice_line_items", "finopsiq.invoices"], projections=["line_total"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT line_total FROM finopsiq.invoice_line_items WHERE invoice_id = (SELECT invoice_id FROM finopsiq.invoices);",
        formatted_query="SELECT line_total FROM finopsiq.invoice_line_items WHERE invoice_id = (SELECT invoice_id FROM finopsiq.invoices);",
    )

    res = await validator.validate("Show line items for all invoices", plan, query, schema)
    assert res.valid is False
    assert res.error_type == ValidationErrorType.GENERATION_ERROR
    assert "uses '=' instead of 'IN'" in res.issues[0]
    assert "Replace '=' with 'IN'" in res.suggestion


@pytest.mark.asyncio
async def test_query_validator_subquery_column_scoping():
    """Validator properly flags selecting columns belonging to a table only inside a nested subquery."""
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="finopsiq.invoices",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="invoice_id", type="uuid"), Field(name="invoice_number", type="text")],
            ),
            SchemaObject(
                name="finopsiq.invoice_line_items",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="invoice_id", type="uuid"), Field(name="item_name", type="text")],
            ),
        ]
    )
    slm_resp = """```json
    {
      "valid": false,
      "error_type": "GENERATION_ERROR",
      "issues": [
        "Column 'invoice_number' belongs to 'finopsiq.invoices', but 'finopsiq.invoices' is only referenced in a nested subquery and not in the FROM/JOIN clause."
      ],
      "suggestion": "Use an explicit JOIN with 'finopsiq.invoices' in the FROM clause so 'invoice_number' can be selected."
    }
    ```"""
    provider = MockValidatorProvider(slm_resp)
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["finopsiq.invoice_line_items", "finopsiq.invoices"], projections=["invoice_number", "item_name"])
    # Query attempts to select invoice_number from invoice_line_items with a subquery
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT invoice_number, item_name FROM finopsiq.invoice_line_items WHERE invoice_id IN (SELECT invoice_id FROM finopsiq.invoices);",
        formatted_query="SELECT invoice_number, item_name FROM finopsiq.invoice_line_items WHERE invoice_id IN (SELECT invoice_id FROM finopsiq.invoices);",
    )

    res = await validator.validate("Show invoice numbers and line item names", plan, query, schema)
    assert res.valid is False
    assert any("invoice_number" in i and "subquery" in i for i in res.issues)
    assert "JOIN" in res.suggestion


@pytest.mark.asyncio
async def test_query_validator_incomplete_placeholder_detection():
    """Queries containing incomplete placeholders like '...' produce clean, readable issues without ANSI codes."""
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="finopsiq.invoice_line_items",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="invoice_id", type="uuid")],
            )
        ]
    )
    slm_resp = """```json
    {
      "valid": false,
      "error_type": "SYNTAX_ERROR",
      "issues": [
        "Query contains incomplete placeholder '...' in WHERE clause near invoice_id = (SELECT invoice_id FROM finopsiq.invoices WHERE ...);"
      ],
      "suggestion": "Replace the incomplete subquery with an explicit JOIN between finopsiq.invoice_line_items and finopsiq.invoices on invoice_id."
    }
    ```"""
    provider = MockValidatorProvider(slm_resp)
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["finopsiq.invoice_line_items"], projections=["*"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT * FROM finopsiq.invoice_line_items WHERE invoice_id = (SELECT invoice_id FROM finopsiq.invoices WHERE ...);",
        formatted_query="SELECT * FROM finopsiq.invoice_line_items WHERE invoice_id = (SELECT invoice_id FROM finopsiq.invoices WHERE ...);",
    )

    res = await validator.validate("Show line items", plan, query, schema)
    assert res.valid is False
    assert res.error_type == ValidationErrorType.SYNTAX_ERROR
    # Must NOT contain raw ANSI escapes like \x1b[4m or [4m
    assert "\x1b" not in res.issues[0]
    assert "[4m" not in res.issues[0]
    assert "[0m" not in res.issues[0]
    assert any("placeholder" in i.lower() or "syntax" in i.lower() for i in res.issues)
    assert res.suggestion is not None
    assert "JOIN" in res.suggestion


class FailingProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("SLM connection failed")


@pytest.mark.asyncio
async def test_query_validator_lenient_fallback_on_slm_failure():
    """When SLM provider fails, validator is lenient and does NOT fail queries deterministically."""
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="finopsiq.invoices",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="invoice_id", type="uuid")],
            )
        ]
    )
    validator = QueryValidator(provider=FailingProvider())
    plan = QueryPlan(sources=["finopsiq.invoices"], projections=["invoice_id"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT invoice_id FROM finopsiq.invoices WHERE contract_number = 'CTR-001';",
        formatted_query="SELECT invoice_id FROM finopsiq.invoices WHERE contract_number = 'CTR-001';",
    )

    res = await validator.validate("Show invoice CTR-001", plan, query, schema)
    # Lenient fallback accepts the query rather than inventing deterministic syntax errors
    assert res.valid is True
    assert res.error_type == ValidationErrorType.VALID
    assert res.issues == []


@pytest.mark.asyncio
async def test_query_validator_strips_ansi_escapes_from_slm_output():
    """Any ANSI escape sequences or control characters in SLM response are stripped."""
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int")],
            )
        ]
    )
    dirty_slm_resp = (
        '{"valid": false, "error_type": "SYNTAX_ERROR", '
        '"issues": ["Syntax error near \\u001b[4m.\\u001b[0m.. with [4mcode[0m"], '
        '"suggestion": "Fix \\u001b[31merror\\u001b[0m immediately."}'
    )
    provider = MockValidatorProvider(dirty_slm_resp)
    validator = QueryValidator(provider=provider)
    plan = QueryPlan(sources=["customers"], projections=["id"])
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT id FROM customers;",
        formatted_query="SELECT id FROM customers;",
    )

    res = await validator.validate("Show customers", plan, query, schema)
    assert res.valid is False
    assert "\x1b" not in res.issues[0]
    assert "[4m" not in res.issues[0]
    assert "[0m" not in res.issues[0]
    assert "\x1b" not in res.suggestion

