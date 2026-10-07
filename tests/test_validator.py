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
    # The unparseable placeholder is caught by the deterministic parser before
    # the SLM is consulted, so the suggestion is the parser's, not the model's.
    assert "syntax" in res.suggestion.lower() or "parse" in res.suggestion.lower()


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
    # Every column here exists in the schema above, so the deterministic check
    # passes and the SLM failure is what the fallback has to absorb.
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT invoice_id FROM finopsiq.invoices WHERE invoice_id = 'CTR-001';",
        formatted_query="SELECT invoice_id FROM finopsiq.invoices WHERE invoice_id = 'CTR-001';",
    )

    res = await validator.validate("Show invoice CTR-001", plan, query, schema)
    # Lenient fallback accepts the query rather than inventing semantic errors
    assert res.valid is True
    assert res.error_type == ValidationErrorType.VALID
    assert res.issues == []


@pytest.mark.asyncio
async def test_query_validator_rejects_unknown_column_without_the_slm():
    """A column that is not on the table it is qualified with is rejected
    deterministically, even when the SLM is unavailable.

    The leniency above covers judgement calls the model would have made; it must
    not extend to facts sqlglot can settle, or the query reaches the database
    only to fail there.
    """
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
    assert res.valid is False
    assert res.error_type == ValidationErrorType.GENERATION_ERROR
    assert any("contract_number" in i for i in res.issues)


@pytest.mark.asyncio
async def test_query_validator_reports_each_bad_column_once():
    """The same bad column in SELECT, WHERE and ORDER BY is one issue, not three."""
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
    sql = (
        "SELECT i.bogus FROM finopsiq.invoices i "
        "WHERE i.bogus = 'x' ORDER BY i.bogus ASC;"
    )
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL, raw_query=sql, formatted_query=sql
    )

    res = await validator.validate("q", plan, query, schema)
    assert res.valid is False
    assert len(res.issues) == 1, res.issues


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



def test_false_omission_claims_are_discarded_but_true_ones_survive():
    """The model reports columns as omitted that are in the SELECT list.

    It compares the plan's spelling (`finopsiq.products.x`) to the query's
    (`products.x`), finds a textual difference, and calls the column missing.
    Whether a column is selected is checkable, so it is checked rather than
    believed -- but a claim about a column the query really lacks must survive.
    """
    from backend.src.core.query_processing.pipeline.validator import (
        _drop_false_omission_claims,
    )

    sql = (
        "SELECT inv.invoice_id, products.product_id, products.product_code "
        "FROM finopsiq.invoice_line_items li "
        "JOIN finopsiq.invoices inv ON li.invoice_id = inv.invoice_id "
        "JOIN finopsiq.products products ON li.product_id = products.product_id;"
    )
    issues = [
        "The query does not include the 'products.product_id' column in the SELECT list.",
        "The query does not include the 'finopsiq.products.product_code' column.",
        "The query does not include the 'vendors.company_name' column in the SELECT list.",
        "The query uses '=' with a multi-row subquery, which raises a runtime error.",
    ]
    kept, discarded = _drop_false_omission_claims(issues, sql)

    assert len(discarded) == 2
    assert any("product_id" in d for d in discarded)
    # Qualifier differences must not protect a claim from being checked.
    assert any("product_code" in d for d in discarded)
    assert len(kept) == 2
    assert any("company_name" in k for k in kept)
    assert any("multi-row subquery" in k for k in kept)


def test_omission_filter_is_inert_on_unparseable_sql():
    """With no AST to check against, every issue is left alone."""
    from backend.src.core.query_processing.pipeline.validator import (
        _drop_false_omission_claims,
    )

    issues = ["The query does not include the 'products.product_id' column."]
    kept, discarded = _drop_false_omission_claims(issues, "SELECT FROM WHERE ...;")
    assert kept == issues
    assert discarded == []


def test_clause_scoped_claims_are_never_discarded():
    """A GROUP BY complaint must survive the omission filter.

    "...does not include these in a GROUP BY clause" matches the omission
    wording and names columns that are present as projections, so the presence
    check would call it false. But the claim is about WHERE the columns appear,
    and discarding it on "the column is somewhere in the query" would silence a
    real grouping bug.
    """
    from backend.src.core.query_processing.pipeline.validator import (
        _drop_false_omission_claims,
    )

    sql = (
        "SELECT products.product_code, vendors.vendor_id, invoices.invoice_id "
        "FROM finopsiq.invoice_line_items li "
        "JOIN finopsiq.invoices invoices ON li.invoice_id = invoices.invoice_id "
        "JOIN finopsiq.products products ON li.product_id = products.product_id;"
    )
    issues = [
        "The query does not group by the required fields as per the query plan. The "
        "plan specifies grouping by 'products.product_id', 'vendors.vendor_id' and "
        "'invoices.invoice_id', but the query does not include these in a GROUP BY clause.",
        "The query does not include 'invoices.invoice_id' in the ORDER BY clause.",
        "The query omits the 'products.product_code' from the HAVING clause.",
        "The query does not include the 'products.product_code' column in the SELECT list.",
    ]
    kept, discarded = _drop_false_omission_claims(issues, sql)

    # Only the plainly-false SELECT-list claim is dropped.
    assert len(discarded) == 1
    assert "SELECT list" in discarded[0]
    assert len(kept) == 3
    assert any("GROUP BY" in k for k in kept)
    assert any("ORDER BY" in k for k in kept)
    assert any("HAVING" in k for k in kept)
