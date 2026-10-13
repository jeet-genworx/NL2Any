"""Tests for database-specific query generators."""

import pytest
from backend.src.schemas.pipeline import MongoQuery, QueryPlan, RelevantSchema
from backend.src.data.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from backend.src.core.query_processing.pipeline.generators.mongo import MongoQueryGenerator
from backend.src.core.query_processing.pipeline.generators.postgres import (
    PostgresQueryGenerator,
    _extract_sql,
    format_row_scope,
)


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


def test_extract_sql_ignores_a_bare_with_in_the_join_plan_notes():
    """The JOIN PLAN is prose, and prose contains the word "with".

    Observed from the live model: no code fence, and a fanout line reading
    "...handled by selecting distinct invoice records with line items..." The
    old scan matched that bare WITH and returned a query starting mid-sentence.
    """
    raw = (
        "JOIN PLAN\n"
        "base: finopsiq.invoices\n"
        "join: finopsiq.invoice_line_items via invoices.invoice_id = invoice_line_items.invoice_id [1-N]\n"
        "fanout: two 1-N joins off the same parent would multiply rows; handled by "
        "selecting distinct invoice records with line items and revisions without aggregation\n"
        "\n"
        "SELECT inv.invoice_number FROM finopsiq.invoices inv WHERE inv.is_deleted IS FALSE;"
    )

    sql = _extract_sql(raw)

    assert sql.upper().startswith("SELECT")
    assert "fanout" not in sql.lower()
    assert "line items and revisions" not in sql


def test_extract_sql_keeps_a_real_cte_after_the_join_plan():
    raw = (
        "JOIN PLAN\n"
        "base: finopsiq.invoices\n"
        "fanout: none\n"
        "\n"
        "WITH totals AS (SELECT invoice_id FROM finopsiq.invoices) "
        "SELECT invoice_id FROM totals;"
    )

    sql = _extract_sql(raw)

    assert sql.upper().startswith("WITH TOTALS")


def test_extract_sql_prefers_the_fence_over_the_join_plan():
    raw = (
        "JOIN PLAN\nbase: finopsiq.invoices\nfanout: none\n\n"
        "```sql\nSELECT inv.invoice_id FROM finopsiq.invoices inv;\n```"
    )

    assert _extract_sql(raw) == "SELECT inv.invoice_id FROM finopsiq.invoices inv;"


def _column_type(name: str) -> str:
    """A plausible declared type, so predicate choice is exercised realistically."""
    if name.startswith(("is_", "has_")):
        return "boolean"
    if name.endswith(("_at", "_on")):
        return "timestamp with time zone"
    return "text"


def _slice_with(*specs) -> RelevantSchema:
    """Build a slice where each spec is (table_name, [column names])."""
    return RelevantSchema(
        objects=[
            SchemaObject(
                name=name,
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name=c, type=_column_type(c)) for c in columns],
            )
            for name, columns in specs
        ],
        relationships=[],
    )


def test_row_scope_names_the_predicate_and_the_tables_without_it():
    """The negative list is the part that works.

    Given only a positive scope-column list, the live model joined on a tenant
    column a table did not have -- the same mistake the deletion-marker
    negative list had already stopped. So every convention gets both lists.
    """
    schema = _slice_with(
        ("app.orders", ["order_id", "tenant_id", "is_deleted"]),
        ("app.order_items", ["order_item_id", "tenant_id"]),
        ("app.order_revisions", ["revision_id"]),
    )

    scope = format_row_scope(schema)

    assert 'is_deleted IS FALSE' in scope
    assert "NO deletion marker" in scope
    assert 'scope column "tenant_id"' in scope
    assert "NO scope column" in scope
    # The revisions table has neither, so it is on both negative lines.
    for line in [l for l in scope.splitlines() if "NO " in l]:
        assert "app.order_revisions" in line
    # And on neither positive line.
    for line in [l for l in scope.splitlines() if "ONLY these" in l or "listed here" in l]:
        assert "app.order_revisions" not in line


def test_row_scope_uses_is_null_for_a_timestamp_deletion_marker():
    """A flag and a timestamp mark deletion differently.

    is_deleted is FALSE for a live row; deleted_at is NULL for one. Emitting
    the wrong predicate either errors or silently returns nothing, so the
    column's type decides.
    """
    scope = format_row_scope(
        _slice_with(("shop.carts", ["cart_id", "deleted_at"]))
    )

    assert "deleted_at IS NULL" in scope
    assert "IS FALSE" not in scope


def test_row_scope_says_a_schema_has_no_such_conventions():
    """Most databases have neither convention; none must be invented for them."""
    scope = format_row_scope(_slice_with(("shop.products", ["product_id", "product_code"])))

    assert "No table in this slice marks rows as deleted" in scope
    assert "Do NOT invent one." in scope
    assert "shop.products" not in scope


def test_row_scope_honours_configured_column_names(monkeypatch):
    """The names are a per-database convention, so they are configuration."""
    from backend.src.config import settings

    monkeypatch.setattr(settings, "soft_delete_columns", "archived_flag")
    monkeypatch.setattr(settings, "row_scope_columns", "company_id")

    scope = format_row_scope(
        _slice_with(("erp.ledger", ["ledger_id", "archived_flag", "company_id"]))
    )

    assert "archived_flag IS NULL" in scope
    assert 'scope column "company_id"' in scope
