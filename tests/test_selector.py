"""Tests for V2 TableSelector stage with complete schema description."""

import json
import pytest
from backend.src.schemas.pipeline import CandidateTable, QuestionAnalysis
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.core.query_processing.pipeline.selector import (
    TableSelector,
    format_schema_description,
    package_selected_schema,
)


class MockSelectorProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""
        self.last_kwargs = {}

    async def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self.response


@pytest.fixture
def sample_schema() -> DatabaseSchema:
    cust_obj = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        description="Customer records and profiles.",
        fields=[
            Field(name="id", type="int", description="Primary key"),
            Field(name="city", type="varchar", description="City name"),
        ],
    )
    orders_obj = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        description="Customer orders.",
        fields=[
            Field(name="id", type="int", description="Order ID"),
            Field(name="customer_id", type="int", description="FK to customers"),
        ],
    )
    items_obj = SchemaObject(
        name="order_items",
        kind=SchemaObjectKind.TABLE,
        description="Line items for orders.",
        fields=[
            Field(name="id", type="int", description="Item ID"),
            Field(name="order_id", type="int", description="FK to orders"),
        ],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[cust_obj, orders_obj, items_obj],
        relationships=[
            Relationship(
                from_object="orders",
                from_field="customer_id",
                to_object="customers",
                to_field="id",
                relationship_type="many_to_one",
            ),
            Relationship(
                from_object="order_items",
                from_field="order_id",
                to_object="orders",
                to_field="id",
                relationship_type="many_to_one",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_table_selector_valid_selection(sample_schema):
    response_json = json.dumps({
        "selected_tables": ["customers", "orders"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Both tables are needed to show customer orders.",
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)

    res = await selector.select(
        question="Show customer orders",
        schema=sample_schema,
    )

    assert res.selected_objects == ["customers", "orders"]
    assert res.sufficient is True
    assert res.missing_objects == []
    assert "Complete Database Schema:" in provider.last_prompt
    assert "customers" in provider.last_prompt
    assert "orders" in provider.last_prompt


@pytest.mark.asyncio
async def test_table_selector_filters_hallucinations(sample_schema):
    # Model attempts to hallucinate 'fake_table'
    response_json = json.dumps({
        "selected_tables": ["customers", "fake_table"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Selected customer data",
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)

    res = await selector.select(
        question="Show customers",
        schema=sample_schema,
    )

    assert "customers" in res.selected_objects
    assert "fake_table" not in res.selected_objects


@pytest.mark.asyncio
async def test_table_selector_passes_retry_feedback(sample_schema):
    response_json = json.dumps({
        "selected_tables": ["customers", "orders", "order_items"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Added order_items based on generator feedback.",
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)

    feedback_msg = "Query Generator reported missing line item details."
    await selector.select(
        question="Show order items",
        schema=sample_schema,
        feedback=feedback_msg,
    )

    assert "PREVIOUS ATTEMPT FEEDBACK (RETRY)" in provider.last_prompt
    assert feedback_msg in provider.last_prompt


@pytest.mark.asyncio
async def test_table_selector_empty_schema():
    empty_schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="empty",
        objects=[],
    )
    selector = TableSelector(provider=None)
    res = await selector.select(question="Show data", schema=empty_schema)

    assert res.selected_objects == []
    assert res.sufficient is False


def test_package_selected_schema(sample_schema):
    relevant = package_selected_schema(sample_schema, ["customers", "orders"])

    assert len(relevant.objects) == 2
    obj_names = {o.name for o in relevant.objects}
    assert obj_names == {"customers", "orders"}
    # Check that relationship connecting customers and orders is included
    assert len(relevant.relationships) == 1
    rel = relevant.relationships[0]
    assert rel.from_object == "orders"
    assert rel.to_object == "customers"
    # Order items relationship should NOT be included since order_items was not selected
    assert not any(r.from_object == "order_items" for r in relevant.relationships)


def test_format_schema_description(sample_schema):
    desc = format_schema_description(sample_schema)

    assert "shop (postgresql)" in desc
    assert "customers" in desc
    assert "Customer records and profiles." in desc
    assert "orders.customer_id -> customers.id" in desc
