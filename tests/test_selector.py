"""Tests for TableSelector stage."""

import json
import pytest
from query_processing.models.pipeline import CandidateTable, QuestionAnalysis
from query_processing.models.schema import DatabaseSchema, DatabaseType, Field, SchemaObject, SchemaObjectKind
from query_processing.pipeline.selector import TableSelector


class MockSelectorProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    async def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        return self.response


@pytest.fixture
def sample_schema() -> DatabaseSchema:
    cust_obj = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="int"), Field(name="city", type="varchar")],
    )
    orders_obj = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="int"), Field(name="customer_id", type="int")],
    )
    items_obj = SchemaObject(
        name="order_items",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="int"), Field(name="order_id", type="int")],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[cust_obj, orders_obj, items_obj],
    )


@pytest.mark.asyncio
async def test_table_selector_valid_selection(sample_schema):
    candidates = [
        CandidateTable(table_name="customers", similarity=0.92, rank=1),
        CandidateTable(table_name="orders", similarity=0.88, rank=2),
    ]

    response_json = json.dumps({
        "selected_objects": ["customers", "orders"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Both tables are needed to show customer orders.",
        "retrieval_hint": None,
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show customer orders", subjective=["customers", "orders"])

    res = await selector.select(qa, candidates, sample_schema)

    assert res.selected_objects == ["customers", "orders"]
    assert res.sufficient is True
    assert res.missing_objects == []


@pytest.mark.asyncio
async def test_table_selector_filters_hallucinations(sample_schema):
    candidates = [
        CandidateTable(table_name="customers", similarity=0.92, rank=1),
        CandidateTable(table_name="orders", similarity=0.85, rank=2),
    ]

    # Model attempts to hallucinate 'fake_table'
    response_json = json.dumps({
        "selected_objects": ["customers", "fake_table"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Selected customer data",
        "retrieval_hint": None,
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show customers", subjective=["customers"])

    res = await selector.select(qa, candidates, sample_schema)

    assert "customers" in res.selected_objects
    assert "fake_table" not in res.selected_objects


@pytest.mark.asyncio
async def test_table_selector_insufficient_selection(sample_schema):
    candidates = [
        CandidateTable(table_name="customers", similarity=0.89, rank=1),
    ]

    response_json = json.dumps({
        "selected_objects": ["customers"],
        "sufficient": False,
        "missing_objects": ["orders"],
        "reason": "Missing orders table needed for order count.",
        "retrieval_hint": "tables containing customer order records and dates",
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="How many orders did each customer place?", subjective=["orders"])

    res = await selector.select(qa, candidates, sample_schema)

    assert res.sufficient is False
    assert "orders" in res.missing_objects
    assert res.retrieval_hint == "tables containing customer order records and dates"


@pytest.mark.asyncio
async def test_table_selector_passes_retry_feedback(sample_schema):
    candidates = [
        CandidateTable(table_name="customers", similarity=0.90, rank=1),
        CandidateTable(table_name="orders", similarity=0.85, rank=2),
    ]

    response_json = json.dumps({
        "selected_objects": ["customers", "orders"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Both tables present now.",
        "retrieval_hint": None,
    })

    provider = MockSelectorProvider(response_json)
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show orders", subjective=["orders"])

    feedback_msg = "Previous attempt was missing orders."
    await selector.select(qa, candidates, sample_schema, feedback=feedback_msg)

    assert "PREVIOUS SELECTION FEEDBACK (RETRY)" in provider.last_prompt
    assert feedback_msg in provider.last_prompt


@pytest.mark.asyncio
async def test_table_selector_zero_candidates(sample_schema):
    selector = TableSelector(provider=None)
    qa = QuestionAnalysis(question="Show astronauts", subjective=["astronauts"])
    res = await selector.select(qa, [], sample_schema)

    assert res.selected_objects == []
    assert res.sufficient is False
