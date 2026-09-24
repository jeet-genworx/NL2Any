"""Tests for TableSelector stage."""

import pytest
from query_processing.models.pipeline import QuestionAnalysis
from query_processing.models.schema import DatabaseSchema, DatabaseType, Field, SchemaObject, SchemaObjectKind
from query_processing.pipeline.selector import TableSelector
from query_processing.retrieval.bm25 import RetrievalResult


class MockSelectorProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response


@pytest.mark.asyncio
async def test_table_selector_filters_hallucinations():
    cust_obj = SchemaObject(name="customers", kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="int")])
    orders_obj = SchemaObject(name="orders", kind=SchemaObjectKind.TABLE, fields=[Field(name="id", type="int")])
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[cust_obj, orders_obj],
    )

    candidates = [
        RetrievalResult(object_name="customers", score=5.0, schema_object=cust_obj),
        RetrievalResult(object_name="orders", score=3.0, schema_object=orders_obj),
    ]

    # Model attempts to hallucinate 'non_existent_table'
    provider = MockSelectorProvider('{"selected_objects": ["customers", "non_existent_table"]}')
    selector = TableSelector(provider=provider)

    qa = QuestionAnalysis(question="Show customers", subjective=["customers"])
    res = await selector.select(qa, candidates, schema)

    assert "customers" in res.selected_objects
    assert "non_existent_table" not in res.selected_objects


@pytest.mark.asyncio
async def test_table_selector_zero_candidates():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[],
    )
    selector = TableSelector(provider=None)
    qa = QuestionAnalysis(question="Show astronauts", subjective=["astronauts"])
    res = await selector.select(qa, [], schema)
    assert res.selected_objects == []
