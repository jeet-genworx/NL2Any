"""Tests for QueryPlanner stage."""

import pytest
from nl2anyquery.models.pipeline import QuestionAnalysis, RelevantSchema
from nl2anyquery.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from nl2anyquery.pipeline.planner import QueryPlanner


class MockPlannerProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return """
        <think>planning</think>
        ```json
        {
          "operation": "select",
          "sources": ["customers"],
          "projections": ["id", "name", "city"],
          "filters": [
            {
              "field": "city",
              "operator": "equals",
              "value": "Chennai"
            }
          ],
          "aggregations": [],
          "group_by": [],
          "order_by": [
            {
              "field": "id",
              "direction": "asc"
            }
          ],
          "limit": 50,
          "relationships_used": []
        }
        ```
        """


@pytest.mark.asyncio
async def test_query_planner_pure_semantics():
    planner = QueryPlanner(provider=MockPlannerProvider())
    qa = QuestionAnalysis(question="Show customers in Chennai")
    schema = RelevantSchema(
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="int"), Field(name="name", type="text"), Field(name="city", type="text")],
            )
        ]
    )

    plan = await planner.plan(qa, schema, DatabaseType.POSTGRESQL)
    assert plan.sources == ["customers"]
    assert len(plan.filters) == 1
    assert plan.filters[0].field == "city"
    assert plan.filters[0].value == "Chennai"
    assert plan.limit == 50
