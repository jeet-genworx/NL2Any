"""Tests for QueryPlanner stage."""

import pytest
from query_processing.models.pipeline import QuestionAnalysis, RelevantSchema
from query_processing.models.schema import DatabaseType, Field, SchemaObject, SchemaObjectKind
from query_processing.pipeline.planner import QueryPlanner


class MockPlannerProvider:
    def __init__(self, response: str | None = None) -> None:
        self.response = response or """
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
        self.last_prompt = ""

    async def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
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


@pytest.mark.asyncio
async def test_query_planner_valid_plan(sample_schema):
    planner = QueryPlanner(provider=MockPlannerProvider())
    qa = QuestionAnalysis(question="Show customers in Chennai")

    plan = await planner.plan(qa, sample_schema, DatabaseType.POSTGRESQL)
    assert plan.sources == ["customers"]
    assert len(plan.filters) == 1
    assert plan.filters[0].field == "city"
    assert plan.filters[0].operator == "equals"
    assert plan.filters[0].value == "Chennai"
    assert plan.limit == 50


@pytest.mark.asyncio
async def test_query_planner_unknown_table(sample_schema):
    # Model generates plan with an unknown source table 'orders'
    raw_response = """
    ```json
    {
      "operation": "select",
      "sources": ["unknown_table"],
      "projections": ["*"],
      "filters": []
    }
    ```
    """
    planner = QueryPlanner(provider=MockPlannerProvider(raw_response))
    qa = QuestionAnalysis(question="Show unknown")

    plan = await planner.plan(qa, sample_schema, DatabaseType.POSTGRESQL)
    # The unknown source table must be filtered, falling back to valid schema object
    assert "unknown_table" not in plan.sources
    assert plan.sources == ["customers"]


@pytest.mark.asyncio
async def test_query_planner_unknown_column(sample_schema):
    # Model generates plan with a filter on non-existent column 'salary'
    raw_response = """
    ```json
    {
      "operation": "select",
      "sources": ["customers"],
      "projections": ["id", "name"],
      "filters": [
        {
          "field": "salary",
          "operator": "greater_than",
          "value": 50000
        },
        {
          "field": "city",
          "operator": "equals",
          "value": "Chennai"
        }
      ]
    }
    ```
    """
    planner = QueryPlanner(provider=MockPlannerProvider(raw_response))
    qa = QuestionAnalysis(question="Show customers with high salary")

    plan = await planner.plan(qa, sample_schema, DatabaseType.POSTGRESQL)
    # Filter on unknown column 'salary' must be dropped, valid filter 'city' preserved
    assert len(plan.filters) == 1
    assert plan.filters[0].field == "city"


@pytest.mark.asyncio
async def test_query_planner_no_sql_syntax(sample_schema):
    # Model mistakenly includes SQL syntax in fields (e.g. 'WHERE city', 'ORDER BY id')
    raw_response = """
    ```json
    {
      "operation": "select",
      "sources": ["customers"],
      "projections": ["name"],
      "filters": [
        {
          "field": "WHERE city",
          "operator": "equals",
          "value": "Chennai"
        }
      ],
      "group_by": ["GROUP BY city"],
      "order_by": [
        {
          "field": "ORDER BY id",
          "direction": "asc"
        }
      ]
    }
    ```
    """
    planner = QueryPlanner(provider=MockPlannerProvider(raw_response))
    qa = QuestionAnalysis(question="Show customers")

    plan = await planner.plan(qa, sample_schema, DatabaseType.POSTGRESQL)
    # SQL keywords must be stripped
    assert plan.filters[0].field == "city"
    assert plan.group_by == ["city"]
    assert plan.order_by[0].field == "id"


@pytest.mark.asyncio
async def test_query_planner_with_feedback(sample_schema):
    provider = MockPlannerProvider()
    planner = QueryPlanner(provider=provider)
    qa = QuestionAnalysis(question="Show customers")

    feedback_msg = "Previous plan was missing city projection."
    await planner.plan(qa, sample_schema, DatabaseType.POSTGRESQL, feedback=feedback_msg)

    assert "ATTENTION - PREVIOUS PLAN FAILED VALIDATION (RETRY)" in provider.last_prompt
    assert feedback_msg in provider.last_prompt
