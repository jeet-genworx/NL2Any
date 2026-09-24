"""Tests for end-to-end NL2AnyQueryOrchestrator with mocks."""

import pytest
from query_processing.models.pipeline import (
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    MongoQuery,
    PolicyResult,
    QueryPlan,
    QuestionAnalysis,
    RelevantSchema,
    SemanticAnalysisResult,
    TableSelectionResult,
    ValidationResult,
)
from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator


class MockGuardrail:
    def __init__(self, decision: GuardrailDecision = GuardrailDecision.READ_QUERY) -> None:
        self.decision = decision

    async def classify(self, question: str) -> GuardrailResult:
        return GuardrailResult(decision=self.decision, reason="Mock decision")

    def handle_basic_question(self, question: str, database_type: str, database_name: str) -> str:
        return "I am NL2AnyQuery."


class MockSemantic:
    async def analyze(self, question: str) -> SemanticAnalysisResult:
        return SemanticAnalysisResult(subjective=["customers"], objective=["all"])


class MockSelector:
    def __init__(self, selected: list[str] | None = None) -> None:
        self.selected = selected if selected is not None else ["customers"]

    async def select(self, *args, **kwargs) -> TableSelectionResult:
        return TableSelectionResult(selected_objects=self.selected)


class MockPlanner:
    async def plan(self, *args, **kwargs) -> QueryPlan:
        return QueryPlan(sources=["customers"], projections=["id", "name"])


class MockPostgresGen:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate_query(self, *args, **kwargs) -> GeneratedQuery:
        self.call_count += 1
        return GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query="SELECT id, name FROM customers;",
            formatted_query="SELECT id, name FROM customers;",
        )


class MockValidator:
    def __init__(self, fail_first_n: int = 0) -> None:
        self.fail_first_n = fail_first_n
        self.attempts = 0

    async def validate(self, *args, **kwargs) -> ValidationResult:
        self.attempts += 1
        if self.attempts <= self.fail_first_n:
            return ValidationResult(valid=False, issues=["Syntax issue"], suggestion="Fix syntax")
        return ValidationResult(valid=True, issues=[], suggestion=None)


class MockPolicy:
    def check(self, *args, **kwargs) -> PolicyResult:
        return PolicyResult(allowed=True, reason="Verified safe")


class MockExecutor:
    def execute(self, *args, **kwargs):
        return ["id", "name"], [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@pytest.fixture
def test_schema() -> DatabaseSchema:
    cust = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="name", type="varchar")],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="test_shop",
        objects=[cust],
    )


@pytest.mark.asyncio
async def test_orchestrator_happy_path(test_schema):
    gen = MockPostgresGen()
    val = MockValidator(fail_first_n=0)
    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector(["customers"]),
        planner=MockPlanner(),
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema
    from query_processing.retrieval.bm25 import BM25Retriever
    orchestrator._bm25_indices[DatabaseType.POSTGRESQL] = BM25Retriever(test_schema)

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")
    assert resp.error is None
    assert resp.guardrail.decision == GuardrailDecision.READ_QUERY
    assert resp.results is not None
    assert resp.results.row_count == 2
    assert resp.attempts == 1


@pytest.mark.asyncio
async def test_orchestrator_retry_loop(test_schema):
    gen = MockPostgresGen()
    val = MockValidator(fail_first_n=1)  # fails attempt 1, passes attempt 2
    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector(["customers"]),
        planner=MockPlanner(),
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema
    from query_processing.retrieval.bm25 import BM25Retriever
    orchestrator._bm25_indices[DatabaseType.POSTGRESQL] = BM25Retriever(test_schema)

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")
    assert resp.error is None
    assert resp.attempts == 2
    assert gen.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_reject_path(test_schema):
    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.REJECT),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema
    from query_processing.retrieval.bm25 import BM25Retriever
    orchestrator._bm25_indices[DatabaseType.POSTGRESQL] = BM25Retriever(test_schema)

    resp = await orchestrator.execute_pipeline("Drop tables", database="postgres")
    assert resp.guardrail.decision == GuardrailDecision.REJECT
    assert resp.error == "Sorry, I can't help with this."
    assert resp.results is None


@pytest.mark.asyncio
async def test_orchestrator_basic_path(test_schema):
    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.BASIC),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema
    from query_processing.retrieval.bm25 import BM25Retriever
    orchestrator._bm25_indices[DatabaseType.POSTGRESQL] = BM25Retriever(test_schema)

    resp = await orchestrator.execute_pipeline("Who are you?", database="postgres")
    assert resp.guardrail.decision == GuardrailDecision.BASIC
    assert resp.basic_answer == "I am NL2AnyQuery."
    assert resp.results is None


@pytest.mark.asyncio
async def test_orchestrator_zero_selected_objects(test_schema):
    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector(selected=[]),  # 0 selected
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema
    from query_processing.retrieval.bm25 import BM25Retriever
    orchestrator._bm25_indices[DatabaseType.POSTGRESQL] = BM25Retriever(test_schema)

    resp = await orchestrator.execute_pipeline("Show astronauts", database="postgres")
    assert "could not find any relevant tables" in resp.error
    assert resp.results is None
