"""Tests for end-to-end NL2AnyQueryOrchestrator with targeted retries and mocks."""

import pytest
from query_processing.models.pipeline import (
    CandidateTable,
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    PolicyResult,
    QueryPlan,
    SemanticAnalysisResult,
    TableSelectionResult,
    ValidationErrorType,
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


class MockEmbeddingProvider:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1] * 384
        self.call_count = 0

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        return self.vector


class MockVectorRetriever:
    def __init__(self, candidates: list[CandidateTable] | None = None) -> None:
        self.candidates = candidates if candidates is not None else [
            CandidateTable(table_name="customers", similarity=0.92, rank=1)
        ]
        self.call_count = 0

    def retrieve(self, query_vector: list[float]) -> list[CandidateTable]:
        self.call_count += 1
        return list(self.candidates)


class MockGuardrail:
    def __init__(self, decision: GuardrailDecision = GuardrailDecision.READ_QUERY) -> None:
        self.decision = decision
        self.call_count = 0

    async def classify(self, question: str) -> GuardrailResult:
        self.call_count += 1
        return GuardrailResult(decision=self.decision, reason="Mock decision")

    def handle_basic_question(self, question: str, database_type: str, database_name: str) -> str:
        return "I am NL2AnyQuery."


class MockSemantic:
    def __init__(self) -> None:
        self.call_count = 0

    async def analyze(self, question: str) -> SemanticAnalysisResult:
        self.call_count += 1
        return SemanticAnalysisResult(subjective=["customers"], objective=["all"])


class MockSelector:
    def __init__(self, results: list[TableSelectionResult] | None = None) -> None:
        self.results = results or [
            TableSelectionResult(selected_objects=["customers"], sufficient=True)
        ]
        self.call_count = 0

    async def select(self, *args, **kwargs) -> TableSelectionResult:
        idx = min(self.call_count, len(self.results) - 1)
        res = self.results[idx]
        self.call_count += 1
        return res


class MockPlanner:
    def __init__(self) -> None:
        self.call_count = 0
        self.last_feedback = None

    async def plan(self, *args, **kwargs) -> QueryPlan:
        self.call_count += 1
        self.last_feedback = kwargs.get("feedback")
        return QueryPlan(sources=["customers"], projections=["id", "name"])


class MockPostgresGen:
    def __init__(self) -> None:
        self.call_count = 0
        self.last_feedback = None

    async def generate_query(self, *args, **kwargs) -> GeneratedQuery:
        self.call_count += 1
        self.last_feedback = kwargs.get("feedback")
        return GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query="SELECT id, name FROM customers;",
            formatted_query="SELECT id, name FROM customers;",
        )


class MockValidator:
    def __init__(self, responses: list[ValidationResult] | None = None) -> None:
        self.responses = responses or [
            ValidationResult(valid=True, error_type=ValidationErrorType.VALID, issues=[])
        ]
        self.attempts = 0

    async def validate(self, *args, **kwargs) -> ValidationResult:
        idx = min(self.attempts, len(self.responses) - 1)
        res = self.responses[idx]
        self.attempts += 1
        return res


class MockPolicy:
    def __init__(self) -> None:
        self.call_count = 0

    def check(self, *args, **kwargs) -> PolicyResult:
        self.call_count += 1
        return PolicyResult(allowed=True, reason="Verified safe")


class MockExecutor:
    def __init__(self) -> None:
        self.call_count = 0

    def execute(self, *args, **kwargs):
        self.call_count += 1
        return ["id", "name"], [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@pytest.fixture
def test_schema() -> DatabaseSchema:
    cust = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="name", type="varchar")],
    )
    orders = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="customer_id", type="integer")],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="test_shop",
        objects=[cust, orders],
    )


@pytest.mark.asyncio
async def test_orchestrator_happy_path(test_schema):
    gen = MockPostgresGen()
    val = MockValidator([ValidationResult(valid=True, error_type=ValidationErrorType.VALID)])
    retriever = MockVectorRetriever([
        CandidateTable(table_name="customers", similarity=0.92, rank=1)
    ])
    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=retriever,
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector([
            TableSelectionResult(selected_objects=["customers"], sufficient=True)
        ]),
        planner=MockPlanner(),
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")
    assert resp.error is None
    assert resp.guardrail.decision == GuardrailDecision.READ_QUERY
    assert resp.results is not None
    assert resp.results.row_count == 2
    assert resp.attempts == 1
    assert resp.selection_retries == 0
    assert resp.validation_retries == 0
    assert len(resp.candidate_tables) == 1
    assert resp.candidate_tables[0].table_name == "customers"


@pytest.mark.asyncio
async def test_orchestrator_syntax_error_generator_retry(test_schema):
    # Attempt 1: Syntax error (generator retry)
    # Attempt 2: Valid
    planner = MockPlanner()
    gen = MockPostgresGen()
    guardrail = MockGuardrail(GuardrailDecision.READ_QUERY)
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    val = MockValidator([
        ValidationResult(
            valid=False,
            error_type=ValidationErrorType.SYNTAX_ERROR,
            issues=["PostgreSQL syntax error in SELECT clause"],
            suggestion="Correct SELECT syntax",
        ),
        ValidationResult(valid=True, error_type=ValidationErrorType.VALID),
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=MockVectorRetriever(),
        guardrail=guardrail,
        semantic=MockSemantic(),
        selector=selector,
        planner=planner,
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
        max_retries=3,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")
    assert resp.error is None
    assert resp.validation_retries == 1
    assert resp.attempts == 2

    # CRITICAL CHECK: Planner must NOT be rerun for a syntax error!
    assert planner.call_count == 1
    # Generator MUST be rerun
    assert gen.call_count == 2
    # Earlier stages must NOT be rerun
    assert guardrail.call_count == 1
    assert selector.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_planner_error_planner_retry(test_schema):
    # Attempt 1: Planner error (triggers planner retry -> generator -> validator)
    # Attempt 2: Valid
    planner = MockPlanner()
    gen = MockPostgresGen()
    guardrail = MockGuardrail(GuardrailDecision.READ_QUERY)
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    val = MockValidator([
        ValidationResult(
            valid=False,
            error_type=ValidationErrorType.PLANNER_ERROR,
            issues=["QueryPlan should aggregate order count per customer"],
            suggestion="Add count aggregation",
        ),
        ValidationResult(valid=True, error_type=ValidationErrorType.VALID),
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=MockVectorRetriever(),
        guardrail=guardrail,
        semantic=MockSemantic(),
        selector=selector,
        planner=planner,
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
        max_retries=3,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customer order counts", database="postgres")
    assert resp.error is None
    assert resp.validation_retries == 1
    assert resp.attempts == 2

    # CRITICAL CHECK: Planner MUST be rerun for a planner error!
    assert planner.call_count == 2
    assert "QueryPlan should aggregate" in planner.last_feedback
    # Generator MUST be rerun
    assert gen.call_count == 2
    # Earlier stages (Guardrail, Selector) must NOT be rerun
    assert guardrail.call_count == 1
    assert selector.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_unsafe_immediate_stop(test_schema):
    # Attempt 1: Unsafe query (e.g. destructive statement detected)
    planner = MockPlanner()
    gen = MockPostgresGen()
    policy = MockPolicy()
    executor = MockExecutor()
    val = MockValidator([
        ValidationResult(
            valid=False,
            error_type=ValidationErrorType.UNSAFE,
            issues=["Destructive SQL statement detected"],
            suggestion=None,
        )
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=MockVectorRetriever(),
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector(),
        planner=planner,
        postgres_gen=gen,
        validator=val,
        policy=policy,
        executor=executor,
        max_retries=3,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Delete all customers", database="postgres")
    assert "unsafe" in resp.error.lower()
    # CRITICAL CHECK: Unsafe queries MUST STOP immediately with no retries and no execution!
    assert val.attempts == 1
    assert gen.call_count == 1
    assert planner.call_count == 1
    assert policy.call_count == 0
    assert executor.call_count == 0
    assert resp.results is None


@pytest.mark.asyncio
async def test_orchestrator_validation_exhaustion(test_schema):
    # Validator fails continuously on every attempt
    gen = MockPostgresGen()
    policy = MockPolicy()
    executor = MockExecutor()
    val = MockValidator([
        ValidationResult(
            valid=False,
            error_type=ValidationErrorType.SYNTAX_ERROR,
            issues=["Persistent syntax error"],
            suggestion="Fix syntax",
        )
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=MockVectorRetriever(),
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector(),
        planner=MockPlanner(),
        postgres_gen=gen,
        validator=val,
        policy=policy,
        executor=executor,
        max_retries=3,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")
    assert "Query failed validation after maximum retries" in resp.error
    # 1 initial + 3 retries = 4 validation attempts
    assert val.attempts == 4
    assert resp.validation_retries == 3
    # Policy and executor must never be called on failed validation
    assert policy.call_count == 0
    assert executor.call_count == 0
    assert resp.results is None


@pytest.mark.asyncio
async def test_orchestrator_zero_embedding_candidates(test_schema):
    retriever = MockVectorRetriever(candidates=[])
    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=retriever,
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show astronauts", database="postgres")
    assert "could not find any relevant tables" in resp.error
    assert resp.candidate_tables == []
    assert resp.results is None


@pytest.mark.asyncio
async def test_orchestrator_selector_retry_success(test_schema):
    retriever = MockVectorRetriever([
        CandidateTable(table_name="customers", similarity=0.88, rank=1)
    ])

    selector = MockSelector([
        TableSelectionResult(
            selected_objects=["customers"],
            sufficient=False,
            missing_objects=["orders"],
            reason="Need orders table for counting orders.",
            retrieval_hint="customer orders placed and dates",
        ),
        TableSelectionResult(
            selected_objects=["customers", "orders"],
            sufficient=True,
            missing_objects=[],
            reason="Both tables present now.",
        ),
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=retriever,
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=selector,
        planner=MockPlanner(),
        postgres_gen=MockPostgresGen(),
        validator=MockValidator(),
        policy=MockPolicy(),
        executor=MockExecutor(),
        max_retries=3,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customer order counts", database="postgres")
    assert resp.error is None
    assert resp.selection_retries == 1
    assert "customers" in resp.selected_objects
    assert resp.results is not None


@pytest.mark.asyncio
async def test_orchestrator_selector_exhaustion(test_schema):
    retriever = MockVectorRetriever([
        CandidateTable(table_name="customers", similarity=0.88, rank=1)
    ])

    selector = MockSelector([
        TableSelectionResult(
            selected_objects=["customers"],
            sufficient=False,
            missing_objects=["unknown_table"],
            reason="Missing unknown table.",
            retrieval_hint="hint",
        )
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=retriever,
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=selector,
        max_retries=3,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show mysterious data", database="postgres")
    assert "could not be identified reliably after maximum retries" in resp.error
    assert resp.selection_retries == 3
    assert resp.results is None


@pytest.mark.asyncio
async def test_orchestrator_reject_path(test_schema):
    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.REJECT),
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = test_schema

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

    resp = await orchestrator.execute_pipeline("Who are you?", database="postgres")
    assert resp.guardrail.decision == GuardrailDecision.BASIC
    assert resp.basic_answer == "I am NL2AnyQuery."
    assert resp.results is None
