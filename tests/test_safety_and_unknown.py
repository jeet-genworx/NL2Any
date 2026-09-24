"""Unit tests for Unknown Schema questions and Safety/Guardrail enforcement."""

import pytest
from query_processing.models.pipeline import (
    CandidateTable,
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
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
from query_processing.pipeline.guardrail import GuardrailClassifier
from query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from query_processing.pipeline.policy import SafetyPolicyValidator
from tests.test_orchestrator import (
    MockEmbeddingProvider,
    MockExecutor,
    MockGuardrail,
    MockPlanner,
    MockPolicy,
    MockPostgresGen,
    MockSelector,
    MockSemantic,
    MockValidator,
    MockVectorRetriever,
)


@pytest.fixture
def mock_shop_schema() -> DatabaseSchema:
    cust = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="city", type="varchar")],
    )
    orders = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="integer"), Field(name="customer_id", type="integer")],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[cust, orders],
    )


# --- Section 10: Unknown Schema Tests ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Show me all astronauts.",
        "How many spaceships do we have?",
        "Show me employees and their salary.",
    ],
)
async def test_unknown_schema_zero_candidates(question, mock_shop_schema):
    """When vector retriever returns 0 candidates for out-of-domain questions, halt cleanly."""
    executor = MockExecutor()
    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=MockVectorRetriever(candidates=[]),  # No candidate above threshold
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        semantic=MockSemantic(),
        selector=MockSelector(),
        executor=executor,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = mock_shop_schema

    resp = await orchestrator.execute_pipeline(question, database="postgres")
    assert resp.error is not None
    assert "could not find any relevant tables" in resp.error.lower()
    assert executor.call_count == 0
    assert resp.results is None


# --- Section 11: Safety Tests ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "destructive_query",
    [
        "Delete all customers.",
        "Update John's email.",
        "Drop the orders table.",
        "Insert a new customer.",
        "Please modify every customer record so the city becomes Chennai.",
        "Remove every customer from the database.",
    ],
)
async def test_guardrail_fast_check_rejects_modifications(destructive_query):
    """Deterministic guardrail keywords should immediately catch destructive phrases."""
    classifier = GuardrailClassifier()
    res = await classifier.classify(destructive_query)
    assert res is not None
    assert res.decision == GuardrailDecision.REJECT


@pytest.mark.asyncio
async def test_policy_rejects_multi_statement(mock_shop_schema):
    """SQLGlot deterministic safety policy must reject multi-statement injection."""
    policy = SafetyPolicyValidator()
    multi_stmt = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT * FROM customers; DROP TABLE customers;",
        formatted_query="SELECT * FROM customers; DROP TABLE customers;",
    )
    res = policy.check(multi_stmt)
    assert res.allowed is False
    assert "multiple" in res.reason.lower() or "read-only" in res.reason.lower()


@pytest.mark.asyncio
async def test_validator_rejects_unsafe_operations():
    """Validator SLM/deterministic check must reject destructive operations as UNSAFE."""
    from query_processing.models.pipeline import QueryPlan, RelevantSchema
    from query_processing.pipeline.validator import QueryValidator

    validator = QueryValidator()
    plan = QueryPlan(sources=["customers"], projections=["*"])
    schema = RelevantSchema(objects=[])
    unsafe_query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="DELETE FROM customers WHERE id = 1;",
        formatted_query="DELETE FROM customers WHERE id = 1;",
    )
    result = await validator.validate(
        question="Delete customer 1",
        plan=plan,
        generated_query=unsafe_query,
        schema=schema,
    )
    assert result.valid is False
    assert result.error_type == ValidationErrorType.UNSAFE
