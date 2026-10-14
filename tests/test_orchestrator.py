"""Tests for end-to-end NL2AnyQueryOrchestrator V2 pipeline."""

import pytest

from backend.src.core.query_processing.pipeline.orchestrator import (
    NL2AnyQueryOrchestrator,
)
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    GeneratorMissingInfo,
    GuardrailDecision,
    GuardrailResult,
    MongoQuery,
    PolicyResult,
    TableSelectionResult,
    ValidationErrorType,
    ValidationResult,
)


class MockGuardrail:
    def __init__(self, decision: GuardrailDecision = GuardrailDecision.READ_QUERY) -> None:
        self.decision = decision
        self.call_count = 0
        self.last_question = ""

    async def classify(self, question: str) -> GuardrailResult:
        self.call_count += 1
        self.last_question = question
        return GuardrailResult(decision=self.decision, reason="Mock decision")

    def handle_basic_question(self, question: str, database_type: str, database_name: str) -> str:
        return "I am NL2AnyQuery."


class MockSelector:
    def __init__(self, results: list[TableSelectionResult] | None = None) -> None:
        self.results = results or [
            TableSelectionResult(selected_objects=["customers"], sufficient=True)
        ]
        self.call_count = 0
        self.last_feedback = None

    async def select(self, *args, **kwargs) -> TableSelectionResult:
        idx = min(self.call_count, len(self.results) - 1)
        res = self.results[idx]
        self.call_count += 1
        self.last_feedback = kwargs.get("feedback")
        return res


class MockPostgresGen:
    def __init__(self, responses: list[GeneratedQuery | GeneratorMissingInfo] | None = None) -> None:
        self.responses = responses or [
            GeneratedQuery(
                database_type=DatabaseType.POSTGRESQL,
                raw_query="SELECT id, name FROM customers;",
                formatted_query="SELECT id, name FROM customers;",
            )
        ]
        self.call_count = 0
        self.last_feedback = None

    async def generate_query(self, *args, **kwargs) -> GeneratedQuery | GeneratorMissingInfo:
        idx = min(self.call_count, len(self.responses) - 1)
        res = self.responses[idx]
        self.call_count += 1
        self.last_feedback = kwargs.get("feedback")
        return res


class MockMongoGen:
    def __init__(self, responses: list[GeneratedQuery | GeneratorMissingInfo] | None = None) -> None:
        self.responses = responses or [
            GeneratedQuery(
                database_type=DatabaseType.MONGODB,
                raw_query=MongoQuery(operation="find", collection="customers", filter={"id": 1}),
                formatted_query='{"operation": "find", "collection": "customers"}',
            )
        ]
        self.call_count = 0
        self.last_feedback = None

    async def generate_query(self, *args, **kwargs) -> GeneratedQuery | GeneratorMissingInfo:
        idx = min(self.call_count, len(self.responses) - 1)
        res = self.responses[idx]
        self.call_count += 1
        self.last_feedback = kwargs.get("feedback")
        return res


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
    def __init__(self, allowed: bool = True, reason: str = "Verified safe") -> None:
        self.allowed = allowed
        self.reason = reason
        self.call_count = 0

    def check(self, *args, **kwargs) -> PolicyResult:
        self.call_count += 1
        return PolicyResult(allowed=self.allowed, reason=self.reason)


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
        relationships=[
            Relationship(
                from_object="orders",
                from_field="customer_id",
                to_object="customers",
                to_field="id",
            )
        ],
    )


@pytest.mark.asyncio
async def test_v2_orchestrator_happy_path(test_schema):
    """Test happy path through Spell Checker -> Guardrail -> Selector -> Generator -> Validator -> Policy -> Executor."""
    guardrail = MockGuardrail(GuardrailDecision.READ_QUERY)
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    gen = MockPostgresGen()
    val = MockValidator([ValidationResult(valid=True, error_type=ValidationErrorType.VALID)])
    policy = MockPolicy()
    executor = MockExecutor()

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=guardrail,
        selector=selector,
        postgres_gen=gen,
        validator=val,
        policy=policy,
        executor=executor,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show all customrs", database="postgres")

    assert resp.error is None
    assert resp.guardrail.decision == GuardrailDecision.READ_QUERY
    # Spell checker must have run BEFORE guardrail
    assert resp.spelling_correction is not None
    assert guardrail.call_count == 1
    assert selector.call_count == 1
    assert gen.call_count == 1
    assert val.attempts == 1
    assert policy.call_count == 1
    assert executor.call_count == 1
    assert resp.results.row_count == 2
    assert resp.attempts == 1
    assert resp.selection_retries == 0
    assert resp.validation_retries == 0


@pytest.mark.asyncio
async def test_v2_orchestrator_pipeline_ordering(test_schema):
    """Verify ordering: Spell Checker runs first, Guardrail receives corrected question."""
    guardrail = MockGuardrail(GuardrailDecision.READ_QUERY)
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    gen = MockPostgresGen()
    val = MockValidator([ValidationResult(valid=True, error_type=ValidationErrorType.VALID)])

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=guardrail,
        selector=selector,
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Shoow all customers", database="postgres")
    # Guardrail must have received the spell-checked question
    assert guardrail.last_question == resp.spelling_correction.corrected_question


@pytest.mark.asyncio
async def test_v2_orchestrator_generator_to_selector_backtrack(test_schema):
    """Test Query Generator -> Table Selector backtrack when Generator reports missing information."""
    selector = MockSelector([
        # Attempt 1: only customers selected
        TableSelectionResult(selected_objects=["customers"], sufficient=True),
        # Attempt 2 (retry): both customers and orders selected
        TableSelectionResult(selected_objects=["customers", "orders"], sufficient=True),
    ])

    gen = MockPostgresGen([
        # Attempt 1: reports missing order information
        GeneratorMissingInfo(
            status="missing_information",
            missing_tables=["orders"],
            reason="Order table is needed to display order history.",
        ),
        # Attempt 2: valid SQL after orders table was added
        GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query="SELECT c.name, o.id FROM customers c JOIN orders o ON c.id = o.customer_id;",
            formatted_query="SELECT c.name, o.id FROM customers c JOIN orders o ON c.id = o.customer_id;",
        ),
    ])

    val = MockValidator([ValidationResult(valid=True, error_type=ValidationErrorType.VALID)])

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        selector=selector,
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
        max_retries=3,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customer order history", database="postgres")

    assert resp.error is None
    assert resp.selection_retries == 1
    assert selector.call_count == 2
    assert "lacks required schema information" in selector.last_feedback
    assert "orders" in selector.last_feedback
    assert gen.call_count == 2
    assert resp.results is not None


@pytest.mark.asyncio
async def test_v2_orchestrator_validator_to_generator_backtrack(test_schema):
    """Test Validator -> Query Generator backtrack when generated query fails validation."""
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    gen = MockPostgresGen([
        GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELEC * FROM customers;", formatted_query="SELEC * FROM customers;"),
        GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT * FROM customers;", formatted_query="SELECT * FROM customers;"),
    ])
    val = MockValidator([
        ValidationResult(
            valid=False,
            error_type=ValidationErrorType.SYNTAX_ERROR,
            issues=["Syntax error near 'SELEC'"],
            suggestion="Use 'SELECT'",
        ),
        ValidationResult(valid=True, error_type=ValidationErrorType.VALID),
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        selector=selector,
        postgres_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=MockExecutor(),
        max_retries=3,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")

    assert resp.error is None
    assert resp.validation_retries == 1
    assert resp.attempts == 2
    # Generator was called twice (initial + 1 retry)
    assert gen.call_count == 2
    assert "Use 'SELECT'" in gen.last_feedback
    # Selector was NOT rerun for a validation error
    assert selector.call_count == 1


@pytest.mark.asyncio
async def test_v2_orchestrator_unsafe_immediate_stop(test_schema):
    """Unsafe queries (DROP, DELETE) must stop immediately with no retries or execution."""
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    gen = MockPostgresGen([
        GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="DROP TABLE customers;", formatted_query="DROP TABLE customers;")
    ])
    val = MockValidator([
        ValidationResult(
            valid=False,
            error_type=ValidationErrorType.UNSAFE,
            issues=["Destructive SQL statement detected"],
            suggestion=None,
        )
    ])
    policy = MockPolicy()
    executor = MockExecutor()

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        selector=selector,
        postgres_gen=gen,
        validator=val,
        policy=policy,
        executor=executor,
        max_retries=3,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Drop table customers", database="postgres")

    assert "unsafe" in resp.error.lower()
    assert val.attempts == 1
    assert gen.call_count == 1
    assert selector.call_count == 1
    assert policy.call_count == 0
    assert executor.call_count == 0


@pytest.mark.asyncio
async def test_v2_orchestrator_guardrail_reject(test_schema):
    """Guardrail REJECT terminates immediately before table selection or query generation."""
    guardrail = MockGuardrail(GuardrailDecision.REJECT)
    selector = MockSelector()
    gen = MockPostgresGen()

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=guardrail,
        selector=selector,
        postgres_gen=gen,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Delete database", database="postgres")
    assert resp.guardrail.decision == GuardrailDecision.REJECT
    assert selector.call_count == 0
    assert gen.call_count == 0


@pytest.mark.asyncio
async def test_v2_orchestrator_guardrail_basic(test_schema):
    """Guardrail BASIC returns metadata answer immediately without querying database."""
    guardrail = MockGuardrail(GuardrailDecision.BASIC)
    selector = MockSelector()
    gen = MockPostgresGen()

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=guardrail,
        selector=selector,
        postgres_gen=gen,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("What database is this?", database="postgres")
    assert resp.basic_answer == "I am NL2AnyQuery."
    assert selector.call_count == 0
    assert gen.call_count == 0


@pytest.mark.asyncio
async def test_v2_orchestrator_selector_retry_exhaustion(test_schema):
    """Table Selector retry limit (3 retries max) is enforced."""
    selector = MockSelector([
        TableSelectionResult(selected_objects=["customers"], sufficient=True)
    ])
    # Generator persistently returns missing information
    gen = MockPostgresGen([
        GeneratorMissingInfo(status="missing_information", missing_tables=["orders"], reason="Need orders")
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        selector=selector,
        postgres_gen=gen,
        max_retries=3,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show orders", database="postgres")
    assert "after maximum retries" in resp.error
    # 1 initial + 3 retries = 4 total attempts
    assert selector.call_count == 4
    assert resp.selection_retries == 3


@pytest.mark.asyncio
async def test_v2_orchestrator_validator_retry_exhaustion(test_schema):
    """Validator retry limit (3 retries max) is enforced."""
    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    gen = MockPostgresGen([
        GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT bad_syntax", formatted_query="SELECT bad_syntax")
    ])
    val = MockValidator([
        ValidationResult(valid=False, error_type=ValidationErrorType.SYNTAX_ERROR, issues=["Syntax error"])
    ])

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        selector=selector,
        postgres_gen=gen,
        validator=val,
        max_retries=3,
    )
    orchestrator.schema_cache._schemas[DatabaseType.POSTGRESQL] = test_schema

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")
    assert "after maximum retries" in resp.error
    # 1 initial + 3 retries = 4 total attempts
    assert gen.call_count == 4
    assert resp.validation_retries == 3


@pytest.mark.asyncio
async def test_v2_orchestrator_mongo_execution():
    """Test full V2 flow for MongoDB target."""
    mongo_schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop_mongo",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.COLLECTION,
                fields=[Field(name="_id", type="string"), Field(name="city", type="string")],
            )
        ],
    )

    selector = MockSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)])
    gen = MockMongoGen()
    val = MockValidator([ValidationResult(valid=True, error_type=ValidationErrorType.VALID)])
    executor = MockExecutor()

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(GuardrailDecision.READ_QUERY),
        selector=selector,
        mongo_gen=gen,
        validator=val,
        policy=MockPolicy(),
        executor=executor,
    )
    orchestrator.schema_cache._schemas[DatabaseType.MONGODB] = mongo_schema

    resp = await orchestrator.execute_pipeline("Show mongo customers", database="mongo")

    assert resp.error is None
    assert gen.call_count == 1
    assert val.attempts == 1
    assert executor.call_count == 1
    assert resp.results.row_count == 2
