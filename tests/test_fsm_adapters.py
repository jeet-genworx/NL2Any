"""Tests for stage adapters ensuring exact argument passing, token resetting, failure conversion, and cancellation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.src.core.query_processing.pipeline.feedback_guardrail import (
    FeedbackGuardrailResult,
)
from backend.src.core.query_processing.pipeline.fsm.adapters import (
    ExecutionAdapter,
    FeedbackGuardrailAdapter,
    PolicyAdapter,
    QueryGenerationAdapter,
    QueryValidationAdapter,
    QuestionGuardrailAdapter,
    SpellingAdapter,
    TableSelectionAdapter,
)
from backend.src.core.query_processing.pipeline.fsm.context import PipelineContext
from backend.src.core.query_processing.pipeline.fsm.events import (
    ExecutionCompleted,
    ExecutionFailed,
    FeedbackGuardrailCompleted,
    FeedbackGuardrailFailed,
    GuardrailCompleted,
    GuardrailFailed,
    PolicyCompleted,
    PolicyFailed,
    QueryGenerationCompleted,
    QueryGenerationFailed,
    QueryMissingInfo,
    SpellingCheckFailed,
    SpellingCompleted,
    TableSelectionCompleted,
    TableSelectionFailed,
    ValidationCompleted,
    ValidationFailed,
)
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.models.targets import DatabaseTarget
from backend.src.schemas.pipeline import (
    ExecutionResult,
    GeneratedQuery,
    GeneratorMissingInfo,
    GuardrailDecision,
    GuardrailResult,
    PolicyResult,
    RelevantSchema,
    SpellingCorrectionResult,
    TableSelectionResult,
    ValidationErrorType,
    ValidationResult,
)


@pytest.mark.asyncio
async def test_spelling_adapter():
    mock_checker = MagicMock()
    mock_checker.check.return_value = SpellingCorrectionResult(
        original_question="custmr", corrected_question="customer"
    )

    adapter = SpellingAdapter(checker=mock_checker)
    ctx = PipelineContext(run_id="run1", question="custmr", jargons=["crm"])

    # 1. Exact arguments and return
    event, tokens = await adapter.execute(ctx)
    mock_checker.check.assert_called_once_with(question="custmr", jargons=["crm"])
    assert isinstance(event, SpellingCompleted)
    assert event.result.corrected_question == "customer"
    assert tokens == 0

    # 2. Exception converted to failure event
    mock_checker.check.side_effect = RuntimeError("SymSpell crashed")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, SpellingCheckFailed)
    assert "SymSpell crashed" in event.error
    assert tokens == 0

    # 3. Cancellation propagates
    mock_checker.check.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_question_guardrail_adapter():
    mock_guardrail = MagicMock()
    mock_guardrail.classify = AsyncMock(
        return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY, reason="Read query")
    )
    mock_guardrail.last_prompt_tokens = 45

    adapter = QuestionGuardrailAdapter(guardrail=mock_guardrail)
    ctx = PipelineContext(
        run_id="run1",
        processed_question="show customers",
        stage_providers={"table_selection": "hf"},
    )

    # 1. Exact arguments, token extraction and reset
    event, tokens = await adapter.execute(ctx)
    mock_guardrail.classify.assert_awaited_once_with("show customers")
    assert isinstance(event, GuardrailCompleted)
    assert event.result.decision == GuardrailDecision.READ_QUERY
    assert tokens == 45
    assert mock_guardrail.last_prompt_tokens == 0

    # 2. Basic question handler invocation
    mock_guardrail.classify = AsyncMock(
        return_value=GuardrailResult(decision=GuardrailDecision.BASIC, reason="")
    )
    mock_guardrail.handle_basic_question = MagicMock(return_value="I am NL2AnyQuery.")
    mock_guardrail.last_prompt_tokens = 12
    ctx.target_name = "postgres"
    ctx.schema = DatabaseSchema(
        database_name="sales_db",
        database_type=DatabaseType.POSTGRESQL,
        tables={},
    )

    event, tokens = await adapter.execute(ctx)
    mock_guardrail.handle_basic_question.assert_called_once_with(
        question="show customers",
        database_type="postgres",
        database_name="sales_db",
    )
    assert isinstance(event, GuardrailCompleted)
    assert event.basic_answer == "I am NL2AnyQuery."
    assert tokens == 12

    # 3. Exception converted to failure event with provider name
    mock_guardrail.classify.side_effect = RuntimeError("Model timeout")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, GuardrailFailed)
    assert "Model timeout" in event.error
    assert event.provider == "hf"
    assert tokens == 0

    # 4. Cancellation propagates
    mock_guardrail.classify.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_feedback_guardrail_adapter():
    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(
        return_value=FeedbackGuardrailResult(decision="ACCEPT", prompt_tokens=20)
    )

    adapter = FeedbackGuardrailAdapter(classifier=mock_classifier)
    ctx = PipelineContext(
        run_id="run1",
        applicable_feedback={"table_selection": "use users table", "validation": "fix syntax"},
    )

    # 1. Exact arguments and token accumulation
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, FeedbackGuardrailCompleted)
    assert len(event.results) == 2
    assert tokens == 40

    # 2. Exception converted to failure event
    mock_classifier.classify.side_effect = RuntimeError("Classifier error")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, FeedbackGuardrailFailed)
    assert "Classifier error" in event.error
    assert tokens == 0

    # 3. Cancellation propagates
    mock_classifier.classify.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_table_selection_adapter():
    mock_selector = MagicMock()
    mock_selector.select = AsyncMock(
        return_value=TableSelectionResult(selected_objects=["orders"], sufficient=True)
    )
    mock_selector.last_prompt_tokens = 85

    adapter = TableSelectionAdapter(selector=mock_selector)
    dummy_schema = DatabaseSchema(
        database_name="sales_db",
        database_type=DatabaseType.POSTGRESQL,
        tables={},
    )
    ctx = PipelineContext(
        run_id="run1",
        processed_question="find orders",
        schema=dummy_schema,
        target_name="postgres",
        s1_user_guidance="Focus on 2024 orders",
        s1_system_retry="[SYSTEM RETRY FEEDBACK]:\nRetry selecting",
        stage_providers={"table_selection": "gemini"},
    )

    # 1. Exact arguments (including feedback composition), token read and reset
    event, tokens = await adapter.execute(ctx)
    expected_feedback = "Focus on 2024 orders\n\n[SYSTEM RETRY FEEDBACK]:\nRetry selecting"
    mock_selector.select.assert_awaited_once_with(
        question="find orders",
        schema=dummy_schema,
        feedback=expected_feedback,
        target_key="postgres",
    )
    assert isinstance(event, TableSelectionCompleted)
    assert event.result.selected_objects == ["orders"]
    assert tokens == 85
    assert mock_selector.last_prompt_tokens == 0

    # 2. Exception converted to failure event including provider name
    mock_selector.select.side_effect = RuntimeError("Table selector failed")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, TableSelectionFailed)
    assert "Table selector failed" in event.error
    assert event.provider == "gemini"
    assert tokens == 0

    # 3. Cancellation propagates
    mock_selector.select.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_query_generation_adapter():
    mock_pg_gen = MagicMock()
    mock_mongo_gen = MagicMock()

    pg_query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT * FROM orders;",
        formatted_query="SELECT * FROM orders;",
    )
    mock_pg_gen.generate_query = AsyncMock(return_value=pg_query)
    mock_pg_gen.last_prompt_tokens = 110

    adapter = QueryGenerationAdapter(postgres_gen=mock_pg_gen, mongo_gen=mock_mongo_gen)
    rel_schema = RelevantSchema()
    ctx = PipelineContext(
        run_id="run1",
        db_type=DatabaseType.POSTGRESQL,
        processed_question="find orders",
        relevant_schema=rel_schema,
        s2_user_guidance="orders only",
        s2_system_retry="[SYSTEM VALIDATOR FEEDBACK (RETRY)]:\nfix syntax",
        stage_providers={"query_generation": "koboldcpp"},
    )

    # 1. Exact arguments for PostgreSQL, tokens read and reset
    event, tokens = await adapter.execute(ctx)
    expected_feedback = "orders only\n\n[SYSTEM VALIDATOR FEEDBACK (RETRY)]:\nfix syntax"
    mock_pg_gen.generate_query.assert_awaited_once_with(
        question="find orders",
        schema=rel_schema,
        feedback=expected_feedback,
    )
    mock_mongo_gen.generate_query.assert_not_called()
    assert isinstance(event, QueryGenerationCompleted)
    assert event.query == pg_query
    assert tokens == 110
    assert mock_pg_gen.last_prompt_tokens == 0

    # 2. GeneratorMissingInfo becomes QueryMissingInfo
    missing_info = GeneratorMissingInfo(missing_tables=["invoices"], reason="Need invoice table")
    mock_pg_gen.generate_query = AsyncMock(return_value=missing_info)
    mock_pg_gen.last_prompt_tokens = 95
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, QueryMissingInfo)
    assert event.missing_info == missing_info
    assert tokens == 95
    assert mock_pg_gen.last_prompt_tokens == 0

    # 3. MongoDB routing
    ctx.db_type = DatabaseType.MONGODB
    mongo_query = GeneratedQuery(
        database_type=DatabaseType.MONGODB,
        raw_query="db.orders.find()",
        formatted_query="db.orders.find()",
    )
    mock_mongo_gen.generate_query = AsyncMock(return_value=mongo_query)
    mock_mongo_gen.last_prompt_tokens = 70
    event, tokens = await adapter.execute(ctx)
    mock_mongo_gen.generate_query.assert_awaited_once_with(
        question="find orders",
        schema=rel_schema,
        feedback=expected_feedback,
    )
    assert isinstance(event, QueryGenerationCompleted)
    assert tokens == 70
    assert mock_mongo_gen.last_prompt_tokens == 0

    # 4. Exception converted to failure event including provider name
    mock_mongo_gen.generate_query.side_effect = RuntimeError("Generation failed")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, QueryGenerationFailed)
    assert "Generation failed" in event.error
    assert event.provider == "koboldcpp"
    assert tokens == 0

    # 5. Cancellation propagates
    mock_mongo_gen.generate_query.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_query_validation_adapter():
    mock_validator = MagicMock()
    mock_child_pg = MagicMock()
    mock_child_mongo = MagicMock()
    mock_validator.postgres_validator = mock_child_pg
    mock_validator.mongo_validator = mock_child_mongo

    val_res = ValidationResult(valid=True, error_type=ValidationErrorType.VALID)
    mock_validator.validate = AsyncMock(return_value=val_res)
    mock_validator.last_prompt_tokens = 130
    mock_child_pg.last_prompt_tokens = 50
    mock_child_mongo.last_prompt_tokens = 40

    adapter = QueryValidationAdapter(validator=mock_validator)
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT 1;",
        formatted_query="SELECT 1;",
    )
    rel_schema = RelevantSchema()
    ctx = PipelineContext(
        run_id="run1",
        processed_question="check query",
        generated_query=query,
        relevant_schema=rel_schema,
        s3_user_guidance="Guidance 3",
        s3_system_retry="Retry 3",
        stage_providers={"validation": "gemini"},
    )

    # 1. Exact arguments, fail_fast=True, token read and reset (including child validators)
    event, tokens = await adapter.execute(ctx)
    expected_feedback = "Guidance 3\n\nRetry 3"
    mock_validator.validate.assert_awaited_once_with(
        question="check query",
        generated_query=query,
        schema=rel_schema,
        feedback=expected_feedback,
        fail_fast=True,
    )
    assert isinstance(event, ValidationCompleted)
    assert event.result == val_res
    assert tokens == 130
    assert mock_validator.last_prompt_tokens == 0
    assert mock_child_pg.last_prompt_tokens == 0
    assert mock_child_mongo.last_prompt_tokens == 0

    # 2. Exception converted to failure event including provider name
    mock_validator.validate.side_effect = RuntimeError("Validator timed out")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, ValidationFailed)
    assert "Validator timed out" in event.error
    assert event.provider == "gemini"
    assert tokens == 0

    # 3. Cancellation propagates
    mock_validator.validate.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_policy_adapter():
    mock_policy = MagicMock()
    pol_res = PolicyResult(allowed=True, reason="Safe")
    mock_policy.check.return_value = pol_res

    adapter = PolicyAdapter(policy=mock_policy)
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT 1;",
        formatted_query="SELECT 1;",
    )
    rel_schema = RelevantSchema()
    ctx = PipelineContext(run_id="run1", generated_query=query, relevant_schema=rel_schema)

    # 1. Exact arguments and return
    event, tokens = await adapter.execute(ctx)
    mock_policy.check.assert_called_once_with(query, schema=rel_schema)
    assert isinstance(event, PolicyCompleted)
    assert event.result == pol_res
    assert tokens == 0

    # 2. Exception converted to failure event
    mock_policy.check.side_effect = RuntimeError("Policy AST error")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, PolicyFailed)
    assert "Policy AST error" in event.error
    assert tokens == 0

    # 3. Cancellation propagates
    mock_policy.check.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)


@pytest.mark.asyncio
async def test_execution_adapter():
    mock_executor = MagicMock()
    mock_processor = MagicMock()

    mock_executor.execute.return_value = (["id"], [{"id": 1}])
    exec_res = ExecutionResult(row_count=1, columns=["id"], preview_rows=[{"id": 1}])
    mock_processor.process.return_value = exec_res

    adapter = ExecutionAdapter(executor=mock_executor, results_processor=mock_processor)
    query = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT 1;",
        formatted_query="SELECT 1;",
    )
    target = DatabaseTarget(
        key="postgres",
        label="PostgreSQL",
        db_type=DatabaseType.POSTGRESQL,
        file_stem="postgres",
        connection_override="mock_conn",
    )
    ctx = PipelineContext(
        run_id="run1",
        generated_query=query,
        database_target=target,
    )

    # 1. Exact arguments and return
    event, tokens = await adapter.execute(ctx)
    mock_executor.execute.assert_called_once_with(query, connection="mock_conn", database=None)
    mock_processor.process.assert_called_once_with(["id"], [{"id": 1}])
    assert isinstance(event, ExecutionCompleted)
    assert event.result == exec_res
    assert tokens == 0

    # 2. Exception converted to failure event
    mock_executor.execute.side_effect = RuntimeError("Connection dropped")
    event, tokens = await adapter.execute(ctx)
    assert isinstance(event, ExecutionFailed)
    assert "Connection dropped" in event.error
    assert tokens == 0

    # 3. Cancellation propagates
    mock_executor.execute.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter.execute(ctx)
