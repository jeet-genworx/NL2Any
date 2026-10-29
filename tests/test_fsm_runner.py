"""Tests for PipelineFSMRunner using fake adapters."""

import asyncio
from typing import Any

import pytest

from backend.src.core.query_processing.pipeline.fsm import (
    ExecutionCompleted,
    GuardrailCompleted,
    PipelineContext,
    PipelineFSMRunner,
    PipelineState,
    PolicyCompleted,
    QueryGenerationCompleted,
    QueryMissingInfo,
    SpellingCompleted,
    StageAdapter,
    StageAdapters,
    TableSelectionCompleted,
    ValidationCompleted,
)
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import (
    ExecutionResult,
    GeneratedQuery,
    GeneratorMissingInfo,
    GuardrailDecision,
    GuardrailResult,
    PolicyResult,
    SpellingCorrectionResult,
    TableSelectionResult,
    ValidationErrorType,
    ValidationResult,
)


class FakeAdapter(StageAdapter):
    """Fake adapter that yields sequential canned responses or invokes a callback."""

    def __init__(self, responses: list[Any] | None = None, tokens: int = 10) -> None:
        self.responses = responses or []
        self.tokens = tokens
        self.call_count = 0

    async def execute(self, context: PipelineContext):
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
        elif self.responses:
            res = self.responses[-1]
        else:
            raise RuntimeError("No canned responses configured for FakeAdapter")
        self.call_count += 1
        if isinstance(res, BaseException):
            raise res
        return res, self.tokens


def _build_fake_adapters(adapter_overrides: dict[PipelineState, StageAdapter] | None = None) -> StageAdapters:
    """Build a default StageAdapters registry populated with happy-path fake adapters."""
    defaults = {
        PipelineState.SPELLING_CHECK: FakeAdapter([
            SpellingCompleted(SpellingCorrectionResult(original_question="show customers", corrected_question="show customers"))
        ]),
        PipelineState.QUESTION_GUARDRAIL: FakeAdapter([
            GuardrailCompleted(GuardrailResult(decision=GuardrailDecision.READ_QUERY))
        ]),
        PipelineState.TABLE_SELECTION: FakeAdapter([
            TableSelectionCompleted(TableSelectionResult(selected_objects=["customers"]))
        ]),
        PipelineState.QUERY_GENERATION: FakeAdapter([
            QueryGenerationCompleted(GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT 1", formatted_query="SELECT 1"))
        ]),
        PipelineState.QUERY_VALIDATION: FakeAdapter([
            ValidationCompleted(ValidationResult(valid=True, error_type=ValidationErrorType.VALID))
        ]),
        PipelineState.POLICY_CHECK: FakeAdapter([
            PolicyCompleted(PolicyResult(allowed=True))
        ]),
        PipelineState.QUERY_EXECUTION: FakeAdapter([
            ExecutionCompleted(ExecutionResult(row_count=1, columns=["id"], preview_rows=[{"id": 1}]))
        ]),
    }
    if adapter_overrides:
        defaults.update(adapter_overrides)
    return StageAdapters(adapters=defaults)


@pytest.mark.asyncio
async def test_runner_happy_path():
    """Verify runner executes all stages in order, collects telemetry, and reaches TERMINAL_SUCCESS."""
    adapters = _build_fake_adapters()
    runner = PipelineFSMRunner(adapters=adapters)
    context = PipelineContext(
        run_id="happy_run",
        question="show customers",
        max_retries=3,
        entry_stage="fresh",
    )

    result_ctx = await runner.run(context, PipelineState.SPELLING_CHECK)

    assert result_ctx.error is None
    assert result_ctx.selected_objects == ["customers"]
    assert result_ctx.generated_query is not None
    assert result_ctx.execution_result is not None
    assert result_ctx.execution_result.row_count == 1
    # Check that latency was accumulated across stages
    assert "spelling_check" in result_ctx.accum_stage_latency_ms
    assert "query_generation" in result_ctx.accum_stage_latency_ms
    assert "query_execution" in result_ctx.accum_stage_latency_ms


@pytest.mark.asyncio
async def test_runner_validation_retry():
    """Verify runner loops from Validation back to Query Generation on invalid query."""
    val_adapter = FakeAdapter([
        # Attempt 1: invalid query
        ValidationCompleted(ValidationResult(valid=False, error_type=ValidationErrorType.SYNTAX_ERROR, issues=["Syntax error near WHERE"])),
        # Attempt 2: valid query
        ValidationCompleted(ValidationResult(valid=True, error_type=ValidationErrorType.VALID)),
    ])
    gen_adapter = FakeAdapter([
        QueryGenerationCompleted(GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT bad", formatted_query="SELECT bad")),
        QueryGenerationCompleted(GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT good", formatted_query="SELECT good")),
    ])

    adapters = _build_fake_adapters({
        PipelineState.QUERY_VALIDATION: val_adapter,
        PipelineState.QUERY_GENERATION: gen_adapter,
    })
    runner = PipelineFSMRunner(adapters=adapters)
    context = PipelineContext(run_id="retry_val_run", max_retries=3)

    result_ctx = await runner.run(context, PipelineState.SPELLING_CHECK)

    assert result_ctx.error is None
    assert result_ctx.validation_retries == 1
    assert gen_adapter.call_count == 2
    assert val_adapter.call_count == 2


@pytest.mark.asyncio
async def test_runner_generator_to_selector_backtrack():
    """Verify runner backtracks from Query Generation to Table Selection on missing_information."""
    selector_adapter = FakeAdapter([
        # Attempt 1: only customers
        TableSelectionCompleted(TableSelectionResult(selected_objects=["customers"])),
        # Attempt 2 (backtrack retry): customers and orders
        TableSelectionCompleted(TableSelectionResult(selected_objects=["customers", "orders"])),
    ])
    gen_adapter = FakeAdapter([
        # Attempt 1: missing orders table
        QueryMissingInfo(GeneratorMissingInfo(missing_tables=["orders"], reason="Need orders table")),
        # Attempt 2: valid query
        QueryGenerationCompleted(GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT 1", formatted_query="SELECT 1")),
    ])

    adapters = _build_fake_adapters({
        PipelineState.TABLE_SELECTION: selector_adapter,
        PipelineState.QUERY_GENERATION: gen_adapter,
    })
    runner = PipelineFSMRunner(adapters=adapters)
    context = PipelineContext(run_id="backtrack_run", max_retries=3)

    result_ctx = await runner.run(context, PipelineState.SPELLING_CHECK)

    assert result_ctx.error is None
    assert result_ctx.selector_retries == 1
    assert selector_adapter.call_count == 2
    assert gen_adapter.call_count == 2
    assert result_ctx.selected_objects == ["customers", "orders"]


@pytest.mark.asyncio
async def test_runner_frozen_stage_1_fail_fast():
    """Verify runner terminates immediately with specific error when Stage 1 is frozen on missing info."""
    gen_adapter = FakeAdapter([
        QueryMissingInfo(GeneratorMissingInfo(missing_tables=["invoices"], reason="Need invoices table"))
    ])
    selector_adapter = FakeAdapter([
        TableSelectionCompleted(TableSelectionResult(selected_objects=["vendors"]))
    ])

    adapters = _build_fake_adapters({
        PipelineState.QUERY_GENERATION: gen_adapter,
        PipelineState.TABLE_SELECTION: selector_adapter,
    })
    runner = PipelineFSMRunner(adapters=adapters)
    context = PipelineContext(
        run_id="frozen_run",
        is_rerun=True,
        entry_stage="query_generation",
        stage_1_frozen=True,
        selected_objects=["vendors"],
        max_retries=3,
    )

    result_ctx = await runner.run(context, PipelineState.QUERY_GENERATION)

    assert result_ctx.error is not None
    assert "Please rerun from Table Selection with guidance" in result_ctx.error
    # Table selection must NEVER be invoked when Stage 1 is frozen
    assert selector_adapter.call_count == 0


@pytest.mark.asyncio
async def test_runner_cap_enforcement():
    """Verify runner terminates with safety cap error if transitions exceed the cap."""
    # Endless table selection retry loop by artificially keeping max_retries high while cap is small
    selector_adapter = FakeAdapter([
        TableSelectionCompleted(TableSelectionResult(selected_objects=[]))
    ])

    adapters = _build_fake_adapters({
        PipelineState.TABLE_SELECTION: selector_adapter,
    })
    runner = PipelineFSMRunner(adapters=adapters)
    # Set max_retries = 1 so cap = 5*1 + 8 = 13
    context = PipelineContext(run_id="cap_run", max_retries=1)

    result_ctx = await runner.run(context, PipelineState.TABLE_SELECTION)

    assert result_ctx.error is not None
    assert result_ctx.selector_retries >= 1


@pytest.mark.asyncio
async def test_runner_cancellation_propagating():
    """Verify asyncio.CancelledError is never caught or swallowed by runner."""
    cancelling_adapter = FakeAdapter([asyncio.CancelledError()])

    adapters = _build_fake_adapters({
        PipelineState.QUERY_GENERATION: cancelling_adapter,
    })
    runner = PipelineFSMRunner(adapters=adapters)
    context = PipelineContext(run_id="cancel_run", max_retries=3)

    with pytest.raises(asyncio.CancelledError):
        await runner.run(context, PipelineState.QUERY_GENERATION)


@pytest.mark.asyncio
async def test_runner_notifies_stage_events():
    """Verify runner invokes on_stage_event with started and completed for each stage."""
    events = []

    def callback(stage: str, event_type: str, summary: dict[str, Any] | None) -> None:
        events.append((stage, event_type, summary))

    adapters = _build_fake_adapters()
    runner = PipelineFSMRunner(adapters=adapters)
    context = PipelineContext(
        run_id="events_run",
        question="show customers",
        max_retries=3,
        entry_stage="fresh",
        on_stage_event=callback,
    )

    result_ctx = await runner.run(context, PipelineState.SPELLING_CHECK)
    assert result_ctx.error is None

    started_stages = [stage for stage, evt_type, _ in events if evt_type == "started"]
    completed_stages = [stage for stage, evt_type, _ in events if evt_type == "completed"]

    assert started_stages == completed_stages
    assert "spelling_check" in started_stages
    assert "table_selection" in started_stages
    assert "query_generation" in started_stages

    table_sel_evt = next(s for stg, evt, s in events if stg == "table_selection" and evt == "completed")
    assert table_sel_evt["selected_objects"] == ["customers"]

