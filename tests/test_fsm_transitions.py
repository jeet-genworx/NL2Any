"""Tests for FSM states, events, context, and pure transition decisions."""

import copy

import pytest

from backend.src.core.query_processing.pipeline.feedback_guardrail import (
    FeedbackGuardrailResult,
)
from backend.src.core.query_processing.pipeline.fsm import (
    CLEAR,
    VALID_TRANSITIONS,
    ContextUpdates,
    ExecutionCompleted,
    ExecutionFailed,
    FeedbackGuardrailCompleted,
    FeedbackGuardrailFailed,
    GuardrailCompleted,
    GuardrailFailed,
    InvalidTransitionError,
    PipelineContext,
    PipelineState,
    PolicyCompleted,
    PolicyFailed,
    QueryGenerationCompleted,
    QueryGenerationFailed,
    QueryMissingInfo,
    SpellingCheckFailed,
    SpellingCompleted,
    TableSelectionCompleted,
    TableSelectionFailed,
    TransitionDecision,
    ValidationCompleted,
    ValidationFailed,
    calculate_transition_cap,
    transition,
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


def _make_dummy_event(event_cls):
    """Create a minimal valid dummy instance for an event class."""
    if event_cls is SpellingCompleted:
        return SpellingCompleted(
            result=SpellingCorrectionResult(
                original_question="test question",
                corrected_question="test question",
            )
        )
    elif event_cls is SpellingCheckFailed:
        return SpellingCheckFailed(error="Spelling model failed")
    elif event_cls is GuardrailCompleted:
        return GuardrailCompleted(
            result=GuardrailResult(decision=GuardrailDecision.READ_QUERY, reason="Valid query")
        )
    elif event_cls is GuardrailFailed:
        return GuardrailFailed(error="Guardrail timeout")
    elif event_cls is FeedbackGuardrailCompleted:
        return FeedbackGuardrailCompleted(
            results=[("table_selection", FeedbackGuardrailResult(decision="ACCEPT", reason="Safe"))]
        )
    elif event_cls is FeedbackGuardrailFailed:
        return FeedbackGuardrailFailed(error="Feedback guardrail failed")
    elif event_cls is TableSelectionCompleted:
        return TableSelectionCompleted(
            result=TableSelectionResult(selected_objects=["customers"], sufficient=True)
        )
    elif event_cls is TableSelectionFailed:
        return TableSelectionFailed(error="Table selector error")
    elif event_cls is QueryGenerationCompleted:
        return QueryGenerationCompleted(
            query=GeneratedQuery(
                database_type=DatabaseType.POSTGRESQL,
                raw_query="SELECT 1;",
                formatted_query="SELECT 1;",
            )
        )
    elif event_cls is QueryMissingInfo:
        return QueryMissingInfo(
            missing_info=GeneratorMissingInfo(
                missing_tables=["orders"], reason="Need orders table"
            )
        )
    elif event_cls is QueryGenerationFailed:
        return QueryGenerationFailed(error="Generator timeout")
    elif event_cls is ValidationCompleted:
        return ValidationCompleted(
            result=ValidationResult(valid=True, error_type=ValidationErrorType.VALID)
        )
    elif event_cls is ValidationFailed:
        return ValidationFailed(error="Validator error")
    elif event_cls is PolicyCompleted:
        return PolicyCompleted(result=PolicyResult(allowed=True, reason="Read-only query"))
    elif event_cls is PolicyFailed:
        return PolicyFailed(error="Policy check error")
    elif event_cls is ExecutionCompleted:
        return ExecutionCompleted(
            result=ExecutionResult(row_count=1, columns=["id"], preview_rows=[{"id": 1}])
        )
    elif event_cls is ExecutionFailed:
        return ExecutionFailed(error="DB connection refused")
    raise ValueError(f"Unknown event class: {event_cls}")


ALL_EVENT_CLASSES = [
    SpellingCompleted,
    SpellingCheckFailed,
    GuardrailCompleted,
    GuardrailFailed,
    FeedbackGuardrailCompleted,
    FeedbackGuardrailFailed,
    TableSelectionCompleted,
    TableSelectionFailed,
    QueryGenerationCompleted,
    QueryMissingInfo,
    QueryGenerationFailed,
    ValidationCompleted,
    ValidationFailed,
    PolicyCompleted,
    PolicyFailed,
    ExecutionCompleted,
    ExecutionFailed,
]


def test_exhaustive_valid_and_invalid_transitions():
    """Verify that every (state, event) pair is deterministically handled using VALID_TRANSITIONS.

    - Every event listed in VALID_TRANSITIONS[state] must return TransitionDecision.
    - Every event NOT listed in VALID_TRANSITIONS[state] must raise InvalidTransitionError.
    """
    context = PipelineContext(
        run_id="test_run_hex_32_chars_123456789",
        entry_stage="fresh",
        stage_1_frozen=False,
        max_retries=3,
        stage_providers={"table_selection": "koboldcpp", "query_generation": "koboldcpp", "validation": "koboldcpp"},
    )

    for state in PipelineState:
        valid_events = VALID_TRANSITIONS.get(state, set())
        for event_cls in ALL_EVENT_CLASSES:
            event = _make_dummy_event(event_cls)
            if event_cls in valid_events:
                decision = transition(state, event, context)
                assert isinstance(decision, TransitionDecision), (
                    f"State {state} with event {event_cls} did not return TransitionDecision"
                )
                assert isinstance(decision.next_state, PipelineState)
            else:
                with pytest.raises(InvalidTransitionError):
                    transition(state, event, context)


def test_worst_case_alternating_cap_and_headroom():
    """Simulate the worst-case alternating path (missing_info -> Stage 1 -> invalid query -> Stage 2).

    Must assert that:
    1. The path reaches a normal terminal error state (retry exhaustion) without the safety cap firing.
    2. The cap has a few transitions of headroom above that path.
    """
    context = PipelineContext(
        run_id="test_run_hex_32_chars_123456789",
        entry_stage="fresh",
        stage_1_frozen=False,
        max_retries=3,
        selected_objects=["customers"],
        stage_providers={"table_selection": "koboldcpp", "query_generation": "koboldcpp", "validation": "koboldcpp"},
    )
    cap = calculate_transition_cap(context.max_retries)

    state = PipelineState.SPELLING_CHECK
    transition_count = 0
    path = [state]

    while not state.is_terminal:
        transition_count += 1
        if transition_count > cap:
            pytest.fail(f"Safety cap {cap} was exceeded! Alternating loop failed to terminate naturally.")

        # Simulate stage event depending on current state
        if state == PipelineState.SPELLING_CHECK:
            event = SpellingCompleted(
                result=SpellingCorrectionResult(original_question="q", corrected_question="q")
            )
        elif state == PipelineState.QUESTION_GUARDRAIL:
            event = GuardrailCompleted(
                result=GuardrailResult(decision=GuardrailDecision.READ_QUERY)
            )
        elif state == PipelineState.TABLE_SELECTION:
            # Selector succeeds in picking tables
            event = TableSelectionCompleted(
                result=TableSelectionResult(selected_objects=["customers", "orders"], sufficient=True)
            )
        elif state == PipelineState.QUERY_GENERATION:
            # Alternating: if selector retries remain, trigger missing_information backtrack
            if context.selector_retries < context.max_retries:
                event = QueryMissingInfo(
                    missing_info=GeneratorMissingInfo(missing_tables=["orders"], reason="Need orders table")
                )
            else:
                # Selector retries are exhausted; generator outputs a query
                event = QueryGenerationCompleted(
                    query=GeneratedQuery(
                        database_type=DatabaseType.POSTGRESQL,
                        raw_query="SELECT * FROM customers;",
                        formatted_query="SELECT * FROM customers;",
                    )
                )
        elif state == PipelineState.QUERY_VALIDATION:
            # Validator repeatedly marks query invalid to exhaust validation retries
            event = ValidationCompleted(
                result=ValidationResult(
                    valid=False,
                    error_type=ValidationErrorType.SYNTAX_ERROR,
                    issues=["Syntax error near FROM"],
                    suggestion="Check query syntax",
                )
            )
        else:
            pytest.fail(f"Unexpected state in alternating path: {state}")

        decision = transition(state, event, context)
        context.apply_updates(decision.updates)
        context.record_event_artifacts(event)
        state = decision.next_state
        path.append(state)

    # 1. Must reach normal TERMINAL_ERROR via retry exhaustion
    assert state == PipelineState.TERMINAL_ERROR
    assert "maximum retries" in (context.error or "").lower()

    # 2. Both retry counters must have reached max_retries
    assert context.selector_retries == context.max_retries
    assert context.validation_retries == context.max_retries

    # 3. Must terminate with a few transitions of headroom under the cap
    headroom = cap - transition_count
    assert headroom >= 2, f"Safety cap is too tight; headroom was {headroom} (expected >= 2)"
    assert headroom <= 10, f"Safety cap is too loose; headroom was {headroom} (expected <= 10)"
    assert transition_count == 17, f"Expected 17 transitions in worst-case path, got {transition_count}"


def test_happy_path_fresh_run_transitions():
    """Verify standard happy-path progression from SPELLING_CHECK to TERMINAL_SUCCESS."""
    context = PipelineContext(
        run_id="test_happy_path_run",
        max_retries=3,
        entry_stage="fresh",
    )

    # 1. Spelling -> Guardrail
    d1 = transition(
        PipelineState.SPELLING_CHECK,
        SpellingCompleted(result=SpellingCorrectionResult(original_question="q", corrected_question="q")),
        context,
    )
    assert d1.next_state == PipelineState.QUESTION_GUARDRAIL

    # 2. Guardrail -> Table Selection
    d2 = transition(
        PipelineState.QUESTION_GUARDRAIL,
        GuardrailCompleted(result=GuardrailResult(decision=GuardrailDecision.READ_QUERY)),
        context,
    )
    assert d2.next_state == PipelineState.TABLE_SELECTION

    # 3. Table Selection -> Query Generation
    d3 = transition(
        PipelineState.TABLE_SELECTION,
        TableSelectionCompleted(result=TableSelectionResult(selected_objects=["users"])),
        context,
    )
    assert d3.next_state == PipelineState.QUERY_GENERATION
    assert d3.updates.s1_system_retry == CLEAR

    # 4. Query Generation -> Query Validation
    d4 = transition(
        PipelineState.QUERY_GENERATION,
        QueryGenerationCompleted(
            query=GeneratedQuery(database_type=DatabaseType.POSTGRESQL, raw_query="SELECT 1;", formatted_query="SELECT 1;")
        ),
        context,
    )
    assert d4.next_state == PipelineState.QUERY_VALIDATION
    assert d4.updates.attempts == 1

    # 5. Query Validation -> Policy Check
    d5 = transition(
        PipelineState.QUERY_VALIDATION,
        ValidationCompleted(result=ValidationResult(valid=True, error_type=ValidationErrorType.VALID)),
        context,
    )
    assert d5.next_state == PipelineState.POLICY_CHECK
    assert d5.updates.s2_system_retry == CLEAR

    # 6. Policy Check -> Execution
    d6 = transition(
        PipelineState.POLICY_CHECK,
        PolicyCompleted(result=PolicyResult(allowed=True)),
        context,
    )
    assert d6.next_state == PipelineState.QUERY_EXECUTION

    # 7. Execution -> Terminal Success
    d7 = transition(
        PipelineState.QUERY_EXECUTION,
        ExecutionCompleted(result=ExecutionResult(row_count=1, columns=["id"], preview_rows=[{"id": 1}])),
        context,
    )
    assert d7.next_state == PipelineState.TERMINAL_SUCCESS


def test_guardrail_decisions():
    """Verify guardrail REJECT and BASIC decisions."""
    context = PipelineContext(run_id="test_guardrail")

    # Reject
    d_reject = transition(
        PipelineState.QUESTION_GUARDRAIL,
        GuardrailCompleted(result=GuardrailResult(decision=GuardrailDecision.REJECT, reason="Unsafe")),
        context,
    )
    assert d_reject.next_state == PipelineState.TERMINAL_ERROR
    assert "Sorry" in (d_reject.updates.error or "")

    # Basic
    d_basic = transition(
        PipelineState.QUESTION_GUARDRAIL,
        GuardrailCompleted(result=GuardrailResult(decision=GuardrailDecision.BASIC, reason="I am NL2AnyQuery.")),
        context,
    )
    assert d_basic.next_state == PipelineState.TERMINAL_SUCCESS
    assert d_basic.updates.basic_answer == "I am NL2AnyQuery."


def test_frozen_stage_1_blocks_generator_backtrack():
    """Verify that when Stage 1 is frozen (rerun at S2 or S3), missing schema info terminates immediately."""
    context = PipelineContext(
        run_id="test_frozen",
        stage_1_frozen=True,
        selector_retries=0,
        max_retries=3,
    )

    decision = transition(
        PipelineState.QUERY_GENERATION,
        QueryMissingInfo(missing_info=GeneratorMissingInfo(missing_tables=["invoices"], reason="Need invoices table")),
        context,
    )
    assert decision.next_state == PipelineState.TERMINAL_ERROR
    assert "Please rerun from Table Selection with guidance" in (decision.updates.error or "")


def test_unsafe_validation_terminates_immediately():
    """Verify that ValidationErrorType.UNSAFE query stops immediately without retry."""
    context = PipelineContext(run_id="test_unsafe", validation_retries=0, max_retries=3)

    decision = transition(
        PipelineState.QUERY_VALIDATION,
        ValidationCompleted(result=ValidationResult(valid=False, error_type=ValidationErrorType.UNSAFE, issues=["DROP TABLE detected"])),
        context,
    )
    assert decision.next_state == PipelineState.TERMINAL_ERROR
    assert "unsafe" in (decision.updates.error or "").lower()
    assert decision.updates.validation_retries is None


def test_context_updates_clear_sentinel():
    """Verify that CLEAR sentinel clears context fields to None without wiping unmentioned fields."""
    context = PipelineContext(
        run_id="test_clear",
        s1_system_retry="Existing retry message",
        s2_system_retry="Existing validator retry",
        error="Previous error",
    )

    updates = ContextUpdates(
        s1_system_retry=CLEAR,
        error="New error",
        # s2_system_retry is None (should remain untouched)
    )

    context.apply_updates(updates)

    assert context.s1_system_retry is None
    assert context.s2_system_retry == "Existing validator retry"
    assert context.error == "New error"


# ---------------------------------------------------------------------------
# Additions: Parametrized table test, Purity test, Success-on-last-attempt test
# ---------------------------------------------------------------------------

PARAMETRIZED_CASES = [
    # 1. Empty selection retry
    (
        "empty_selection_retry",
        PipelineState.TABLE_SELECTION,
        TableSelectionCompleted(TableSelectionResult(selected_objects=[])),
        {"selector_retries": 0, "max_retries": 3},
        PipelineState.TABLE_SELECTION,
        {
            "selector_retries": 1,
            "s1_system_retry": (
                "[SYSTEM RETRY FEEDBACK]:\nNo tables were selected. Please select all "
                "required tables from the database schema to answer the question."
            ),
        },
    ),
    # 2. Empty selection retry exhaustion
    (
        "empty_selection_exhaustion",
        PipelineState.TABLE_SELECTION,
        TableSelectionCompleted(TableSelectionResult(selected_objects=[])),
        {"selector_retries": 3, "max_retries": 3},
        PipelineState.TERMINAL_ERROR,
        {
            "error": "I could not find any relevant tables or collections in the database schema matching your question."
        },
    ),
    # 3. Policy rejection
    (
        "policy_rejection",
        PipelineState.POLICY_CHECK,
        PolicyCompleted(PolicyResult(allowed=False, reason="Destructive write statement rejected")),
        {},
        PipelineState.TERMINAL_ERROR,
        {"error": "Policy rejected query: Destructive write statement rejected"},
    ),
    # 4. Feedback guardrail accept routing: table_selection
    (
        "feedback_accept_route_s1",
        PipelineState.FEEDBACK_GUARDRAIL,
        FeedbackGuardrailCompleted([("table_selection", FeedbackGuardrailResult(decision="ACCEPT"))]),
        {"entry_stage": "table_selection"},
        PipelineState.TABLE_SELECTION,
        {},
    ),
    # 5. Feedback guardrail accept routing: query_generation
    (
        "feedback_accept_route_s2",
        PipelineState.FEEDBACK_GUARDRAIL,
        FeedbackGuardrailCompleted([("query_generation", FeedbackGuardrailResult(decision="ACCEPT"))]),
        {"entry_stage": "query_generation"},
        PipelineState.QUERY_GENERATION,
        {},
    ),
    # 6. Feedback guardrail accept routing: validation
    (
        "feedback_accept_route_s3",
        PipelineState.FEEDBACK_GUARDRAIL,
        FeedbackGuardrailCompleted([("validation", FeedbackGuardrailResult(decision="ACCEPT"))]),
        {"entry_stage": "validation"},
        PipelineState.QUERY_VALIDATION,
        {},
    ),
    # 6b. Validation fallthrough via explicit flag
    (
        "validation_fallthrough_via_flag",
        PipelineState.QUERY_VALIDATION,
        ValidationCompleted(ValidationResult(valid=False, error_type=ValidationErrorType.SYNTAX_ERROR, issues=["syntax error"])),
        {"stage_3_fallthrough_allowed": True},
        PipelineState.QUERY_GENERATION,
        {"stage_3_fallthrough_allowed": False, "validation_retries": 0, "attempts": 1},
    ),
    # 7. Failure: SpellingCheckFailed
    (
        "failure_spelling",
        PipelineState.SPELLING_CHECK,
        SpellingCheckFailed("Network timeout"),
        {},
        PipelineState.TERMINAL_ERROR,
        {"error": "Stage 1 (Spelling) failed: Network timeout"},
    ),
    # 8. Failure: GuardrailFailed
    (
        "failure_guardrail",
        PipelineState.QUESTION_GUARDRAIL,
        GuardrailFailed("Rate limit exceeded"),
        {"stage_providers": {"table_selection": "hf"}},
        PipelineState.TERMINAL_ERROR,
        {"error": "Stage 1 (Guardrail) failed using provider 'hf': Rate limit exceeded"},
    ),
    # 9. Failure: FeedbackGuardrailFailed
    (
        "failure_feedback_guardrail",
        PipelineState.FEEDBACK_GUARDRAIL,
        FeedbackGuardrailFailed("Classifier down"),
        {},
        PipelineState.TERMINAL_ERROR,
        {"error": "Feedback safety check failed: Classifier down"},
    ),
    # 10. Failure: TableSelectionFailed
    (
        "failure_table_selection",
        PipelineState.TABLE_SELECTION,
        TableSelectionFailed("JSON parse error"),
        {"stage_providers": {"table_selection": "gemini"}},
        PipelineState.TERMINAL_ERROR,
        {"error": "Stage 1 (Table Selection) failed using provider 'gemini': JSON parse error"},
    ),
    # 11. Failure: QueryGenerationFailed
    (
        "failure_query_generation",
        PipelineState.QUERY_GENERATION,
        QueryGenerationFailed("Context exceeded"),
        {"stage_providers": {"query_generation": "koboldcpp"}},
        PipelineState.TERMINAL_ERROR,
        {"error": "Stage 2 (Query Generation) failed using provider 'koboldcpp': Context exceeded"},
    ),
    # 12. Failure: ValidationFailed
    (
        "failure_validation",
        PipelineState.QUERY_VALIDATION,
        ValidationFailed("Parser crashed"),
        {"stage_providers": {"validation": "gemini"}},
        PipelineState.TERMINAL_ERROR,
        {"error": "Stage 3 (Validation) failed using provider 'gemini': Parser crashed"},
    ),
    # 13. Failure: PolicyFailed
    (
        "failure_policy",
        PipelineState.POLICY_CHECK,
        PolicyFailed("AST recursion depth exceeded"),
        {},
        PipelineState.TERMINAL_ERROR,
        {"error": "Safety policy check error: AST recursion depth exceeded"},
    ),
    # 14. Failure: ExecutionFailed
    (
        "failure_execution",
        PipelineState.QUERY_EXECUTION,
        ExecutionFailed("Connection refused"),
        {},
        PipelineState.TERMINAL_ERROR,
        {"error": "Database execution error: Connection refused"},
    ),
    # 15. Exact text: missing-info backtrack
    (
        "missing_info_backtrack_text",
        PipelineState.QUERY_GENERATION,
        QueryMissingInfo(GeneratorMissingInfo(missing_tables=["invoices"], reason="Need invoice total")),
        {"selected_objects": ["vendors"], "selector_retries": 0, "max_retries": 3, "stage_1_frozen": False},
        PipelineState.TABLE_SELECTION,
        {
            "selector_retries": 1,
            "s1_system_retry": (
                "[SYSTEM RETRY FEEDBACK]:\n"
                "The query generator determined that the previous selection of tables ['vendors'] "
                "lacks required schema information: Need invoice total. Missing tables/concepts: [invoices]. "
                "Please re-evaluate the full schema and select ALL necessary tables."
            ),
        },
    ),
    # 16. Exact text: validator retry
    (
        "validator_retry_text",
        PipelineState.QUERY_VALIDATION,
        ValidationCompleted(
            ValidationResult(
                valid=False,
                error_type=ValidationErrorType.SYNTAX_ERROR,
                issues=["Syntax error near WHERE"],
                suggestion="Add WHERE condition",
            )
        ),
        {"validation_retries": 0, "max_retries": 3},
        PipelineState.QUERY_GENERATION,
        {
            "validation_retries": 1,
            "s2_system_retry": (
                "[SYSTEM VALIDATOR FEEDBACK (RETRY)]:\n"
                "Syntax error near WHERE. Suggestion: Add WHERE condition"
            ),
        },
    ),
]


@pytest.mark.parametrize(
    "case_id,state,event,context_setup,expected_next_state,expected_updates",
    PARAMETRIZED_CASES,
)
def test_parameterized_transitions(
    case_id, state, event, context_setup, expected_next_state, expected_updates
):
    """Table-driven test verifying exact next_state and key ContextUpdates matching orchestrator.py strings."""
    context = PipelineContext(
        run_id="test_param_run",
        **context_setup,
    )

    decision = transition(state, event, context)

    assert decision.next_state == expected_next_state, f"Case {case_id}: unexpected next_state"
    for attr, expected_val in expected_updates.items():
        actual_val = getattr(decision.updates, attr)
        assert actual_val == expected_val, (
            f"Case {case_id}: mismatch on updates.{attr}.\n"
            f"Expected: {expected_val!r}\n"
            f"Actual:   {actual_val!r}"
        )


def test_transition_function_purity():
    """Verify that calling transition() never mutates the passed context in place."""
    context = PipelineContext(
        run_id="test_purity_run",
        selector_retries=1,
        validation_retries=2,
        attempts=3,
        selected_objects=["orders", "customers"],
        stage_providers={"table_selection": "koboldcpp", "query_generation": "koboldcpp", "validation": "koboldcpp"},
        s1_user_guidance="Initial guidance",
        s1_system_retry="Retry message",
    )

    for state, valid_events in VALID_TRANSITIONS.items():
        for event_cls in valid_events:
            event = _make_dummy_event(event_cls)
            snapshot = copy.deepcopy(context)

            decision = transition(state, event, context)

            assert isinstance(decision, TransitionDecision)
            # Assert context fields were untouched
            assert context.selector_retries == snapshot.selector_retries
            assert context.validation_retries == snapshot.validation_retries
            assert context.attempts == snapshot.attempts
            assert context.selected_objects == snapshot.selected_objects
            assert context.s1_user_guidance == snapshot.s1_user_guidance
            assert context.s1_system_retry == snapshot.s1_system_retry
            assert context.stage_providers == snapshot.stage_providers
            assert context.error == snapshot.error


def test_worst_case_alternating_cap_success_on_last_attempt():
    """Verify worst-case alternating path when the final attempt succeeds.

    Path progresses through:
    missing_info -> Stage 1 -> invalid query -> Stage 2 (until retries exhausted)
    Then succeeds on last attempt -> Policy Check -> Execution -> TERMINAL_SUCCESS.
    Asserts cap does not fire and positive headroom exists.
    """
    context = PipelineContext(
        run_id="test_worst_case_success",
        max_retries=3,
        selected_objects=["customers"],
        stage_providers={"table_selection": "koboldcpp", "query_generation": "koboldcpp", "validation": "koboldcpp"},
    )
    cap = calculate_transition_cap(context.max_retries)

    state = PipelineState.SPELLING_CHECK
    transition_count = 0
    path = [state]

    while not state.is_terminal:
        transition_count += 1
        if transition_count > cap:
            pytest.fail(f"Safety cap {cap} fired unexpectedly!")

        if state == PipelineState.SPELLING_CHECK:
            event = SpellingCompleted(
                result=SpellingCorrectionResult(original_question="q", corrected_question="q")
            )
        elif state == PipelineState.QUESTION_GUARDRAIL:
            event = GuardrailCompleted(
                result=GuardrailResult(decision=GuardrailDecision.READ_QUERY)
            )
        elif state == PipelineState.TABLE_SELECTION:
            event = TableSelectionCompleted(
                result=TableSelectionResult(selected_objects=["customers", "orders"], sufficient=True)
            )
        elif state == PipelineState.QUERY_GENERATION:
            if context.selector_retries < context.max_retries:
                # Alternate back to Table Selection
                event = QueryMissingInfo(
                    missing_info=GeneratorMissingInfo(missing_tables=["orders"], reason="Need orders")
                )
            else:
                # Selector retries are exhausted; generate valid query
                event = QueryGenerationCompleted(
                    query=GeneratedQuery(
                        database_type=DatabaseType.POSTGRESQL,
                        raw_query="SELECT * FROM customers;",
                        formatted_query="SELECT * FROM customers;",
                    )
                )
        elif state == PipelineState.QUERY_VALIDATION:
            if context.validation_retries < context.max_retries:
                # Alternate back to Query Generation
                event = ValidationCompleted(
                    result=ValidationResult(
                        valid=False,
                        error_type=ValidationErrorType.SYNTAX_ERROR,
                        issues=["Syntax error near FROM"],
                    )
                )
            else:
                # Last allowed validation attempt succeeds!
                event = ValidationCompleted(
                    result=ValidationResult(valid=True, error_type=ValidationErrorType.VALID)
                )
        elif state == PipelineState.POLICY_CHECK:
            event = PolicyCompleted(result=PolicyResult(allowed=True, reason="Safe"))
        elif state == PipelineState.QUERY_EXECUTION:
            event = ExecutionCompleted(
                result=ExecutionResult(row_count=1, columns=["id"], preview_rows=[{"id": 1}])
            )
        else:
            pytest.fail(f"Unexpected state in path: {state}")

        decision = transition(state, event, context)
        context.apply_updates(decision.updates)
        context.record_event_artifacts(event)
        state = decision.next_state
        path.append(state)

    # Must reach TERMINAL_SUCCESS
    assert state == PipelineState.TERMINAL_SUCCESS
    assert context.error is None

    # Safety cap did not fire
    assert transition_count < cap
    headroom = cap - transition_count
    assert headroom >= 2, f"Expected headroom >= 2, got {headroom}"
    assert transition_count == 19, f"Expected 19 transitions, got {transition_count}"


def test_worst_case_rerun_stage3_fallthrough_cap():
    """Simulate worst-case rerun starting at Stage 3 with invalid query, fallthrough to Stage 2, and retry exhaustion.

    Must assert:
    1. Reaches normal TERMINAL_ERROR via validation retry exhaustion.
    2. Safety cap does not fire.
    3. Positive headroom exists below the cap.
    """
    context = PipelineContext(
        run_id="test_worst_case_rerun_s3",
        is_rerun=True,
        entry_stage="validation",
        stage_1_frozen=True,
        stage_3_fallthrough_allowed=True,
        max_retries=3,
        selected_objects=["customers"],
        stage_providers={"table_selection": "koboldcpp", "query_generation": "koboldcpp", "validation": "koboldcpp"},
    )
    cap = calculate_transition_cap(context.max_retries)

    state = PipelineState.FEEDBACK_GUARDRAIL
    transition_count = 0
    path = [state]

    while not state.is_terminal:
        transition_count += 1
        if transition_count > cap:
            pytest.fail(f"Safety cap {cap} fired unexpectedly in rerun path!")

        if state == PipelineState.FEEDBACK_GUARDRAIL:
            event = FeedbackGuardrailCompleted(
                results=[("validation", FeedbackGuardrailResult(decision="ACCEPT"))]
            )
        elif state == PipelineState.QUERY_GENERATION:
            event = QueryGenerationCompleted(
                query=GeneratedQuery(
                    database_type=DatabaseType.POSTGRESQL,
                    raw_query="SELECT * FROM customers;",
                    formatted_query="SELECT * FROM customers;",
                )
            )
        elif state == PipelineState.QUERY_VALIDATION:
            event = ValidationCompleted(
                result=ValidationResult(
                    valid=False,
                    error_type=ValidationErrorType.SYNTAX_ERROR,
                    issues=["Invalid SQL syntax"],
                    suggestion="Fix syntax",
                )
            )
        else:
            pytest.fail(f"Unexpected state in rerun path: {state}")

        decision = transition(state, event, context)
        context.apply_updates(decision.updates)
        context.record_event_artifacts(event)
        state = decision.next_state
        path.append(state)

    # Must reach normal TERMINAL_ERROR via validation retry exhaustion
    assert state == PipelineState.TERMINAL_ERROR
    assert "maximum retries" in (context.error or "").lower()
    assert context.validation_retries == context.max_retries
    assert transition_count < cap
    headroom = cap - transition_count
    assert headroom >= 2, f"Expected headroom >= 2, got {headroom}"
    assert transition_count == 10, f"Expected 10 transitions in Stage 3 fallthrough retry exhaustion, got {transition_count}"

