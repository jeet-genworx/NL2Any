"""Pure deterministic transition function and valid transition mappings for the pipeline FSM."""

from backend.src.core.query_processing.pipeline.fsm.context import (
    CLEAR,
    ContextUpdates,
    PipelineContext,
    TransitionDecision,
)
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
    StageEvent,
    TableSelectionCompleted,
    TableSelectionFailed,
    ValidationCompleted,
    ValidationFailed,
)
from backend.src.core.query_processing.pipeline.fsm.states import PipelineState
from backend.src.schemas.pipeline import GuardrailDecision, ValidationErrorType
from backend.src.utils.text_utils import clean_unreadable_characters, strip_ansi_escapes


class InvalidTransitionError(Exception):
    """Raised when an event is received that is not valid for the current state."""


# ---------------------------------------------------------------------------
# Canonical VALID_TRANSITIONS mapping shared by transition() and completeness tests
# ---------------------------------------------------------------------------
VALID_TRANSITIONS: dict[PipelineState, set[type[StageEvent]]] = {
    PipelineState.SPELLING_CHECK: {
        SpellingCompleted,
        SpellingCheckFailed,
    },
    PipelineState.QUESTION_GUARDRAIL: {
        GuardrailCompleted,
        GuardrailFailed,
    },
    PipelineState.FEEDBACK_GUARDRAIL: {
        FeedbackGuardrailCompleted,
        FeedbackGuardrailFailed,
    },
    PipelineState.TABLE_SELECTION: {
        TableSelectionCompleted,
        TableSelectionFailed,
    },
    PipelineState.QUERY_GENERATION: {
        QueryGenerationCompleted,
        QueryMissingInfo,
        QueryGenerationFailed,
    },
    PipelineState.QUERY_VALIDATION: {
        ValidationCompleted,
        ValidationFailed,
    },
    PipelineState.POLICY_CHECK: {
        PolicyCompleted,
        PolicyFailed,
    },
    PipelineState.QUERY_EXECUTION: {
        ExecutionCompleted,
        ExecutionFailed,
    },
    PipelineState.TERMINAL_SUCCESS: set(),
    PipelineState.TERMINAL_ERROR: set(),
}


def calculate_transition_cap(max_retries: int) -> int:
    """Calculate maximum permitted FSM transitions with headroom for bounded retries."""
    return 5 * max_retries + 8


def transition(
    state: PipelineState,
    event: StageEvent,
    context: PipelineContext,
) -> TransitionDecision:
    """Evaluate pure transition decision from current state and event.

    Raises:
        InvalidTransitionError: If the event type is not allowed for the state in VALID_TRANSITIONS.
    """
    valid_event_types = VALID_TRANSITIONS.get(state, set())
    if type(event) not in valid_event_types:
        raise InvalidTransitionError(
            f"Event {type(event).__name__} is not valid for state {state.value}."
        )

    # 1. Spelling Check
    if state == PipelineState.SPELLING_CHECK:
        if isinstance(event, SpellingCompleted):
            return TransitionDecision(next_state=PipelineState.QUESTION_GUARDRAIL)
        elif isinstance(event, SpellingCheckFailed):
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(error=f"Stage 1 (Spelling) failed: {event.error}"),
            )

    # 2. Question Guardrail
    elif state == PipelineState.QUESTION_GUARDRAIL:
        if isinstance(event, GuardrailCompleted):
            decision = event.result.decision
            if decision == GuardrailDecision.REJECT:
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_ERROR,
                    updates=ContextUpdates(error="Sorry, I can't help with this."),
                )
            elif decision == GuardrailDecision.BASIC:
                ans = getattr(event, "basic_answer", None) or event.result.reason or "I am NL2AnyQuery."
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_SUCCESS,
                    updates=ContextUpdates(basic_answer=ans),
                )
            else:  # READ_QUERY
                return TransitionDecision(next_state=PipelineState.TABLE_SELECTION)
        elif isinstance(event, GuardrailFailed):
            s1_name = getattr(event, "provider", None) or context.stage_providers.get("table_selection", "default")
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(
                    error=f"Stage 1 (Guardrail) failed using provider '{s1_name}': {event.error}"
                ),
            )

    # 3. Feedback Guardrail (Rerun Only)
    elif state == PipelineState.FEEDBACK_GUARDRAIL:
        if isinstance(event, FeedbackGuardrailCompleted):
            rejected_items = [
                (stg, res)
                for stg, res in event.results
                if getattr(res, "decision", "") == "REJECT"
            ]
            if rejected_items:
                stg, res = rejected_items[0]
                reason = getattr(res, "reason", "Rejected by safety guidelines")
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_ERROR,
                    updates=ContextUpdates(
                        error=f"Feedback for {stg} was rejected by the safety guardrail: {reason}"
                    ),
                )

            # Jump to entry stage
            if context.entry_stage == "query_generation":
                return TransitionDecision(next_state=PipelineState.QUERY_GENERATION)
            elif context.entry_stage == "validation":
                return TransitionDecision(next_state=PipelineState.QUERY_VALIDATION)
            else:
                return TransitionDecision(next_state=PipelineState.TABLE_SELECTION)

        elif isinstance(event, FeedbackGuardrailFailed):
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(error=f"Feedback safety check failed: {event.error}"),
            )

    # 4. Table Selection (Stage 1)
    elif state == PipelineState.TABLE_SELECTION:
        if isinstance(event, TableSelectionCompleted):
            if not event.result.selected_objects:
                reason = event.result.reason or ""
                if reason.startswith("Table selection failed due to error:"):
                    err_detail = reason.replace("Table selection failed due to error:", "").strip()
                    s1_name = context.stage_providers.get("table_selection", "default")
                    return TransitionDecision(
                        next_state=PipelineState.TERMINAL_ERROR,
                        updates=ContextUpdates(
                            error=f"Stage 1 (Table Selection) failed using provider '{s1_name}': {err_detail}"
                        ),
                    )
                if context.selector_retries >= context.max_retries:
                    fallback_err = (
                        reason
                        or "I could not find any relevant tables or collections in the database schema matching your question."
                    )
                    return TransitionDecision(
                        next_state=PipelineState.TERMINAL_ERROR,
                        updates=ContextUpdates(error=fallback_err),
                    )
                # Bounded retry in Table Selection
                retry_msg = (
                    "No tables were selected. Please select all required tables "
                    "from the database schema to answer the question."
                )
                return TransitionDecision(
                    next_state=PipelineState.TABLE_SELECTION,
                    updates=ContextUpdates(
                        selector_retries=context.selector_retries + 1,
                        s1_system_retry=f"[SYSTEM RETRY FEEDBACK]:\n{retry_msg}",
                    ),
                )

            # Successfully selected tables
            return TransitionDecision(
                next_state=PipelineState.QUERY_GENERATION,
                updates=ContextUpdates(s1_system_retry=CLEAR),
            )

        elif isinstance(event, TableSelectionFailed):
            s1_name = getattr(event, "provider", None) or context.stage_providers.get("table_selection", "default")
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(
                    error=f"Stage 1 (Table Selection) failed using provider '{s1_name}': {event.error}"
                ),
            )

    # 5. Query Generation (Stage 2)
    elif state == PipelineState.QUERY_GENERATION:
        if isinstance(event, QueryGenerationCompleted):
            return TransitionDecision(
                next_state=PipelineState.QUERY_VALIDATION,
                updates=ContextUpdates(attempts=context.validation_retries + 1),
            )

        elif isinstance(event, QueryMissingInfo):
            if context.stage_1_frozen:
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_ERROR,
                    updates=ContextUpdates(
                        error=(
                            f"Required schema information missing: {event.missing_info.reason}. "
                            "Please rerun from Table Selection with guidance."
                        )
                    ),
                )
            if context.selector_retries >= context.max_retries:
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_ERROR,
                    updates=ContextUpdates(
                        error=(
                            "Required schema information could not be identified after maximum retries: "
                            f"{event.missing_info.reason}"
                        )
                    ),
                )

            # Backtrack to Table Selection
            missing_tables = event.missing_info.missing_tables
            missing_str = ", ".join(missing_tables) if missing_tables else "additional tables"
            sys_fb = (
                f"The query generator determined that the previous selection of tables {context.selected_objects} "
                f"lacks required schema information: {event.missing_info.reason}. "
                f"Missing tables/concepts: [{missing_str}]. Please re-evaluate the full schema and select ALL necessary tables."
            )
            return TransitionDecision(
                next_state=PipelineState.TABLE_SELECTION,
                updates=ContextUpdates(
                    selector_retries=context.selector_retries + 1,
                    s1_system_retry=f"[SYSTEM RETRY FEEDBACK]:\n{sys_fb}",
                ),
            )

        elif isinstance(event, QueryGenerationFailed):
            s2_name = getattr(event, "provider", None) or context.stage_providers.get("query_generation", "default")
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(
                    error=f"Stage 2 (Query Generation) failed using provider '{s2_name}': {event.error}"
                ),
            )

    # 6. Query Validation (Stage 3)
    elif state == PipelineState.QUERY_VALIDATION:
        if isinstance(event, ValidationCompleted):
            if event.result.valid:
                return TransitionDecision(
                    next_state=PipelineState.POLICY_CHECK,
                    updates=ContextUpdates(
                        s2_system_retry=CLEAR,
                        stage_3_fallthrough_allowed=False,
                    ),
                )

            if event.result.error_type == ValidationErrorType.UNSAFE:
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_ERROR,
                    updates=ContextUpdates(error="Generated query is unsafe. Execution rejected."),
                )

            # Check if this was the initial validation check on a loaded query when starting at validation
            if context.stage_3_fallthrough_allowed or (context.entry_stage == "validation" and not context.initial_val_checked):
                return TransitionDecision(
                    next_state=PipelineState.QUERY_GENERATION,
                    updates=ContextUpdates(
                        stage_3_fallthrough_allowed=False,
                        initial_val_checked=True,
                        validation_retries=0,
                        attempts=1,
                    ),
                )

            if context.validation_retries >= context.max_retries:
                return TransitionDecision(
                    next_state=PipelineState.TERMINAL_ERROR,
                    updates=ContextUpdates(
                        error="Query failed validation after maximum retries. Manual review required."
                    ),
                )

            # Retry Query Generation
            clean_issues = [
                clean_unreadable_characters(strip_ansi_escapes(str(i)))
                for i in event.result.issues
                if i
            ]
            feedback_msg = "; ".join(clean_issues)
            if event.result.suggestion:
                clean_sugg = clean_unreadable_characters(
                    strip_ansi_escapes(str(event.result.suggestion))
                )
                feedback_msg += f". Suggestion: {clean_sugg}"
            clean_fb = clean_unreadable_characters(strip_ansi_escapes(feedback_msg))
            system_retry_part = f"[SYSTEM VALIDATOR FEEDBACK (RETRY)]:\n{clean_fb}"

            return TransitionDecision(
                next_state=PipelineState.QUERY_GENERATION,
                updates=ContextUpdates(
                    validation_retries=context.validation_retries + 1,
                    s2_system_retry=system_retry_part,
                ),
            )

        elif isinstance(event, ValidationFailed):
            s3_name = getattr(event, "provider", None) or context.stage_providers.get("validation", "default")
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(
                    error=f"Stage 3 (Validation) failed using provider '{s3_name}': {event.error}"
                ),
            )

    # 7. Safety Policy Check (Stage 4)
    elif state == PipelineState.POLICY_CHECK:
        if isinstance(event, PolicyCompleted):
            if event.result.allowed:
                return TransitionDecision(next_state=PipelineState.QUERY_EXECUTION)
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(error=f"Policy rejected query: {event.result.reason}"),
            )
        elif isinstance(event, PolicyFailed):
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(error=f"Safety policy check error: {event.error}"),
            )

    # 8. Query Execution (Stage 5)
    elif state == PipelineState.QUERY_EXECUTION:
        if isinstance(event, ExecutionCompleted):
            return TransitionDecision(next_state=PipelineState.TERMINAL_SUCCESS)
        elif isinstance(event, ExecutionFailed):
            return TransitionDecision(
                next_state=PipelineState.TERMINAL_ERROR,
                updates=ContextUpdates(error=f"Database execution error: {event.error}"),
            )

    raise InvalidTransitionError(
        f"Unhandled event {type(event).__name__} for state {state.value}."
    )
