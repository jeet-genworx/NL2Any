"""Finite State Machine states for the natural-language-to-query pipeline."""

from enum import Enum


class PipelineState(str, Enum):
    """Discrete states of the NL2AnyQuery pipeline."""

    SPELLING_CHECK = "spelling_check"
    QUESTION_GUARDRAIL = "question_guardrail"
    FEEDBACK_GUARDRAIL = "feedback_guardrail"
    TABLE_SELECTION = "table_selection"
    QUERY_GENERATION = "query_generation"
    QUERY_VALIDATION = "query_validation"
    POLICY_CHECK = "policy_check"
    QUERY_EXECUTION = "query_execution"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_ERROR = "terminal_error"

    @property
    def is_terminal(self) -> bool:
        """Return True if this is an end state of the pipeline."""
        return self in (PipelineState.TERMINAL_SUCCESS, PipelineState.TERMINAL_ERROR)
