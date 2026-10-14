"""Typed events emitted by stage adapters and consumed by the FSM transition function."""

from dataclasses import dataclass

from backend.src.core.query_processing.pipeline.feedback_guardrail import (
    FeedbackGuardrailResult,
)
from backend.src.schemas.pipeline import (
    ExecutionResult,
    GeneratedQuery,
    GeneratorMissingInfo,
    GuardrailResult,
    PolicyResult,
    SpellingCorrectionResult,
    TableSelectionResult,
    ValidationResult,
)


@dataclass(frozen=True)
class StageEvent:
    """Base class for all events emitted by stage execution."""


# ---------------------------------------------------------------------------
# Spelling Stage Events
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpellingCompleted(StageEvent):
    result: SpellingCorrectionResult


@dataclass(frozen=True)
class SpellingCheckFailed(StageEvent):
    error: str


# ---------------------------------------------------------------------------
# Question Guardrail Events
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardrailCompleted(StageEvent):
    result: GuardrailResult
    basic_answer: str | None = None


@dataclass(frozen=True)
class GuardrailFailed(StageEvent):
    error: str
    provider: str | None = None


# ---------------------------------------------------------------------------
# Feedback Guardrail Events (Rerun Only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeedbackGuardrailCompleted(StageEvent):
    # List of (stage_name, FeedbackGuardrailResult)
    results: list[tuple[str, FeedbackGuardrailResult]]


@dataclass(frozen=True)
class FeedbackGuardrailFailed(StageEvent):
    error: str


# ---------------------------------------------------------------------------
# Table Selection Events (Stage 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSelectionCompleted(StageEvent):
    result: TableSelectionResult


@dataclass(frozen=True)
class TableSelectionFailed(StageEvent):
    error: str
    provider: str | None = None


# ---------------------------------------------------------------------------
# Query Generation Events (Stage 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QueryGenerationCompleted(StageEvent):
    query: GeneratedQuery


@dataclass(frozen=True)
class QueryMissingInfo(StageEvent):
    missing_info: GeneratorMissingInfo


@dataclass(frozen=True)
class QueryGenerationFailed(StageEvent):
    error: str
    provider: str | None = None


# ---------------------------------------------------------------------------
# Query Validation Events (Stage 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationCompleted(StageEvent):
    result: ValidationResult


@dataclass(frozen=True)
class ValidationFailed(StageEvent):
    error: str
    provider: str | None = None


# ---------------------------------------------------------------------------
# Policy Check Events (Stage 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyCompleted(StageEvent):
    result: PolicyResult


@dataclass(frozen=True)
class PolicyFailed(StageEvent):
    error: str


# ---------------------------------------------------------------------------
# Query Execution Events (Stage 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionCompleted(StageEvent):
    result: ExecutionResult


@dataclass(frozen=True)
class ExecutionFailed(StageEvent):
    error: str
