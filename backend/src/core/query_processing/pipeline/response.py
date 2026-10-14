"""Assembly of the single response shape every exit from the pipeline returns.

A run can end at any stage -- rejected by the guardrail, unable to find tables,
failing validation, or with rows -- and all of them answer with the same
`PipelineResponse`, so the caller reads one shape and the fields that never got
filled are simply empty.
"""

from __future__ import annotations

from typing import Any

from backend.src.data.models.schema import DatabaseType, Relationship
from backend.src.schemas.pipeline import (
    CandidateTable,
    ExecutionResult,
    GeneratedQuery,
    GuardrailResult,
    PipelineResponse,
    PolicyResult,
    RelevantSchema,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
    TableApprovalRecord,
    TableApprovalRequest,
    TableNeighborhood,
    ValidationResult,
)
from backend.src.utils.text_utils import clean_unreadable_characters, strip_ansi_escapes

# Engine names that carry no more information than the engine type itself, so a
# response labelled with one is labelled with the type instead.
_GENERIC_TARGET_NAMES = ("postgres", "postgresql", "mongo", "mongodb")


def _clean(value: Any) -> str:
    return clean_unreadable_characters(strip_ansi_escapes(str(value)))


def _sanitize_validation(validation: ValidationResult | None) -> None:
    """Strip terminal escapes from validator text, in place.

    Validator output can carry whatever a database driver printed, and it goes
    on to a frontend and back into a prompt.
    """
    if validation is None:
        return
    if validation.issues:
        validation.issues = [_clean(i) for i in validation.issues if i]
    if validation.suggestion:
        validation.suggestion = _clean(validation.suggestion)


def build_response(
    db_type: DatabaseType,
    question: str,
    guardrail: GuardrailResult,
    semantic: SemanticAnalysisResult | None = None,
    linguistic: dict[str, Any] | None = None,
    candidates: list[CandidateTable] | None = None,
    selected: list[str] | None = None,
    selection_retries: int = 0,
    selected_relationships: list[Relationship] | None = None,
    connector_objects: list[str] | None = None,
    unjoinable_objects: list[str] | None = None,
    selection_notice: str | None = None,
    table_neighborhood: TableNeighborhood | None = None,
    status: str = "complete",
    approval: TableApprovalRequest | None = None,
    table_approval: TableApprovalRecord | None = None,
    schema: RelevantSchema | None = None,
    plan: Any = None,
    query: GeneratedQuery | None = None,
    attempts: int = 1,
    validation_retries: int = 0,
    validation: ValidationResult | None = None,
    policy: PolicyResult | None = None,
    results: ExecutionResult | None = None,
    basic_answer: str | None = None,
    error: str | None = None,
    target_name: str | None = None,
    provider: str | None = None,
    providers: dict[str, str] | None = None,
    spelling_correction: SpellingCorrectionResult | None = None,
) -> PipelineResponse:
    """Fill a `PipelineResponse` from whatever the run managed to produce."""
    cand_list = candidates or []
    _sanitize_validation(validation)
    db_label = (
        target_name
        if (target_name and target_name not in _GENERIC_TARGET_NAMES)
        else db_type.value
    )
    return PipelineResponse(
        database=db_label,
        question=question,
        provider=provider,
        providers=providers,
        guardrail=guardrail,
        spelling_correction=spelling_correction,
        basic_answer=basic_answer,
        semantic_analysis=semantic,
        linguistic_analysis=linguistic,
        candidate_tables=cand_list,
        candidate_objects=[c.table_name for c in cand_list],
        selected_objects=selected or [],
        selection_retries=selection_retries,
        selected_relationships=selected_relationships or [],
        connector_objects=connector_objects or [],
        unjoinable_objects=unjoinable_objects or [],
        selection_notice=selection_notice,
        table_neighborhood=table_neighborhood,
        status=status,
        approval=approval,
        table_approval=table_approval,
        relevant_schema=schema.model_dump() if schema else None,
        query_plan=plan,
        generated_query=query.formatted_query if query else None,
        attempts=attempts,
        validation_retries=validation_retries,
        validation=validation,
        policy=policy,
        results=results,
        error=error,
    )
