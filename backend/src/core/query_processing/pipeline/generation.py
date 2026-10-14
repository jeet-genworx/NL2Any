"""The planner -> generator -> validator loop, and its targeted retries.

A failed validation is not retried blindly. The validator says which stage was
at fault, so a planning mistake re-plans and a generation mistake re-generates
against the same plan, each with the issue text as feedback.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from backend.src.core.query_processing.pipeline.planner import QueryPlanner
from backend.src.core.query_processing.pipeline.validator import QueryValidator
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    QueryPlan,
    QuestionAnalysis,
    RelevantSchema,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.utils.text_utils import clean_unreadable_characters, strip_ansi_escapes

logger = logging.getLogger(__name__)


class GenerationOutcome(NamedTuple):
    """The last plan, query and verdict of the loop, with why it stopped."""

    plan: Any
    query: GeneratedQuery | None
    validation: ValidationResult | None
    retries: int
    error: str | None


def _feedback_from(validation: ValidationResult) -> str:
    """The validator's complaint, as text safe to put back in a prompt.

    Validator issues can quote whatever a database driver printed, escape codes
    included, and this string goes into the next prompt and on to a frontend.
    """
    clean_issues = [
        clean_unreadable_characters(strip_ansi_escapes(str(i)))
        for i in validation.issues
        if i
    ]
    message = "; ".join(clean_issues)
    if validation.suggestion:
        clean_sugg = clean_unreadable_characters(strip_ansi_escapes(str(validation.suggestion)))
        message += f". Suggestion: {clean_sugg}"
    return clean_unreadable_characters(strip_ansi_escapes(message))


async def plan_generate_validate(
    *,
    question: str,
    analysis: QuestionAnalysis,
    relevant_schema: RelevantSchema,
    db_type: DatabaseType,
    planner: QueryPlanner,
    generator: Any,
    validator: QueryValidator,
    max_retries: int,
    initial_plan: QueryPlan | None = None,
) -> GenerationOutcome:
    """Query Planner -> Generator -> Validator bounded targeted retry loop."""
    query_plan = initial_plan
    if query_plan is None:
        query_plan = await planner.plan(
            question_analysis=analysis,
            relevant_schema=relevant_schema,
            database_type=db_type,
        )

    last_query: GeneratedQuery | None = None
    last_val: ValidationResult | None = None
    gen_feedback: str | None = None
    planner_feedback: str | None = None
    retries = 0
    exhausted = "Query failed validation after maximum retries. Manual review required."

    while retries <= max_retries:
        if query_plan is None:
            query_plan = await planner.plan(
                question_analysis=analysis,
                relevant_schema=relevant_schema,
                database_type=db_type,
                feedback=planner_feedback,
            )
            planner_feedback = None

        try:
            last_query = await generator.generate_query(
                question=question,
                plan=query_plan,
                schema=relevant_schema,
                feedback=gen_feedback,
            )
            gen_feedback = None
        except Exception as gen_err:
            if retries >= max_retries:
                return GenerationOutcome(
                    query_plan,
                    last_query,
                    last_val,
                    retries,
                    f"Query generation failed after maximum retries: {gen_err}",
                )
            retries += 1
            gen_feedback = f"Generation error: {gen_err}"
            continue

        last_val = await validator.validate(
            question=question,
            plan=query_plan,
            generated_query=last_query,
            schema=relevant_schema,
        )

        if last_val.valid:
            return GenerationOutcome(query_plan, last_query, last_val, retries, None)

        if last_val.error_type == ValidationErrorType.UNSAFE:
            return GenerationOutcome(
                query_plan,
                last_query,
                last_val,
                retries,
                "Generated query is unsafe. Execution rejected.",
            )

        if retries >= max_retries:
            return GenerationOutcome(query_plan, last_query, last_val, retries, exhausted)

        retries += 1
        feedback_msg = _feedback_from(last_val)
        if last_val.error_type == ValidationErrorType.PLANNER_ERROR:
            # The plan itself was wrong, so there is nothing to re-generate
            # against: it is thrown away and re-planned with the complaint.
            planner_feedback = feedback_msg
            query_plan = None
        else:
            gen_feedback = feedback_msg

    return GenerationOutcome(query_plan, last_query, last_val, retries, exhausted)
