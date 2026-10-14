"""Stage handler adapters translating stage execution and exceptions into typed FSM events."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from backend.src.core.query_processing.pipeline.feedback import compose_feedback_prompt
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
    StageEvent,
    TableSelectionCompleted,
    TableSelectionFailed,
    ValidationCompleted,
    ValidationFailed,
)
from backend.src.core.query_processing.pipeline.fsm.states import PipelineState
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import (
    GeneratorMissingInfo,
    GuardrailDecision,
)


class StageAdapter(ABC):
    """Abstract base adapter for executing a stage within the FSM."""

    @abstractmethod
    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        """Execute the stage and return a tuple of (event, prompt_tokens).

        Raises:
            asyncio.CancelledError: Propagated without swallowing.
        """


class SpellingAdapter(StageAdapter):
    """Executes SymSpell spelling correction."""

    def __init__(self, checker: Any) -> None:
        self.checker = checker

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            res = self.checker.check(question=context.question, jargons=context.jargons)
            return SpellingCompleted(result=res), 0
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return SpellingCheckFailed(error=str(err)), 0


class QuestionGuardrailAdapter(StageAdapter):
    """Executes initial query intent classification guardrail."""

    def __init__(self, guardrail: Any) -> None:
        self.guardrail = guardrail

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            res = await self.guardrail.classify(context.processed_question)
            tok = getattr(self.guardrail, "last_prompt_tokens", 0)
            self.guardrail.last_prompt_tokens = 0
            basic_ans = None
            if res.decision == GuardrailDecision.BASIC and hasattr(self.guardrail, "handle_basic_question"):
                db_key = context.target_name or ""
                db_name = context.schema.database_name if context.schema else ""
                basic_ans = self.guardrail.handle_basic_question(
                    question=context.processed_question,
                    database_type=db_key,
                    database_name=db_name,
                )
            return GuardrailCompleted(result=res, basic_answer=basic_ans), tok
        except asyncio.CancelledError:
            raise
        except Exception as err:
            s1_name = context.stage_providers.get("table_selection", "default")
            return GuardrailFailed(error=str(err), provider=s1_name), 0


class FeedbackGuardrailAdapter(StageAdapter):
    """Executes human feedback safety guardrail for partial reruns."""

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            results: list[tuple[str, Any]] = []
            total_tok = 0
            for stg, fb_text in context.applicable_feedback.items():
                res = await self.classifier.classify(feedback=fb_text, stage=stg)
                results.append((stg, res))
                total_tok += getattr(res, "prompt_tokens", 0)
            return FeedbackGuardrailCompleted(results=results), total_tok
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return FeedbackGuardrailFailed(error=str(err)), 0


class TableSelectionAdapter(StageAdapter):
    """Executes Stage 1 Table Selector with active feedback."""

    def __init__(self, selector: Any) -> None:
        self.selector = selector

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            fb = compose_feedback_prompt(context.s1_user_guidance, context.s1_system_retry)
            res = await self.selector.select(
                question=context.processed_question,
                schema=context.schema,
                feedback=fb,
                target_key=context.target_name,
            )
            tok = getattr(self.selector, "last_prompt_tokens", 0)
            self.selector.last_prompt_tokens = 0
            return TableSelectionCompleted(result=res), tok
        except asyncio.CancelledError:
            raise
        except Exception as err:
            s1_name = context.stage_providers.get("table_selection", "default")
            return TableSelectionFailed(error=str(err), provider=s1_name), 0


class QueryGenerationAdapter(StageAdapter):
    """Executes Stage 2 Query Generator (PostgreSQL or MongoDB) with active feedback."""

    def __init__(self, postgres_gen: Any, mongo_gen: Any) -> None:
        self.postgres_gen = postgres_gen
        self.mongo_gen = mongo_gen

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        generator = (
            self.postgres_gen
            if context.db_type == DatabaseType.POSTGRESQL
            else self.mongo_gen
        )
        try:
            fb = compose_feedback_prompt(context.s2_user_guidance, context.s2_system_retry)
            res = await generator.generate_query(
                question=context.processed_question,
                schema=context.relevant_schema,
                feedback=fb,
            )
            tok = getattr(generator, "last_prompt_tokens", 0)
            generator.last_prompt_tokens = 0
            if isinstance(res, GeneratorMissingInfo):
                return QueryMissingInfo(missing_info=res), tok
            return QueryGenerationCompleted(query=res), tok
        except asyncio.CancelledError:
            raise
        except Exception as err:
            s2_name = context.stage_providers.get("query_generation", "default")
            return QueryGenerationFailed(error=str(err), provider=s2_name), 0


class QueryValidationAdapter(StageAdapter):
    """Executes Stage 3 Query Validator with active feedback and fail-fast syntax checking."""

    def __init__(self, validator: Any) -> None:
        self.validator = validator

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            fb = compose_feedback_prompt(context.s3_user_guidance, context.s3_system_retry)
            res = await self.validator.validate(
                question=context.processed_question,
                generated_query=context.generated_query,
                schema=context.relevant_schema,
                feedback=fb,
                fail_fast=True,
            )
            tok = getattr(self.validator, "last_prompt_tokens", 0)
            if hasattr(self.validator, "postgres_validator"):
                self.validator.postgres_validator.last_prompt_tokens = 0
            if hasattr(self.validator, "mongo_validator"):
                self.validator.mongo_validator.last_prompt_tokens = 0
            try:
                self.validator.last_prompt_tokens = 0
            except (AttributeError, TypeError):
                pass
            return ValidationCompleted(result=res), tok
        except asyncio.CancelledError:
            raise
        except Exception as err:
            s3_name = context.stage_providers.get("validation", "default")
            return ValidationFailed(error=str(err), provider=s3_name), 0


class PolicyAdapter(StageAdapter):
    """Executes deterministic read-only Safety Policy Validator."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            res = self.policy.check(context.generated_query, schema=context.relevant_schema)
            return PolicyCompleted(result=res), 0
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return PolicyFailed(error=str(err)), 0


class ExecutionAdapter(StageAdapter):
    """Executes verified query on target database and processes result rows."""

    def __init__(self, executor: Any, results_processor: Any) -> None:
        self.executor = executor
        self.results_processor = results_processor

    async def execute(self, context: PipelineContext) -> tuple[StageEvent, int]:
        try:
            conn = context.database_target.connection if context.database_target else None
            mongo_db = context.database_target.mongo_database if context.database_target else None
            columns, raw_rows = self.executor.execute(
                context.generated_query,
                connection=conn,
                database=mongo_db,
            )
            res = self.results_processor.process(columns, raw_rows)
            return ExecutionCompleted(result=res), 0
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return ExecutionFailed(error=str(err)), 0


@dataclass
class StageAdapters:
    """Registry mapping pipeline states to their respective stage adapters."""

    adapters: dict[PipelineState, StageAdapter]

    async def execute(self, state: PipelineState, context: PipelineContext) -> tuple[StageEvent, int]:
        """Execute the adapter mapped to the specified state."""
        adapter = self.adapters.get(state)
        if not adapter:
            raise ValueError(f"No adapter registered for state '{state.value}'")
        return await adapter.execute(context)
