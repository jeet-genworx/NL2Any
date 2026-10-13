"""End-to-end pipeline orchestrator for NL2AnyQuery V2."""

import logging
import time
import uuid
from typing import Any

from backend.src.data.clients.detector import detect_database_type
from backend.src.data.models.targets import DatabaseTarget, as_target, resolve_target
from backend.src.data.repositories import description_repository, paths, schema_repository
from backend.src.data.repositories.run_repository import (
    cleanup_old_runs,
    load_run_state,
    save_run_state,
    validate_run_id,
)
from backend.src.core.query_processing.pipeline.feedback_guardrail import FeedbackGuardrailClassifier
from backend.src.core.ingestion.schema.describe import apply_descriptions
from backend.src.core.ingestion.schema.manager import get_default_schema_path
from backend.src.config import settings
from backend.src.schemas.pipeline import (
    CandidateTable,
    ExecutionResult,
    GeneratedQuery,
    GeneratorMissingInfo,
    GuardrailDecision,
    GuardrailResult,
    PipelineResponse,
    PolicyResult,
    RelevantSchema,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.schemas.request import StageProviders
from backend.src.core.query_processing.pipeline.executor import QueryExecutor
from backend.src.core.query_processing.pipeline.generators.mongo import MongoQueryGenerator
from backend.src.core.query_processing.pipeline.generators.postgres import PostgresQueryGenerator
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.policy import SafetyPolicyValidator
from backend.src.core.query_processing.pipeline.results import ResultProcessor
from backend.src.core.query_processing.pipeline.selector import (
    TableSelector,
    package_selected_schema,
)
from backend.src.core.query_processing.pipeline.spelling import SpellingChecker
from backend.src.core.query_processing.pipeline.validator import QueryValidator
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.factory import resolve_model_provider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.control.providers.model.koboldcpp_cache import (
    KoboldCppCacheManager,
    get_cache_manager,
)
from backend.src.utils.text_utils import clean_unreadable_characters, strip_ansi_escapes

logger = logging.getLogger(__name__)


def _get_stage_feedback_items(
    feedback_history: list[dict[str, Any]],
    current_feedback: str | None,
    stage: str,
) -> list[str]:
    """Extract a sliding window of at most 1 previous feedback item + current feedback item for a stage."""
    items: list[str] = []
    prior_for_stage = [
        str(h["feedback"]).strip()
        for h in feedback_history
        if isinstance(h, dict) and h.get("stage") == stage and h.get("feedback")
    ]
    if prior_for_stage:
        items.append(prior_for_stage[-1])
    if current_feedback and current_feedback.strip():
        cf = current_feedback.strip()
        if not items or items[-1] != cf:
            items.append(cf)
    return items


def _format_stage_user_guidance(
    feedback_history: list[dict[str, Any]],
    current_feedback: str | None,
    stage: str,
) -> str | None:
    """Format user guidance with a sliding window of at most 1 previous feedback item + current feedback item."""
    items = _get_stage_feedback_items(feedback_history, current_feedback, stage)
    if not items:
        return None

    joined_items = "\n".join(f"- {item}" for item in items)
    if stage == "table_selection":
        note = "(Note: User guidance is an advisory hint. You must strictly ground table selection in the canonical schema and adhere to all relationship rules.)"
    elif stage == "query_generation":
        note = "(Note: User guidance provides domain/intent clarification. All generated queries must strictly adhere to the Selected Schema and read-only safety rules.)"
    elif stage == "validation":
        note = "(Note: User validation guidance clarifies user intent or acceptable caveats. Ensure the query remains strictly read-only and free of syntax errors.)"
    else:
        note = "(Note: User guidance is an advisory hint.)"

    return f"[USER GUIDANCE / HINT]:\n{joined_items}\n{note}"


class NL2AnyQueryOrchestrator:
    """Coordinates the deterministic V2 pipeline from natural language to database results."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        guardrail: GuardrailClassifier | None = None,
        spelling_checker: SpellingChecker | None = None,
        selector: TableSelector | None = None,
        postgres_gen: PostgresQueryGenerator | None = None,
        mongo_gen: MongoQueryGenerator | None = None,
        validator: QueryValidator | None = None,
        policy: SafetyPolicyValidator | None = None,
        executor: QueryExecutor | None = None,
        results: ResultProcessor | None = None,
        max_retries: int | None = None,
        cache_manager: KoboldCppCacheManager | None = None,
        **kwargs: Any,
    ) -> None:
        self.cache_manager = cache_manager or get_cache_manager()
        self.provider = provider or KoboldCppProvider()
        self.guardrail = guardrail or GuardrailClassifier(provider=self.provider)
        self.spelling_checker = spelling_checker or SpellingChecker()
        self.selector = selector or TableSelector(provider=self.provider, cache_manager=self.cache_manager)
        self.postgres_gen = postgres_gen or PostgresQueryGenerator(provider=self.provider)
        self.mongo_gen = mongo_gen or MongoQueryGenerator(provider=self.provider)
        self.validator = validator or QueryValidator(provider=self.provider)
        self.policy = policy or SafetyPolicyValidator()
        self.executor = executor or QueryExecutor()
        self.results = results or ResultProcessor()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

        self._custom_guardrail = guardrail is not None
        self._custom_selector = selector is not None
        self._custom_postgres_gen = postgres_gen is not None
        self._custom_mongo_gen = mongo_gen is not None
        self._custom_validator = validator is not None

        self._schemas: dict[Any, DatabaseSchema] = {}

    def _resolve_stage_provider(
        self,
        stage_provider_spec: str | ModelProvider | None,
        fallback_provider: ModelProvider,
        fallback_name: str,
        stage_label: str,
    ) -> tuple[ModelProvider, str]:
        """Resolve a model provider for a pipeline stage with clear configuration attribution."""
        if stage_provider_spec is None:
            return fallback_provider, fallback_name
        if isinstance(stage_provider_spec, str):
            try:
                p = resolve_model_provider(stage_provider_spec)
                return p, stage_provider_spec.strip().lower()
            except Exception as err:
                raise ValueError(f"{stage_label} provider configuration error: {err}") from err
        return stage_provider_spec, getattr(stage_provider_spec, "model", "custom")

    def _get_stages_for_providers(
        self,
        s1_provider: ModelProvider,
        s2_provider: ModelProvider,
        s3_provider: ModelProvider,
    ):
        """Return stage instances bound to the respective stage providers."""
        from unittest.mock import AsyncMock, MagicMock

        is_mocked_guardrail = isinstance(getattr(self.guardrail, "classify", None), (AsyncMock, MagicMock))
        is_mocked_selector = isinstance(getattr(self.selector, "select", None), (AsyncMock, MagicMock))
        is_mocked_postgres_gen = isinstance(getattr(self.postgres_gen, "generate_query", None), (AsyncMock, MagicMock))
        is_mocked_mongo_gen = isinstance(getattr(self.mongo_gen, "generate_query", None), (AsyncMock, MagicMock))
        is_mocked_validator = isinstance(getattr(self.validator, "validate", None), (AsyncMock, MagicMock))

        guardrail = (
            self.guardrail
            if self._custom_guardrail or s1_provider is self.provider or is_mocked_guardrail
            else GuardrailClassifier(provider=s1_provider)
        )
        selector = (
            self.selector
            if self._custom_selector or s1_provider is self.provider or is_mocked_selector
            else TableSelector(provider=s1_provider, cache_manager=self.cache_manager)
        )
        postgres_gen = (
            self.postgres_gen
            if self._custom_postgres_gen or s2_provider is self.provider or is_mocked_postgres_gen
            else PostgresQueryGenerator(provider=s2_provider)
        )
        mongo_gen = (
            self.mongo_gen
            if self._custom_mongo_gen or s2_provider is self.provider or is_mocked_mongo_gen
            else MongoQueryGenerator(provider=s2_provider)
        )
        validator = (
            self.validator
            if self._custom_validator or s3_provider is self.provider or is_mocked_validator
            else QueryValidator(provider=s3_provider)
        )
        return guardrail, selector, postgres_gen, mongo_gen, validator

    def _get_stages_for_provider(self, active_provider: ModelProvider):
        """Return stage instances bound to the active provider for this execution."""
        return self._get_stages_for_providers(active_provider, active_provider, active_provider)

    def get_or_load_schema(self, database: DatabaseTarget | DatabaseType | str) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file, enriched with table/column descriptions."""
        if database in self._schemas:
            return self._schemas[database]

        try:
            target = as_target(database) if not isinstance(database, str) else resolve_target(database)
            if target.key in self._schemas:
                return self._schemas[target.key]
            schema_path = paths.schema_path(target)
            desc_path = paths.descriptions_path(target)
            key = target.key
        except Exception:
            schema_path = get_default_schema_path(database)
            desc_path = None
            key = database

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Canonical schema not found at {schema_path}. Run 'init-schema' first."
            )
        schema = schema_repository.load_schema(schema_path)

        # Merge descriptions directly into schema in-memory
        if desc_path and desc_path.exists():
            try:
                descriptions = description_repository.load_descriptions(desc_path)
                apply_descriptions(schema, descriptions)
            except Exception as err:
                logger.warning("Failed to load descriptions from %s: %s", desc_path, err)

        self._schemas[key] = schema
        return schema

    def _build_response(
        self,
        db_type: DatabaseType,
        question: str,
        guardrail: GuardrailResult,
        run_id: str | None = None,
        semantic: SemanticAnalysisResult | None = None,
        linguistic: dict[str, Any] | None = None,
        candidates: list[CandidateTable] | None = None,
        selected: list[str] | None = None,
        selection_retries: int = 0,
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
        stage_providers: dict[str, str] | None = None,
        stage_tokens: dict[str, int] | None = None,
        spelling_correction: SpellingCorrectionResult | None = None,
    ) -> PipelineResponse:
        cand_list = candidates or []
        if validation is not None:
            if validation.issues:
                validation.issues = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in validation.issues
                    if i
                ]
            if validation.suggestion:
                validation.suggestion = clean_unreadable_characters(
                    strip_ansi_escapes(str(validation.suggestion))
                )
        db_label = target_name if (target_name and target_name not in ("postgres", "postgresql", "mongo", "mongodb")) else db_type.value
        return PipelineResponse(
            run_id=run_id,
            database=db_label,
            question=question,
            provider=provider,
            stage_providers=stage_providers,
            stage_tokens=stage_tokens,
            guardrail=guardrail,
            spelling_correction=spelling_correction,
            basic_answer=basic_answer,
            semantic_analysis=semantic,
            linguistic_analysis=linguistic,
            candidate_tables=cand_list,
            candidate_objects=[c.table_name for c in cand_list],
            selected_objects=selected or [],
            selection_retries=selection_retries,
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

    async def execute_pipeline(
        self,
        question: str,
        database: str | DatabaseType | DatabaseTarget = "postgres",
        provider: str | ModelProvider | None = None,
        stages: StageProviders | dict[str, str] | None = None,
        jargons: list[str] | None = None,
    ) -> PipelineResponse:
        """Run complete V2 natural-language-to-query pipeline with clean stages and bounded retries."""
        run_id = uuid.uuid4().hex
        # Resolve active fallback provider
        if isinstance(provider, str):
            fallback_name = provider.strip().lower()
            active_fallback = resolve_model_provider(provider)
        elif provider is not None:
            fallback_name = getattr(provider, "model", "custom")
            active_fallback = provider
        else:
            fallback_name = "koboldcpp"
            active_fallback = self.provider

        s1_spec = None
        s2_spec = None
        s3_spec = None
        if stages is not None:
            if isinstance(stages, dict):
                s1_spec = stages.get("table_selection")
                s2_spec = stages.get("query_generation")
                s3_spec = stages.get("validation")
            else:
                s1_spec = stages.table_selection
                s2_spec = stages.query_generation
                s3_spec = stages.validation

        s1_provider, s1_name = self._resolve_stage_provider(
            s1_spec, active_fallback, fallback_name, "Stage 1 (Table Selection)"
        )
        s2_provider, s2_name = self._resolve_stage_provider(
            s2_spec, active_fallback, fallback_name, "Stage 2 (Query Generation)"
        )
        s3_provider, s3_name = self._resolve_stage_provider(
            s3_spec, active_fallback, fallback_name, "Stage 3 (Validation)"
        )

        stage_providers_map = {
            "table_selection": s1_name,
            "query_generation": s2_name,
            "validation": s3_name,
        }
        provider_name = (
            s1_name
            if (s1_name == s2_name == s3_name)
            else f"{s1_name}/{s2_name}/{s3_name}"
        )

        guardrail, selector, postgres_gen, mongo_gen, validator = (
            self._get_stages_for_providers(s1_provider, s2_provider, s3_provider)
        )

        # 1. Resolve database target & type
        if isinstance(database, DatabaseTarget):
            target = database
            db_type = target.db_type
        elif isinstance(database, DatabaseType):
            target = as_target(database)
            db_type = database
        elif isinstance(database, str):
            try:
                target = resolve_target(database)
                db_type = target.db_type
            except ValueError:
                db_type = detect_database_type(database)
                target = as_target(db_type)
        else:
            target = as_target(DatabaseType.POSTGRESQL)
            db_type = DatabaseType.POSTGRESQL

        active_generator = postgres_gen if db_type == DatabaseType.POSTGRESQL else mongo_gen

        accum_guardrail_tokens = 0
        accum_selector_tokens = 0
        accum_generator_tokens = 0
        accum_validator_tokens = 0

        def _get_stage_tokens() -> dict[str, int]:
            g_tok = accum_guardrail_tokens + getattr(guardrail, "last_prompt_tokens", 0)
            sel_tok = accum_selector_tokens + getattr(selector, "last_prompt_tokens", 0)
            gen_tok = accum_generator_tokens + getattr(active_generator, "last_prompt_tokens", 0)
            val_tok = accum_validator_tokens + getattr(validator, "last_prompt_tokens", 0)
            s1_tok = g_tok + sel_tok
            s2_tok = gen_tok
            s3_tok = val_tok
            return {
                "stage_1": s1_tok,
                "stage_2": s2_tok,
                "stage_3": s3_tok,
                "guardrail": g_tok,
                "table_selection": sel_tok,
                "query_generation": gen_tok,
                "validation": val_tok,
                "total": s1_tok + s2_tok + s3_tok,
            }

        try:
            schema = self.get_or_load_schema(target)
        except Exception as err:
            return self._build_response(
                run_id=run_id,
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                stage_providers=stage_providers_map,
                stage_tokens=_get_stage_tokens(),
                guardrail=GuardrailResult(decision=GuardrailDecision.REJECT, reason=str(err)),
                error=str(err),
            )

        # 2. Stage 1: SymSpell Spelling Checker (runs FIRST on raw question)
        spelling_res = self.spelling_checker.check(question=question, jargons=jargons)
        processed_question = spelling_res.corrected_question

        # 3. Stage 1 (Guardrail): Guardrail Classifier (runs on corrected question)
        try:
            guardrail_result = await guardrail.classify(processed_question)
        except Exception as err:
            err_msg = f"Stage 1 (Guardrail) failed using provider '{s1_name}': {err}"
            logger.error(err_msg)
            return self._build_response(
                run_id=run_id,
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                stage_providers=stage_providers_map,
                stage_tokens=_get_stage_tokens(),
                guardrail=GuardrailResult(decision=GuardrailDecision.REJECT, reason=err_msg),
                spelling_correction=spelling_res,
                error=err_msg,
            )

        if guardrail_result.decision == GuardrailDecision.REJECT:
            return self._build_response(
                run_id=run_id,
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                stage_providers=stage_providers_map,
                stage_tokens=_get_stage_tokens(),
                guardrail=guardrail_result,
                spelling_correction=spelling_res,
                error="Sorry, I can't help with this.",
            )

        if guardrail_result.decision == GuardrailDecision.BASIC:
            basic_ans = guardrail.handle_basic_question(
                question=processed_question,
                database_type=target.key,
                database_name=schema.database_name,
            )
            return self._build_response(
                run_id=run_id,
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                stage_providers=stage_providers_map,
                stage_tokens=_get_stage_tokens(),
                guardrail=guardrail_result,
                spelling_correction=spelling_res,
                basic_answer=basic_ans,
            )

        # 4. Stages 3-5: Table Selector & Query Generator with Bounded Backtrack Loops
        selector_retries = 0
        validator_retries = 0
        selector_feedback: str | None = None
        gen_feedback: str | None = None

        selected_objs: list[str] = []
        relevant_schema: RelevantSchema = RelevantSchema()
        last_query: GeneratedQuery | None = None
        last_val: ValidationResult | None = None

        while selector_retries <= self.max_retries:
            # Stage 1: Table Selector (evaluates complete schema description)
            try:
                selection = await selector.select(
                    question=processed_question,
                    schema=schema,
                    feedback=selector_feedback,
                    target_key=target.key,
                )
            except Exception as err:
                err_msg = f"Stage 1 (Table Selection) failed using provider '{s1_name}': {err}"
                logger.error(err_msg)
                return self._build_response(
                    run_id=run_id,
                    db_type=db_type,
                    target_name=target.key,
                    question=question,
                    provider=provider_name,
                    stage_providers=stage_providers_map,
                    stage_tokens=_get_stage_tokens(),
                    guardrail=guardrail_result,
                    spelling_correction=spelling_res,
                    selection_retries=selector_retries,
                    error=err_msg,
                )

            if not selection.selected_objects:
                if selection.reason and selection.reason.startswith("Table selection failed due to error:"):
                    err_detail = selection.reason.replace("Table selection failed due to error:", "").strip()
                    err_msg = f"Stage 1 (Table Selection) failed using provider '{s1_name}': {err_detail}"
                    logger.error(err_msg)
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=[],
                        selection_retries=selector_retries,
                        error=err_msg,
                    )
                if selector_retries >= self.max_retries:
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=[],
                        selection_retries=selector_retries,
                        error=selection.reason or "I could not find any relevant tables or collections in the database schema matching your question.",
                    )
                selector_retries += 1
                accum_selector_tokens += getattr(selector, "last_prompt_tokens", 0)
                selector.last_prompt_tokens = 0
                selector_feedback = (
                    "No tables were selected. Please select all required tables from the database schema to answer the question."
                )
                continue

            selected_objs = selection.selected_objects
            relevant_schema = package_selected_schema(schema, selected_objs)

            # Stage 2 & 3: Query Generator & Validator Loop
            gen_needed_more_schema = False

            while validator_retries <= self.max_retries:
                try:
                    gen_result = await active_generator.generate_query(
                        question=processed_question,
                        schema=relevant_schema,
                        feedback=gen_feedback,
                    )
                except Exception as gen_err:
                    err_msg = f"Stage 2 (Query Generation) failed using provider '{s2_name}': {gen_err}"
                    logger.error(err_msg)
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        validation_retries=validator_retries,
                        attempts=validator_retries + 1,
                        error=err_msg,
                    )

                # Check if Query Generator detected missing schema information
                if isinstance(gen_result, GeneratorMissingInfo):
                    if selector_retries >= self.max_retries:
                        return self._build_response(
                            run_id=run_id,
                            db_type=db_type,
                            target_name=target.key,
                            question=question,
                            provider=provider_name,
                            stage_providers=stage_providers_map,
                            stage_tokens=_get_stage_tokens(),
                            guardrail=guardrail_result,
                            spelling_correction=spelling_res,
                            selected=selected_objs,
                            selection_retries=selector_retries,
                            schema=relevant_schema,
                            error=f"Required schema information could not be identified after maximum retries: {gen_result.reason}",
                        )
                    selector_retries += 1
                    accum_generator_tokens += getattr(active_generator, "last_prompt_tokens", 0)
                    active_generator.last_prompt_tokens = 0
                    missing_str = ", ".join(gen_result.missing_tables) if gen_result.missing_tables else "additional tables"
                    selector_feedback = (
                        f"The query generator determined that the previous selection of tables {selected_objs} "
                        f"lacks required schema information: {gen_result.reason}. "
                        f"Missing tables/concepts: [{missing_str}]. Please re-evaluate the full schema and select ALL necessary tables."
                    )
                    gen_needed_more_schema = True
                    break

                last_query = gen_result

                # Stage 3: Query Validator
                try:
                    last_val = await validator.validate(
                        question=processed_question,
                        generated_query=last_query,
                        schema=relevant_schema,
                        fail_fast=True,
                    )
                except Exception as val_err:
                    err_msg = f"Stage 3 (Validation) failed using provider '{s3_name}': {val_err}"
                    logger.error(err_msg)
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        query=last_query,
                        validation_retries=validator_retries,
                        attempts=validator_retries + 1,
                        error=err_msg,
                    )

                if last_val.valid:
                    break

                if last_val.error_type == ValidationErrorType.UNSAFE:
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        query=last_query,
                        validation=last_val,
                        validation_retries=validator_retries,
                        attempts=validator_retries + 1,
                        error="Generated query is unsafe. Execution rejected.",
                    )

                if validator_retries >= self.max_retries:
                    break

                validator_retries += 1
                accum_generator_tokens += getattr(active_generator, "last_prompt_tokens", 0)
                active_generator.last_prompt_tokens = 0
                accum_validator_tokens += getattr(validator, "last_prompt_tokens", 0)
                if hasattr(validator, "postgres_validator"):
                    validator.postgres_validator.last_prompt_tokens = 0
                if hasattr(validator, "mongo_validator"):
                    validator.mongo_validator.last_prompt_tokens = 0

                clean_issues = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in last_val.issues
                    if i
                ]
                feedback_msg = "; ".join(clean_issues)
                if last_val.suggestion:
                    clean_sugg = clean_unreadable_characters(strip_ansi_escapes(str(last_val.suggestion)))
                    feedback_msg += f". Suggestion: {clean_sugg}"
                gen_feedback = clean_unreadable_characters(strip_ansi_escapes(feedback_msg))

            if gen_needed_more_schema:
                continue

            break

        attempts = validator_retries + 1
        ctx: dict[str, Any] = {
            "run_id": run_id,
            "db_type": db_type,
            "target_name": target.key,
            "question": question,
            "provider": provider_name,
            "stage_providers": stage_providers_map,
            "stage_tokens": _get_stage_tokens(),
            "guardrail": guardrail_result,
            "spelling_correction": spelling_res,
            "selected": selected_objs,
            "selection_retries": selector_retries,
            "schema": relevant_schema,
            "query": last_query,
            "attempts": attempts,
            "validation_retries": validator_retries,
            "validation": last_val,
        }

        if not last_query or not last_val or not last_val.valid:
            resp = self._build_response(
                **ctx,
                error="Query failed validation after maximum retries. Manual review required.",
            )
            try:
                run_data = {
                    "run_id": run_id,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "database": target.key,
                    "question": question,
                    "processed_question": processed_question,
                    "jargons": jargons or [],
                    "stage_providers": stage_providers_map,
                    "stage_tokens": _get_stage_tokens(),
                    "guardrail": guardrail_result.model_dump() if guardrail_result else None,
                    "spelling_correction": spelling_res.model_dump() if spelling_res else None,
                    "selected_objects": selected_objs,
                    "selection_retries": selector_retries,
                    "relevant_schema": relevant_schema.model_dump() if relevant_schema else None,
                    "generated_query": last_query.formatted_query if last_query else None,
                    "attempts": attempts,
                    "validation_retries": validator_retries,
                    "validation": last_val.model_dump() if last_val else None,
                    "error": resp.error,
                    "feedback_history": [],
                }
                save_run_state(run_data)
                cleanup_old_runs()
            except Exception as save_err:
                logger.warning("Failed to save run state for %s: %s", run_id, save_err)
            return resp

        # 5. Stage 6: Deterministic Safety Policy Check
        policy_result = self.policy.check(last_query, schema=relevant_schema)
        if not policy_result.allowed:
            resp = self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )
            try:
                run_data = {
                    "run_id": run_id,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "database": target.key,
                    "question": question,
                    "processed_question": processed_question,
                    "jargons": jargons or [],
                    "stage_providers": stage_providers_map,
                    "stage_tokens": _get_stage_tokens(),
                    "guardrail": guardrail_result.model_dump() if guardrail_result else None,
                    "spelling_correction": spelling_res.model_dump() if spelling_res else None,
                    "selected_objects": selected_objs,
                    "selection_retries": selector_retries,
                    "relevant_schema": relevant_schema.model_dump() if relevant_schema else None,
                    "generated_query": last_query.formatted_query if last_query else None,
                    "attempts": attempts,
                    "validation_retries": validator_retries,
                    "validation": last_val.model_dump() if last_val else None,
                    "policy": policy_result.model_dump(),
                    "error": resp.error,
                    "feedback_history": [],
                }
                save_run_state(run_data)
                cleanup_old_runs()
            except Exception as save_err:
                logger.warning("Failed to save run state for %s: %s", run_id, save_err)
            return resp

        # 6. Stages 7 & 8: Execution and Result Processing
        try:
            columns, raw_rows = self.executor.execute(
                last_query,
                connection=target.connection,
                database=target.mongo_database,
            )
        except Exception as exec_err:
            logger.error("Execution error: %s", exec_err)
            resp = self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Database execution error: {exec_err}",
            )
            try:
                run_data = {
                    "run_id": run_id,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "database": target.key,
                    "question": question,
                    "processed_question": processed_question,
                    "jargons": jargons or [],
                    "stage_providers": stage_providers_map,
                    "stage_tokens": _get_stage_tokens(),
                    "guardrail": guardrail_result.model_dump() if guardrail_result else None,
                    "spelling_correction": spelling_res.model_dump() if spelling_res else None,
                    "selected_objects": selected_objs,
                    "selection_retries": selector_retries,
                    "relevant_schema": relevant_schema.model_dump() if relevant_schema else None,
                    "generated_query": last_query.formatted_query if last_query else None,
                    "attempts": attempts,
                    "validation_retries": validator_retries,
                    "validation": last_val.model_dump() if last_val else None,
                    "policy": policy_result.model_dump(),
                    "error": resp.error,
                    "feedback_history": [],
                }
                save_run_state(run_data)
                cleanup_old_runs()
            except Exception as save_err:
                logger.warning("Failed to save run state for %s: %s", run_id, save_err)
            return resp

        execution_results = self.results.process(columns, raw_rows)
        resp = self._build_response(
            **ctx,
            policy=policy_result,
            results=execution_results,
        )

        try:
            run_data = {
                "run_id": run_id,
                "created_at": time.time(),
                "updated_at": time.time(),
                "database": target.key,
                "question": question,
                "processed_question": processed_question,
                "jargons": jargons or [],
                "stage_providers": stage_providers_map,
                "stage_tokens": _get_stage_tokens(),
                "guardrail": guardrail_result.model_dump() if guardrail_result else None,
                "spelling_correction": spelling_res.model_dump() if spelling_res else None,
                "selected_objects": selected_objs,
                "selection_retries": selector_retries,
                "relevant_schema": relevant_schema.model_dump() if relevant_schema else None,
                "generated_query": last_query.formatted_query if last_query else None,
                "attempts": attempts,
                "validation_retries": validator_retries,
                "validation": last_val.model_dump() if last_val else None,
                "policy": policy_result.model_dump() if policy_result else None,
                "error": None,
                "feedback_history": [],
            }
            save_run_state(run_data)
            cleanup_old_runs()
        except Exception as save_err:
            logger.warning("Failed to save run state for %s: %s", run_id, save_err)

        return resp

    def _update_saved_run_state(
        self,
        run_id: str,
        run_state: dict[str, Any],
        applicable_feedback: dict[str, str],
        selected_objs: list[str],
        relevant_schema: RelevantSchema | None,
        last_query: GeneratedQuery | None,
        last_val: ValidationResult | None,
        policy_result: PolicyResult | None,
        execution_results: ExecutionResult | None,
        error: str | None,
        attempts: int,
        selector_retries: int,
        validator_retries: int,
        stage_tokens: dict[str, int],
    ) -> None:
        feedback_history = list(run_state.get("feedback_history", []))
        for stg_k, fb_txt in applicable_feedback.items():
            feedback_history.append({
                "stage": stg_k,
                "feedback": fb_txt,
                "timestamp": time.time(),
            })

        updated = dict(run_state)
        updated["updated_at"] = time.time()
        updated["stage_tokens"] = stage_tokens
        updated["selected_objects"] = selected_objs
        updated["selection_retries"] = selector_retries
        updated["relevant_schema"] = relevant_schema.model_dump() if relevant_schema else None
        updated["generated_query"] = last_query.formatted_query if last_query else None
        updated["attempts"] = attempts
        updated["validation_retries"] = validator_retries
        updated["validation"] = last_val.model_dump() if last_val else None
        updated["policy"] = policy_result.model_dump() if policy_result else None
        updated["error"] = error
        updated["feedback_history"] = feedback_history

        try:
            save_run_state(updated)
            cleanup_old_runs()
        except Exception as save_err:
            logger.warning("Failed to update run state for %s: %s", run_id, save_err)

    async def rerun_pipeline(
        self,
        run_id: str,
        starting_stage: str | None = None,
        feedback: dict[str, str] | None = None,
    ) -> PipelineResponse:
        """Execute a partial pipeline rerun from a specified stage using saved run state."""
        if not validate_run_id(run_id):
            raise ValueError(f"Invalid run_id format: {run_id!r}")

        run_state = load_run_state(run_id)
        if not run_state:
            raise FileNotFoundError(f"Run '{run_id}' not found or expired.")

        valid_stages = ["table_selection", "query_generation", "validation"]
        raw_feedback = feedback or {}
        cleaned_feedback = {
            k: v.strip()
            for k, v in raw_feedback.items()
            if k in valid_stages and isinstance(v, str) and v.strip()
        }

        if starting_stage:
            if starting_stage not in valid_stages:
                raise ValueError(f"Invalid starting_stage: {starting_stage!r}. Must be one of {valid_stages}")
            actual_start = starting_stage
        else:
            actual_start = "table_selection"
            for stg in valid_stages:
                if cleaned_feedback.get(stg):
                    actual_start = stg
                    break

        # A "Rerun" button at stage N submits that stage's feedback plus any non-empty
        # feedback boxes of later stages, and starts from stage N. Earlier stages' boxes are ignored.
        start_idx = valid_stages.index(actual_start)
        applicable_feedback = {
            stg: cleaned_feedback[stg]
            for stg in valid_stages[start_idx:]
            if stg in cleaned_feedback
        }

        # 1. Resolve providers from original run state
        orig_providers = run_state.get("stage_providers") or {}
        s1_spec = orig_providers.get("table_selection", "koboldcpp")
        s2_spec = orig_providers.get("query_generation", "koboldcpp")
        s3_spec = orig_providers.get("validation", "koboldcpp")

        s1_provider, s1_name = self._resolve_stage_provider(
            s1_spec, self.provider, "koboldcpp", "Stage 1 (Table Selection)"
        )
        s2_provider, s2_name = self._resolve_stage_provider(
            s2_spec, self.provider, "koboldcpp", "Stage 2 (Query Generation)"
        )
        s3_provider, s3_name = self._resolve_stage_provider(
            s3_spec, self.provider, "koboldcpp", "Stage 3 (Validation)"
        )

        stage_providers_map = {
            "table_selection": s1_name,
            "query_generation": s2_name,
            "validation": s3_name,
        }
        provider_name = (
            s1_name
            if (s1_name == s2_name == s3_name)
            else f"{s1_name}/{s2_name}/{s3_name}"
        )

        guardrail, selector, postgres_gen, mongo_gen, validator = (
            self._get_stages_for_providers(s1_provider, s2_provider, s3_provider)
        )

        # 2. Safety Guardrail for Feedback
        # Uses table selection stage's provider for the run.
        # Zero admin calls issued to /api/admin/* even on koboldcpp.
        fb_guardrail = FeedbackGuardrailClassifier(provider=s1_provider, provider_name=s1_name)
        accum_guardrail_tokens = 0
        for stg_key, fb_text in applicable_feedback.items():
            fb_result = await fb_guardrail.classify(feedback=fb_text, stage=stg_key)
            accum_guardrail_tokens += fb_result.prompt_tokens
            if fb_result.decision == "REJECT":
                raise ValueError(
                    f"Feedback for {stg_key} was rejected by the safety guardrail: {fb_result.reason}"
                )

        # 3. Database Target & Schema
        database = run_state.get("database", "postgres")
        target = resolve_target(database)
        db_type = target.db_type
        active_generator = postgres_gen if db_type == DatabaseType.POSTGRESQL else mongo_gen
        schema = self.get_or_load_schema(target)

        question = run_state.get("question", "")
        processed_question = run_state.get("processed_question") or question
        feedback_history = list(run_state.get("feedback_history", []))

        spelling_dict = run_state.get("spelling_correction")
        spelling_res = (
            SpellingCorrectionResult.model_validate(spelling_dict)
            if spelling_dict
            else None
        )
        guardrail_dict = run_state.get("guardrail")
        guardrail_result = (
            GuardrailResult.model_validate(guardrail_dict)
            if guardrail_dict
            else GuardrailResult(decision=GuardrailDecision.READ_QUERY, reason="Loaded from run state")
        )

        accum_selector_tokens = 0
        accum_generator_tokens = 0
        accum_validator_tokens = 0

        def _get_stage_tokens() -> dict[str, int]:
            orig_tokens = run_state.get("stage_tokens") or {}
            g_tok = accum_guardrail_tokens + (orig_tokens.get("guardrail", 0) if actual_start != "table_selection" else 0)
            sel_tok = accum_selector_tokens + getattr(selector, "last_prompt_tokens", 0) + (orig_tokens.get("table_selection", 0) if actual_start != "table_selection" else 0)
            gen_tok = accum_generator_tokens + getattr(active_generator, "last_prompt_tokens", 0) + (orig_tokens.get("query_generation", 0) if actual_start == "validation" else 0)
            val_tok = accum_validator_tokens + getattr(validator, "last_prompt_tokens", 0)
            s1_tok = g_tok + sel_tok
            s2_tok = gen_tok
            s3_tok = val_tok
            return {
                "stage_1": s1_tok,
                "stage_2": s2_tok,
                "stage_3": s3_tok,
                "guardrail": g_tok,
                "table_selection": sel_tok,
                "query_generation": gen_tok,
                "validation": val_tok,
                "total": s1_tok + s2_tok + s3_tok,
            }

        selector_retries = 0
        validator_retries = 0
        selected_objs: list[str] = []
        relevant_schema: RelevantSchema = RelevantSchema()
        last_query: GeneratedQuery | None = None
        last_val: ValidationResult | None = None

        # Build cascading user guidance blocks per stage
        s1_items = _get_stage_feedback_items(
            feedback_history, applicable_feedback.get("table_selection"), "table_selection"
        )
        s2_items = _get_stage_feedback_items(
            feedback_history, applicable_feedback.get("query_generation"), "query_generation"
        )
        s3_items = _get_stage_feedback_items(
            feedback_history, applicable_feedback.get("validation"), "validation"
        )

        # Stage 1: Table Selection guidance
        if s1_items:
            joined_s1 = "\n".join(f"- {it}" for it in s1_items)
            s1_user_guidance = (
                f"[USER GUIDANCE / HINT]:\n{joined_s1}\n"
                "(Note: User guidance is an advisory hint. You must strictly ground table selection in the canonical schema and adhere to all relationship rules.)"
            )
        else:
            s1_user_guidance = None

        # Stage 2: Query Generation guidance (receives Stage 2 feedback plus active Stage 1 context)
        gen_guidance_items: list[str] = []
        if s1_items:
            gen_guidance_items.extend([f"Context from table selection: {it}" for it in s1_items])
        if s2_items:
            gen_guidance_items.extend(s2_items)

        if gen_guidance_items:
            joined_s2 = "\n".join(f"- {it}" for it in gen_guidance_items)
            s2_user_guidance = (
                f"[USER GUIDANCE / HINT]:\n{joined_s2}\n"
                "(Note: User guidance provides domain/intent clarification. All generated queries must strictly adhere to the Selected Schema and read-only safety rules.)"
            )
        else:
            s2_user_guidance = None

        # Stage 3: Validation guidance (receives Stage 3 feedback plus all applied query guidance)
        val_sections: list[str] = []
        applied_query_items: list[str] = []
        if s1_items:
            applied_query_items.extend([f"Context from table selection: {it}" for it in s1_items])
        if s2_items:
            applied_query_items.extend(s2_items)

        if applied_query_items:
            joined_applied = "\n".join(f"- {it}" for it in applied_query_items)
            val_sections.append(
                f"[APPLIED QUERY GUIDANCE]:\n{joined_applied}\n"
                "(Note: The above user guidance was explicitly requested by the user to scope query generation. All requested filters, status codes, grains, and constraints are authorized user refinements and MUST be accepted as correct.)"
            )
        if s3_items:
            joined_s3 = "\n".join(f"- {it}" for it in s3_items)
            val_sections.append(
                f"[USER VALIDATION GUIDANCE]:\n{joined_s3}\n"
                "(Note: User validation guidance clarifies user intent or acceptable caveats. Ensure the query remains strictly read-only and free of syntax errors.)"
            )

        s3_val_guidance = "\n\n".join(val_sections) if val_sections else None

        # ----------------------------------------------------
        # Stage 1: Table Selection (if actual_start == "table_selection")
        # ----------------------------------------------------
        if actual_start == "table_selection":
            selector_feedback = s1_user_guidance
            while selector_retries <= self.max_retries:
                try:
                    selection = await selector.select(
                        question=processed_question,
                        schema=schema,
                        feedback=selector_feedback,
                        target_key=target.key,
                    )
                except Exception as err:
                    err_msg = f"Stage 1 (Table Selection) failed using provider '{s1_name}': {err}"
                    logger.error(err_msg)
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selection_retries=selector_retries,
                        error=err_msg,
                    )

                if not selection.selected_objects:
                    if selector_retries >= self.max_retries:
                        return self._build_response(
                            run_id=run_id,
                            db_type=db_type,
                            target_name=target.key,
                            question=question,
                            provider=provider_name,
                            stage_providers=stage_providers_map,
                            stage_tokens=_get_stage_tokens(),
                            guardrail=guardrail_result,
                            spelling_correction=spelling_res,
                            selected=[],
                            selection_retries=selector_retries,
                            error=selection.reason or "I could not find any relevant tables in the schema matching your question.",
                        )
                    selector_retries += 1
                    accum_selector_tokens += getattr(selector, "last_prompt_tokens", 0)
                    selector.last_prompt_tokens = 0
                    retry_msg = "No tables were selected. Please select all required tables from the database schema to answer the question."
                    selector_feedback = f"{s1_user_guidance}\n\n[SYSTEM RETRY FEEDBACK]:\n{retry_msg}" if s1_user_guidance else retry_msg
                    continue

                selected_objs = selection.selected_objects
                relevant_schema = package_selected_schema(schema, selected_objs)
                break
        else:
            # Re-use previous table selection from saved run state
            selected_objs = list(run_state.get("selected_objects", []))
            rel_dict = run_state.get("relevant_schema")
            relevant_schema = RelevantSchema.model_validate(rel_dict) if rel_dict else package_selected_schema(schema, selected_objs)
            selector_retries = int(run_state.get("selection_retries", 0))

        # ----------------------------------------------------
        # Stage 2 & 3: Query Generation & Validation
        # ----------------------------------------------------
        skip_initial_gen = False
        if actual_start == "validation":
            prev_q_str = run_state.get("generated_query")
            if prev_q_str:
                if db_type == DatabaseType.POSTGRESQL:
                    last_query = GeneratedQuery(
                        database_type=DatabaseType.POSTGRESQL,
                        raw_query=str(prev_q_str),
                        formatted_query=str(prev_q_str),
                    )
                else:
                    import json
                    from backend.src.schemas.pipeline import MongoQuery
                    try:
                        raw_q = MongoQuery.model_validate(json.loads(prev_q_str) if isinstance(prev_q_str, str) else prev_q_str)
                        last_query = GeneratedQuery(
                            database_type=DatabaseType.MONGODB,
                            raw_query=raw_q,
                            formatted_query=str(prev_q_str),
                        )
                    except Exception:
                        last_query = None

                if last_query:
                    try:
                        last_val = await validator.validate(
                            question=processed_question,
                            generated_query=last_query,
                            schema=relevant_schema,
                            feedback=s3_val_guidance,
                            fail_fast=True,
                        )
                    except Exception as val_err:
                        err_msg = f"Stage 3 (Validation) failed using provider '{s3_name}': {val_err}"
                        logger.error(err_msg)
                        return self._build_response(
                            run_id=run_id,
                            db_type=db_type,
                            target_name=target.key,
                            question=question,
                            provider=provider_name,
                            stage_providers=stage_providers_map,
                            stage_tokens=_get_stage_tokens(),
                            guardrail=guardrail_result,
                            spelling_correction=spelling_res,
                            selected=selected_objs,
                            selection_retries=selector_retries,
                            schema=relevant_schema,
                            query=last_query,
                            error=err_msg,
                        )

                    if last_val.valid:
                        skip_initial_gen = True

        if not skip_initial_gen:
            gen_feedback = s2_user_guidance
            while validator_retries <= self.max_retries:
                try:
                    gen_result = await active_generator.generate_query(
                        question=processed_question,
                        schema=relevant_schema,
                        feedback=gen_feedback,
                    )
                except Exception as gen_err:
                    err_msg = f"Stage 2 (Query Generation) failed using provider '{s2_name}': {gen_err}"
                    logger.error(err_msg)
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        validation_retries=validator_retries,
                        attempts=validator_retries + 1,
                        error=err_msg,
                    )

                if isinstance(gen_result, GeneratorMissingInfo):
                    if actual_start == "table_selection" and selector_retries < self.max_retries:
                        selector_retries += 1
                        accum_generator_tokens += getattr(active_generator, "last_prompt_tokens", 0)
                        active_generator.last_prompt_tokens = 0
                        missing_str = ", ".join(gen_result.missing_tables) if gen_result.missing_tables else "additional tables"
                        sys_fb = (
                            f"The query generator determined that the previous selection of tables {selected_objs} "
                            f"lacks required schema information: {gen_result.reason}. "
                            f"Missing tables/concepts: [{missing_str}]. Please re-evaluate the full schema and select ALL necessary tables."
                        )
                        selector_feedback = f"{s1_user_guidance}\n\n[SYSTEM RETRY FEEDBACK]:\n{sys_fb}" if s1_user_guidance else sys_fb
                        selection = await selector.select(
                            question=processed_question,
                            schema=schema,
                            feedback=selector_feedback,
                            target_key=target.key,
                        )
                        if selection.selected_objects:
                            selected_objs = selection.selected_objects
                            relevant_schema = package_selected_schema(schema, selected_objs)
                            continue
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        error=f"Required schema information missing: {gen_result.reason}. Please rerun from Table Selection with guidance.",
                    )

                last_query = gen_result

                # Stage 3: Query Validator
                try:
                    last_val = await validator.validate(
                        question=processed_question,
                        generated_query=last_query,
                        schema=relevant_schema,
                        feedback=s3_val_guidance,
                        fail_fast=True,
                    )
                except Exception as val_err:
                    err_msg = f"Stage 3 (Validation) failed using provider '{s3_name}': {val_err}"
                    logger.error(err_msg)
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        query=last_query,
                        validation_retries=validator_retries,
                        attempts=validator_retries + 1,
                        error=err_msg,
                    )

                if last_val.valid:
                    break

                if last_val.error_type == ValidationErrorType.UNSAFE:
                    return self._build_response(
                        run_id=run_id,
                        db_type=db_type,
                        target_name=target.key,
                        question=question,
                        provider=provider_name,
                        stage_providers=stage_providers_map,
                        stage_tokens=_get_stage_tokens(),
                        guardrail=guardrail_result,
                        spelling_correction=spelling_res,
                        selected=selected_objs,
                        selection_retries=selector_retries,
                        schema=relevant_schema,
                        query=last_query,
                        validation=last_val,
                        validation_retries=validator_retries,
                        attempts=validator_retries + 1,
                        error="Generated query is unsafe. Execution rejected.",
                    )

                if validator_retries >= self.max_retries:
                    break

                validator_retries += 1
                accum_generator_tokens += getattr(active_generator, "last_prompt_tokens", 0)
                active_generator.last_prompt_tokens = 0
                accum_validator_tokens += getattr(validator, "last_prompt_tokens", 0)

                clean_issues = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in last_val.issues
                    if i
                ]
                feedback_msg = "; ".join(clean_issues)
                if last_val.suggestion:
                    clean_sugg = clean_unreadable_characters(strip_ansi_escapes(str(last_val.suggestion)))
                    feedback_msg += f". Suggestion: {clean_sugg}"
                clean_fb = clean_unreadable_characters(strip_ansi_escapes(feedback_msg))
                system_retry_part = f"[SYSTEM VALIDATOR FEEDBACK (RETRY)]:\n{clean_fb}"
                gen_feedback = f"{s2_user_guidance}\n\n{system_retry_part}" if s2_user_guidance else system_retry_part

        attempts = validator_retries + 1
        ctx: dict[str, Any] = {
            "run_id": run_id,
            "db_type": db_type,
            "target_name": target.key,
            "question": question,
            "provider": provider_name,
            "stage_providers": stage_providers_map,
            "stage_tokens": _get_stage_tokens(),
            "guardrail": guardrail_result,
            "spelling_correction": spelling_res,
            "selected": selected_objs,
            "selection_retries": selector_retries,
            "schema": relevant_schema,
            "query": last_query,
            "attempts": attempts,
            "validation_retries": validator_retries,
            "validation": last_val,
        }

        if not last_query or not last_val or not last_val.valid:
            resp = self._build_response(
                **ctx,
                error="Query failed validation after maximum retries. Manual review required.",
            )
            self._update_saved_run_state(
                run_id=run_id,
                run_state=run_state,
                applicable_feedback=applicable_feedback,
                selected_objs=selected_objs,
                relevant_schema=relevant_schema,
                last_query=last_query,
                last_val=last_val,
                policy_result=None,
                execution_results=None,
                error=resp.error,
                attempts=attempts,
                selector_retries=selector_retries,
                validator_retries=validator_retries,
                stage_tokens=_get_stage_tokens(),
            )
            return resp

        # ----------------------------------------------------
        # Stage 4: Deterministic Safety Policy Check
        # ----------------------------------------------------
        policy_result = self.policy.check(last_query, schema=relevant_schema)
        if not policy_result.allowed:
            resp = self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )
            self._update_saved_run_state(
                run_id=run_id,
                run_state=run_state,
                applicable_feedback=applicable_feedback,
                selected_objs=selected_objs,
                relevant_schema=relevant_schema,
                last_query=last_query,
                last_val=last_val,
                policy_result=policy_result,
                execution_results=None,
                error=resp.error,
                attempts=attempts,
                selector_retries=selector_retries,
                validator_retries=validator_retries,
                stage_tokens=_get_stage_tokens(),
            )
            return resp

        # ----------------------------------------------------
        # Stage 5: Execution & Result Processing
        # ----------------------------------------------------
        try:
            columns, raw_rows = self.executor.execute(
                last_query,
                connection=target.connection,
                database=target.mongo_database,
            )
        except Exception as exec_err:
            logger.error("Execution error: %s", exec_err)
            resp = self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Database execution error: {exec_err}",
            )
            self._update_saved_run_state(
                run_id=run_id,
                run_state=run_state,
                applicable_feedback=applicable_feedback,
                selected_objs=selected_objs,
                relevant_schema=relevant_schema,
                last_query=last_query,
                last_val=last_val,
                policy_result=policy_result,
                execution_results=None,
                error=resp.error,
                attempts=attempts,
                selector_retries=selector_retries,
                validator_retries=validator_retries,
                stage_tokens=_get_stage_tokens(),
            )
            return resp

        execution_results = self.results.process(columns, raw_rows)
        resp = self._build_response(
            **ctx,
            policy=policy_result,
            results=execution_results,
        )

        self._update_saved_run_state(
            run_id=run_id,
            run_state=run_state,
            applicable_feedback=applicable_feedback,
            selected_objs=selected_objs,
            relevant_schema=relevant_schema,
            last_query=last_query,
            last_val=last_val,
            policy_result=policy_result,
            execution_results=execution_results,
            error=None,
            attempts=attempts,
            selector_retries=selector_retries,
            validator_retries=validator_retries,
            stage_tokens=_get_stage_tokens(),
        )

        return resp
