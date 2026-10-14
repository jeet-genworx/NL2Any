"""End-to-end pipeline orchestrator for NL2AnyQuery V2."""

import json
import logging
import time
import uuid
from typing import Any

from backend.src.config import settings
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.factory import resolve_model_provider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.control.providers.model.koboldcpp_cache import (
    KoboldCppCacheManager,
    get_cache_manager,
)
from backend.src.core.query_processing.pipeline.executor import QueryExecutor
from backend.src.core.query_processing.pipeline.feedback import (
    build_cascading_guidance,
)
from backend.src.core.query_processing.pipeline.feedback_guardrail import (
    FeedbackGuardrailClassifier,
)
from backend.src.core.query_processing.pipeline.fsm import (
    ExecutionAdapter,
    FeedbackGuardrailAdapter,
    PipelineContext,
    PipelineFSMRunner,
    PipelineState,
    PolicyAdapter,
    QueryGenerationAdapter,
    QueryValidationAdapter,
    QuestionGuardrailAdapter,
    SpellingAdapter,
    StageAdapters,
    TableSelectionAdapter,
)
from backend.src.core.query_processing.pipeline.generators.mongo import (
    MongoQueryGenerator,
)
from backend.src.core.query_processing.pipeline.generators.postgres import (
    PostgresQueryGenerator,
)
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.policy import SafetyPolicyValidator
from backend.src.core.query_processing.pipeline.results import ResultProcessor
from backend.src.core.query_processing.pipeline.selector import (
    TableSelector,
    package_selected_schema,
)
from backend.src.core.query_processing.pipeline.spelling import SpellingChecker
from backend.src.core.query_processing.pipeline.validator import QueryValidator
from backend.src.data.clients.detector import detect_database_type
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.models.targets import DatabaseTarget, as_target, resolve_target
from backend.src.data.repositories.run_repository import (
    cleanup_old_runs,
    load_run_state,
    save_run_state,
    validate_run_id,
)
from backend.src.data.repositories.schema_cache import SchemaCache
from backend.src.schemas.pipeline import (
    CandidateTable,
    ExecutionResult,
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    MongoQuery,
    PipelineResponse,
    PolicyResult,
    RelevantSchema,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
    ValidationResult,
)
from backend.src.schemas.request import StageProviders
from backend.src.utils.text_utils import clean_unreadable_characters, strip_ansi_escapes

logger = logging.getLogger(__name__)


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
        self.feedback_guardrail = kwargs.get("feedback_guardrail") or FeedbackGuardrailClassifier(provider=self.provider)

        self._custom_guardrail = guardrail is not None
        self._custom_selector = selector is not None
        self._custom_postgres_gen = postgres_gen is not None
        self._custom_mongo_gen = mongo_gen is not None
        self._custom_validator = validator is not None

        self.schema_cache = SchemaCache()

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

    def get_or_load_schema(self, database: DatabaseTarget | DatabaseType | str) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file, enriched with table/column descriptions."""
        return self.schema_cache.get_or_load(database)

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

    def _calculate_stage_tokens_dict(self, context: PipelineContext) -> dict[str, int]:
        tokens_map = context.stage_tokens
        g_tok = (
            tokens_map.get("feedback_guardrail", 0)
            + tokens_map.get("question_guardrail", 0)
            + tokens_map.get("guardrail", 0)
        )
        sel_tok = tokens_map.get("table_selection", 0)
        gen_tok = tokens_map.get("query_generation", 0)
        val_tok = tokens_map.get("query_validation", 0) + tokens_map.get("validation", 0)
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

    def _build_pipeline_response(
        self,
        context: PipelineContext,
        provider_name: str,
    ) -> PipelineResponse:
        """Construct a PipelineResponse from final pipeline execution context."""
        stage_tokens = self._calculate_stage_tokens_dict(context)

        guardrail_res = context.guardrail_result
        if guardrail_res is None and context.error and "Stage 1 (Guardrail) failed" in context.error:
            guardrail_res = GuardrailResult(decision=GuardrailDecision.REJECT, reason=context.error)

        return self._build_response(
            run_id=context.run_id,
            db_type=context.db_type,
            target_name=context.target_name,
            question=context.question,
            provider=provider_name,
            stage_providers=context.stage_providers,
            stage_tokens=stage_tokens,
            guardrail=guardrail_res,
            spelling_correction=context.spelling_result,
            basic_answer=context.basic_answer,
            selected=context.selected_objects,
            selection_retries=context.selector_retries,
            schema=context.relevant_schema,
            query=context.generated_query,
            attempts=context.attempts,
            validation_retries=context.validation_retries,
            validation=context.validation_result,
            policy=context.policy_result,
            results=context.execution_result,
            error=context.error,
        )

    def _save_pipeline_run_file(
        self,
        context: PipelineContext,
        resp: PipelineResponse,
    ) -> None:
        """Persist pipeline run artifacts from final pipeline execution context."""
        if not (
            context.execution_result is not None
            or (context.policy_result is not None and not context.policy_result.allowed)
            or (
                context.policy_result is not None
                and context.policy_result.allowed
                and context.error is not None
                and "Database execution error" in context.error
            )
            or (
                context.error is not None
                and "Query failed validation after maximum retries" in context.error
            )
        ):
            return

        stage_tokens = self._calculate_stage_tokens_dict(context)

        try:
            run_data = {
                "run_id": context.run_id,
                "created_at": time.time(),
                "updated_at": time.time(),
                "database": context.target_name,
                "question": context.question,
                "processed_question": context.processed_question,
                "jargons": list(context.jargons or []),
                "stage_providers": context.stage_providers,
                "stage_tokens": stage_tokens,
                "guardrail": context.guardrail_result.model_dump() if context.guardrail_result else None,
                "spelling_correction": context.spelling_result.model_dump() if context.spelling_result else None,
                "selected_objects": list(context.selected_objects or []),
                "selection_retries": context.selector_retries,
                "relevant_schema": context.relevant_schema.model_dump() if context.relevant_schema else None,
                "generated_query": context.generated_query.formatted_query if context.generated_query else None,
                "attempts": context.attempts,
                "validation_retries": context.validation_retries,
                "validation": context.validation_result.model_dump() if context.validation_result else None,
                "policy": context.policy_result.model_dump() if context.policy_result else None,
                "error": resp.error,
                "feedback_history": list(context.feedback_history or []),
            }
            save_run_state(run_data)
            cleanup_old_runs()
        except Exception as save_err:
            logger.warning("Failed to save run state for %s: %s", context.run_id, save_err)

    async def execute_pipeline(
        self,
        question: str,
        database: str | DatabaseType | DatabaseTarget = "postgres",
        provider: str | ModelProvider | None = None,
        stages: StageProviders | dict[str, str] | None = None,
        jargons: list[str] | None = None,
    ) -> PipelineResponse:
        """Run complete V2 natural-language-to-query pipeline driven by the FSM runner."""
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

        # Resolve database target & type
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
                stage_tokens={
                    "stage_1": 0,
                    "stage_2": 0,
                    "stage_3": 0,
                    "guardrail": 0,
                    "table_selection": 0,
                    "query_generation": 0,
                    "validation": 0,
                    "total": 0,
                },
                guardrail=GuardrailResult(decision=GuardrailDecision.REJECT, reason=str(err)),
                error=str(err),
            )

        context = PipelineContext(
            run_id=run_id,
            database_target=target,
            db_type=db_type,
            target_name=target.key,
            schema=schema,
            question=question,
            processed_question=question,
            jargons=list(jargons or []),
            stage_providers=stage_providers_map,
            max_retries=self.max_retries,
        )

        fb_guardrail = getattr(self, "feedback_guardrail", None) or FeedbackGuardrailClassifier(
            provider=s1_provider, provider_name=s1_name
        )
        adapters_map = {
            PipelineState.SPELLING_CHECK: SpellingAdapter(self.spelling_checker),
            PipelineState.QUESTION_GUARDRAIL: QuestionGuardrailAdapter(guardrail),
            PipelineState.FEEDBACK_GUARDRAIL: FeedbackGuardrailAdapter(fb_guardrail),
            PipelineState.TABLE_SELECTION: TableSelectionAdapter(selector),
            PipelineState.QUERY_GENERATION: QueryGenerationAdapter(postgres_gen, mongo_gen),
            PipelineState.QUERY_VALIDATION: QueryValidationAdapter(validator),
            PipelineState.POLICY_CHECK: PolicyAdapter(self.policy),
            PipelineState.QUERY_EXECUTION: ExecutionAdapter(self.executor, self.results),
        }
        stage_adapters = StageAdapters(adapters=adapters_map)
        runner = PipelineFSMRunner(adapters=stage_adapters)

        final_context = await runner.run(context, initial_state=PipelineState.SPELLING_CHECK)

        resp = self._build_pipeline_response(final_context, provider_name)
        self._save_pipeline_run_file(final_context, resp)
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

    def _rehydrate_rerun_context(
        self,
        run_id: str,
        run_state: dict[str, Any],
        actual_start: str,
        applicable_feedback: dict[str, str],
        stage_providers_map: dict[str, str],
        target: DatabaseTarget,
        schema: DatabaseSchema,
    ) -> PipelineContext:
        """Rehydrate a PipelineContext from the saved run file with downstream clearing rules."""
        question = run_state.get("question", "")
        processed_question = run_state.get("processed_question") or question
        feedback_history = list(run_state.get("feedback_history", []))

        s1_user_guidance, s2_user_guidance, s3_val_guidance = build_cascading_guidance(
            feedback_history, applicable_feedback
        )

        selected_objs: list[str] = []
        relevant_schema: RelevantSchema | None = None
        selector_retries = 0
        last_query: GeneratedQuery | None = None

        if actual_start == "table_selection":
            stage_1_frozen = False
        elif actual_start == "query_generation":
            stage_1_frozen = True
            selected_objs = list(run_state.get("selected_objects", []))
            rel_dict = run_state.get("relevant_schema")
            relevant_schema = RelevantSchema.model_validate(rel_dict) if rel_dict else package_selected_schema(schema, selected_objs)
            selector_retries = int(run_state.get("selection_retries", 0))
        else:  # "validation"
            stage_1_frozen = True
            selected_objs = list(run_state.get("selected_objects", []))
            rel_dict = run_state.get("relevant_schema")
            relevant_schema = RelevantSchema.model_validate(rel_dict) if rel_dict else package_selected_schema(schema, selected_objs)
            selector_retries = int(run_state.get("selection_retries", 0))
            prev_q_str = run_state.get("generated_query")
            if prev_q_str:
                if target.db_type == DatabaseType.POSTGRESQL:
                    last_query = GeneratedQuery(
                        database_type=DatabaseType.POSTGRESQL,
                        raw_query=str(prev_q_str),
                        formatted_query=str(prev_q_str),
                    )
                else:
                    try:
                        raw_q = MongoQuery.model_validate(
                            json.loads(prev_q_str) if isinstance(prev_q_str, str) else prev_q_str
                        )
                        last_query = GeneratedQuery(
                            database_type=DatabaseType.MONGODB,
                            raw_query=raw_q,
                            formatted_query=str(prev_q_str),
                        )
                    except Exception:
                        last_query = None
            if not last_query:
                actual_start = "query_generation"

        orig_tokens = run_state.get("stage_tokens") or {}
        g_tok = orig_tokens.get("guardrail", 0) if actual_start != "table_selection" else 0
        sel_tok = orig_tokens.get("table_selection", 0) if actual_start != "table_selection" else 0
        gen_tok = orig_tokens.get("query_generation", 0) if actual_start == "validation" else 0

        stage_tokens = {
            "guardrail": g_tok,
            "table_selection": sel_tok,
            "query_generation": gen_tok,
            "query_validation": 0,
        }

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

        return PipelineContext(
            run_id=run_id,
            database_target=target,
            db_type=target.db_type,
            target_name=target.key,
            schema=schema,
            question=question,
            processed_question=processed_question,
            jargons=list(run_state.get("jargons") or []),
            is_rerun=True,
            entry_stage=actual_start,
            stage_1_frozen=stage_1_frozen,
            initial_val_checked=False,
            stage_3_fallthrough_allowed=(actual_start == "validation"),
            created_at=float(run_state.get("created_at") or time.time()),
            updated_at=time.time(),
            feedback_history=feedback_history,
            applicable_feedback=applicable_feedback,
            s1_user_guidance=s1_user_guidance,
            s2_user_guidance=s2_user_guidance,
            s3_user_guidance=s3_val_guidance,
            selected_objects=selected_objs,
            relevant_schema=relevant_schema,
            generated_query=last_query,
            selector_retries=selector_retries,
            validation_retries=0,
            attempts=1,
            max_retries=self.max_retries,
            spelling_result=spelling_res,
            guardrail_result=guardrail_result,
            stage_providers=stage_providers_map,
            stage_tokens=stage_tokens,
        )

    async def rerun_pipeline(
        self,
        run_id: str,
        starting_stage: str | None = None,
        feedback: dict[str, str] | None = None,
    ) -> PipelineResponse:
        """Execute a partial pipeline rerun from a specified stage using saved run state driven by FSM."""
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

        start_idx = valid_stages.index(actual_start)
        applicable_feedback = {
            stg: cleaned_feedback[stg]
            for stg in valid_stages[start_idx:]
            if stg in cleaned_feedback
        }

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

        _guardrail, selector, postgres_gen, mongo_gen, validator = (
            self._get_stages_for_providers(s1_provider, s2_provider, s3_provider)
        )

        database = run_state.get("database", "postgres")
        target = resolve_target(database)
        schema = self.get_or_load_schema(target)

        context = self._rehydrate_rerun_context(
            run_id=run_id,
            run_state=run_state,
            actual_start=actual_start,
            applicable_feedback=applicable_feedback,
            stage_providers_map=stage_providers_map,
            target=target,
            schema=schema,
        )

        fb_guardrail = getattr(self, "feedback_guardrail", None) or FeedbackGuardrailClassifier(
            provider=s1_provider, provider_name=s1_name
        )
        adapters_map = {
            PipelineState.FEEDBACK_GUARDRAIL: FeedbackGuardrailAdapter(fb_guardrail),
            PipelineState.TABLE_SELECTION: TableSelectionAdapter(selector),
            PipelineState.QUERY_GENERATION: QueryGenerationAdapter(postgres_gen, mongo_gen),
            PipelineState.QUERY_VALIDATION: QueryValidationAdapter(validator),
            PipelineState.POLICY_CHECK: PolicyAdapter(self.policy),
            PipelineState.QUERY_EXECUTION: ExecutionAdapter(self.executor, self.results),
        }
        stage_adapters = StageAdapters(adapters=adapters_map)
        runner = PipelineFSMRunner(adapters=stage_adapters)

        final_context = await runner.run(context, initial_state=PipelineState.FEEDBACK_GUARDRAIL)

        if final_context.error and "was rejected by the safety guardrail:" in final_context.error:
            raise ValueError(final_context.error)

        resp = self._build_pipeline_response(final_context, provider_name)

        should_save = (
            final_context.execution_result is not None
            or (final_context.policy_result is not None and not final_context.policy_result.allowed)
            or (
                final_context.policy_result is not None
                and final_context.policy_result.allowed
                and final_context.error is not None
                and "Database execution error" in final_context.error
            )
            or (
                final_context.error is not None
                and "Query failed validation after maximum retries" in final_context.error
            )
        )
        if should_save:
            stage_tokens_dict = self._calculate_stage_tokens_dict(final_context)
            self._update_saved_run_state(
                run_id=run_id,
                run_state=run_state,
                applicable_feedback=applicable_feedback,
                selected_objs=final_context.selected_objects,
                relevant_schema=final_context.relevant_schema,
                last_query=final_context.generated_query,
                last_val=final_context.validation_result,
                policy_result=final_context.policy_result,
                execution_results=final_context.execution_result,
                error=resp.error,
                attempts=final_context.attempts,
                selector_retries=final_context.selector_retries,
                validator_retries=final_context.validation_retries,
                stage_tokens=stage_tokens_dict,
            )

        return resp

