"""End-to-end pipeline orchestrator for NL2AnyQuery V2."""

import logging
from typing import Any

from backend.src.data.clients.detector import detect_database_type
from backend.src.data.models.targets import DatabaseTarget, as_target, resolve_target
from backend.src.data.repositories import description_repository, paths, schema_repository
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
        guardrail = (
            self.guardrail
            if self._custom_guardrail or s1_provider is self.provider
            else GuardrailClassifier(provider=s1_provider)
        )
        selector = (
            self.selector
            if self._custom_selector or s1_provider is self.provider
            else TableSelector(provider=s1_provider, cache_manager=self.cache_manager)
        )
        postgres_gen = (
            self.postgres_gen
            if self._custom_postgres_gen or s2_provider is self.provider
            else PostgresQueryGenerator(provider=s2_provider)
        )
        mongo_gen = (
            self.mongo_gen
            if self._custom_mongo_gen or s2_provider is self.provider
            else MongoQueryGenerator(provider=s2_provider)
        )
        validator = (
            self.validator
            if self._custom_validator or s3_provider is self.provider
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
            return self._build_response(
                **ctx,
                error="Query failed validation after maximum retries. Manual review required.",
            )

        # 5. Stage 6: Deterministic Safety Policy Check
        policy_result = self.policy.check(last_query, schema=relevant_schema)
        if not policy_result.allowed:
            return self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )

        # 6. Stages 7 & 8: Execution and Result Processing
        try:
            columns, raw_rows = self.executor.execute(
                last_query,
                connection=target.connection,
                database=target.mongo_database,
            )
        except Exception as exec_err:
            logger.error("Execution error: %s", exec_err)
            return self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Database execution error: {exec_err}",
            )

        execution_results = self.results.process(columns, raw_rows)
        return self._build_response(
            **ctx,
            policy=policy_result,
            results=execution_results,
        )
