"""End-to-end pipeline orchestrator for NL2AnyQuery."""

import asyncio
import logging
from typing import Any

from backend.src.data.clients.detector import detect_database_type
from backend.src.data.models.targets import DatabaseTarget, as_target, resolve_target
from backend.src.data.repositories import paths, schema_repository
from backend.src.core.ingestion.schema.manager import get_default_schema_path
from backend.src.core.ingestion.schema.toml_store import load_schema_file
from backend.src.config import settings
from backend.src.schemas.pipeline import (
    CandidateTable,
    ExecutionResult,
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    PipelineResponse,
    PolicyResult,
    QuestionAnalysis,
    RelevantSchema,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.schemas.pipeline import JoinPathResolution, QueryPlan
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.core.query_processing.nlp.linguistic import LinguisticAnalyzer
from backend.src.core.query_processing.pipeline.executor import QueryExecutor
from backend.src.core.query_processing.pipeline.expansion import SchemaExpander
from backend.src.core.query_processing.pipeline.generators.mongo import MongoQueryGenerator
from backend.src.core.query_processing.pipeline.generators.postgres import PostgresQueryGenerator
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.path_selector import JoinPathSelector
from backend.src.core.query_processing.pipeline.planner import QueryPlanner
from backend.src.core.query_processing.pipeline.policy import SafetyPolicyValidator
from backend.src.core.query_processing.pipeline.results import ResultProcessor
from backend.src.core.query_processing.pipeline.selector import TableSelector
from backend.src.core.query_processing.pipeline.semantic import SemanticAnalyzer
from backend.src.core.query_processing.pipeline.spelling import SpellingChecker
from backend.src.core.query_processing.pipeline.validator import QueryValidator
from backend.src.control.providers.embedding.base import EmbeddingProvider
from backend.src.control.providers.embedding.koboldcpp import KoboldCppEmbeddingProvider
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.factory import resolve_model_provider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.core.query_processing.retrieval.store import EmbeddingStore
from backend.src.core.query_processing.retrieval.vector import VectorRetriever
from backend.src.core.query_processing.pipeline.descriptions_client import DescriptionsClient
from backend.src.utils.text_utils import clean_unreadable_characters, strip_ansi_escapes

logger = logging.getLogger(__name__)


class NL2AnyQueryOrchestrator:
    """Coordinates the deterministic pipeline from natural language to database results."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_store: EmbeddingStore | None = None,
        vector_retriever: VectorRetriever | None = None,
        guardrail: GuardrailClassifier | None = None,
        spelling_checker: SpellingChecker | None = None,
        semantic: SemanticAnalyzer | None = None,
        linguistic: LinguisticAnalyzer | None = None,
        selector: TableSelector | None = None,
        expander: SchemaExpander | None = None,
        path_selector: JoinPathSelector | None = None,
        planner: QueryPlanner | None = None,
        postgres_gen: PostgresQueryGenerator | None = None,
        mongo_gen: MongoQueryGenerator | None = None,
        validator: QueryValidator | None = None,
        policy: SafetyPolicyValidator | None = None,
        executor: QueryExecutor | None = None,
        results: ResultProcessor | None = None,
        max_retries: int | None = None,
        descriptions_client: DescriptionsClient | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.embedding_provider = embedding_provider or KoboldCppEmbeddingProvider()
        self.embedding_store = embedding_store
        self.vector_retriever = vector_retriever
        self.guardrail = guardrail or GuardrailClassifier(provider=self.provider)
        self.spelling_checker = spelling_checker or SpellingChecker()
        self.semantic = semantic or SemanticAnalyzer(provider=self.provider)
        self.linguistic = linguistic or LinguisticAnalyzer()
        self.selector = selector or TableSelector(provider=self.provider)
        self.expander = expander or SchemaExpander()
        self.path_selector = path_selector or JoinPathSelector(provider=self.provider)
        self.descriptions_client = descriptions_client or DescriptionsClient()
        self.planner = planner or QueryPlanner(provider=self.provider)
        self.postgres_gen = postgres_gen or PostgresQueryGenerator(provider=self.provider)
        self.mongo_gen = mongo_gen or MongoQueryGenerator(provider=self.provider)
        self.validator = validator or QueryValidator(provider=self.provider)
        self.policy = policy or SafetyPolicyValidator()
        self.executor = executor or QueryExecutor()
        self.results = results or ResultProcessor()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

        self._custom_guardrail = guardrail is not None
        self._custom_semantic = semantic is not None
        self._custom_selector = selector is not None
        self._custom_path_selector = path_selector is not None
        self._custom_planner = planner is not None
        self._custom_postgres_gen = postgres_gen is not None
        self._custom_mongo_gen = mongo_gen is not None
        self._custom_validator = validator is not None

        self._schemas: dict[Any, DatabaseSchema] = {}
        self._vector_retrievers: dict[Any, VectorRetriever] = {}

    def _get_stages_for_provider(self, active_provider: ModelProvider):
        """Return stage instances bound to the active provider for this execution."""
        if active_provider is self.provider:
            return (
                self.guardrail,
                self.semantic,
                self.selector,
                self.path_selector,
                self.planner,
                self.postgres_gen,
                self.mongo_gen,
                self.validator,
            )
        guardrail = (
            self.guardrail
            if self._custom_guardrail
            else GuardrailClassifier(provider=active_provider)
        )
        semantic = (
            self.semantic
            if self._custom_semantic
            else SemanticAnalyzer(provider=active_provider)
        )
        selector = (
            self.selector
            if self._custom_selector
            else TableSelector(provider=active_provider)
        )
        path_selector = (
            self.path_selector
            if self._custom_path_selector
            else JoinPathSelector(provider=active_provider)
        )
        planner = (
            self.planner
            if self._custom_planner
            else QueryPlanner(provider=active_provider)
        )
        postgres_gen = (
            self.postgres_gen
            if self._custom_postgres_gen
            else PostgresQueryGenerator(provider=active_provider)
        )
        mongo_gen = (
            self.mongo_gen
            if self._custom_mongo_gen
            else MongoQueryGenerator(provider=active_provider)
        )
        validator = (
            self.validator
            if self._custom_validator
            else QueryValidator(provider=active_provider)
        )
        return (
            guardrail,
            semantic,
            selector,
            path_selector,
            planner,
            postgres_gen,
            mongo_gen,
            validator,
        )

    def get_or_load_schema(self, database: DatabaseTarget | DatabaseType | str) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file."""
        if database in self._schemas:
            return self._schemas[database]

        try:
            target = as_target(database) if not isinstance(database, str) else resolve_target(database)
            if target.key in self._schemas:
                return self._schemas[target.key]
            schema_path = paths.schema_path(target)
        except Exception:
            schema_path = get_default_schema_path(database)
            target = None

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Canonical schema not found at {schema_path}. Run 'init-schema' first."
            )
        schema = schema_repository.load_schema(schema_path)
        key = target.key if target else database
        self._schemas[key] = schema
        return schema

    def _get_or_load_vector_retriever(self, database: DatabaseTarget | DatabaseType | str) -> VectorRetriever:
        """Get or initialize the vector retriever for this database's embeddings.

        An explicitly injected vector_retriever/embedding_store overrides for all
        database types; otherwise each target lazily loads its own embeddings file.
        """
        if self.vector_retriever is not None:
            return self.vector_retriever

        if database in self._vector_retrievers:
            return self._vector_retrievers[database]

        try:
            target = as_target(database) if not isinstance(database, str) else resolve_target(database)
            key = target.key
            emb_path = str(paths.embeddings_path(target))
        except Exception:
            key = database
            emb_path = (
                settings.postgres_embeddings_path
                if database == DatabaseType.POSTGRESQL
                else settings.mongo_embeddings_path
            )

        if key not in self._vector_retrievers:
            store = self.embedding_store or EmbeddingStore(emb_path)
            self._vector_retrievers[key] = VectorRetriever(store)
        return self._vector_retrievers[key]

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
        join_path_resolution: JoinPathResolution | None = None,
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
        query_repr = (
            query.formatted_query
            if query
            else ("No query generated" if (attempts > 1 and not basic_answer and not error) else None)
        )
        db_label = target_name if (target_name and target_name not in ("postgres", "postgresql", "mongo", "mongodb")) else db_type.value
        return PipelineResponse(
            database=db_label,
            question=question,
            provider=provider,
            guardrail=guardrail,
            spelling_correction=spelling_correction,
            basic_answer=basic_answer,
            semantic_analysis=semantic,
            linguistic_analysis=linguistic,
            candidate_tables=cand_list,
            candidate_objects=[c.table_name for c in cand_list],
            selected_objects=selected or [],
            selection_retries=selection_retries,
            join_path_resolution=join_path_resolution,
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

    async def _select_tables_vector(
        self,
        question: str,
        analysis: QuestionAnalysis,
        query_vector: list[float],
        schema: DatabaseSchema,
        database: Any,
        selector: TableSelector | None = None,
    ) -> tuple[list[str], list[CandidateTable], int, str | None]:
        """Embedding-based table candidate retrieval + Table Selector SLM bounded retry."""
        active_selector = selector or self.selector
        try:
            retriever = self._get_or_load_vector_retriever(database)
            initial_candidates = retriever.retrieve(query_vector)
            if not initial_candidates and (analysis.subjective or analysis.nouns):
                fallback_hint = " ".join(dict.fromkeys(analysis.subjective + analysis.nouns))
                try:
                    fallback_vec = await self.embedding_provider.embed(fallback_hint)
                    initial_candidates = retriever.retrieve(fallback_vec)
                except Exception as fallback_err:
                    logger.warning("Fallback candidate retrieval failed: %s", fallback_err)
        except Exception as err:
            logger.error("Vector retrieval failed: %s", err)
            return [], [], 0, f"Vector retrieval failed: {err}"

        if not initial_candidates:
            return [], [], 0, "I could not find any relevant tables in the database schema matching your question."

        current_candidates = list(initial_candidates)
        attempt = 0
        feedback: str | None = None

        while True:
            selection = await active_selector.select(
                question_analysis=analysis,
                candidates=current_candidates,
                schema=schema,
                feedback=feedback,
            )
            if selection.sufficient and selection.selected_objects:
                return selection.selected_objects, current_candidates, attempt, None

            if attempt >= self.max_retries:
                logger.warning("Table selector failed sufficiency after %d retries.", attempt)
                return selection.selected_objects, current_candidates, attempt, (
                    "Relevant schema could not be identified reliably after maximum retries."
                )

            attempt += 1
            hint = selection.retrieval_hint or f"{question} {' '.join(selection.missing_objects)}"
            try:
                hint_vec = await self.embedding_provider.embed(hint)
                new_cands = retriever.retrieve(hint_vec)
            except Exception as embed_err:
                logger.warning("Retry embedding retrieval failed: %s", embed_err)
                new_cands = []

            merged = {c.table_name: c.similarity for c in current_candidates}
            for nc in new_cands:
                if nc.table_name not in merged or nc.similarity > merged[nc.table_name]:
                    merged[nc.table_name] = nc.similarity

            sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: settings.effective_similarity_top_k]
            current_candidates = [
                CandidateTable(table_name=tbl, similarity=sim, rank=r)
                for r, (tbl, sim) in enumerate(sorted_merged, start=1)
            ]
            missing_info = f" Missing: {selection.missing_objects}." if selection.missing_objects else ""
            feedback = (
                f"Previous attempt selected {selection.selected_objects} but was marked insufficient. "
                f"Reason: {selection.reason}.{missing_info}"
            )

    async def _enrich_schema_descriptions(
        self,
        relevant_schema: RelevantSchema,
        database: Any,
    ) -> None:
        """Fetch table and column descriptions from the Descriptions API and populate in place."""
        if not self.descriptions_client:
            return

        target_slug = getattr(database, "key", None) or (
            database.value if isinstance(database, DatabaseType) else str(database)
        )
        for obj in relevant_schema.objects:
            try:
                desc = await self.descriptions_client.get_table_description(
                    database_type=target_slug,
                    table_name=obj.name,
                )
                if desc:
                    if desc.description:
                        obj.description = desc.description
                    for col_name, col_desc in desc.columns.items():
                        if col_desc:
                            field = obj.get_field(col_name)
                            if field:
                                field.description = col_desc
            except Exception as err:
                logger.warning(
                    "Failed to enrich descriptions for %s.%s: %s",
                    target_slug,
                    obj.name,
                    err,
                )

    async def _resolve_paths_and_expand(
        self,
        analysis: QuestionAnalysis,
        selected_objs: list[str],
        schema: DatabaseSchema,
        database: Any,
        path_selector: JoinPathSelector | None = None,
    ) -> tuple[RelevantSchema, JoinPathResolution]:
        """Resolve how the selected objects join, then expand and enrich the slice.

        Selection names the objects a question is about; it does not say how they
        connect, and two selected tables with no column in common leave the
        generator to invent a join. The path selector walks the schema's real
        relationships and returns the objects actually needed -- the selected
        ones the question still wants, plus every intermediate table the chosen
        paths pass through -- which is what the expander then materializes.
        """
        active_path_selector = path_selector or self.path_selector
        resolution = await active_path_selector.resolve(
            question_analysis=analysis,
            selected_objects=selected_objs,
            schema=schema,
        )
        if resolution.connector_objects:
            logger.info(
                "Join path resolution added connector objects %s via paths %s",
                resolution.connector_objects,
                resolution.chosen_path_ids,
            )
        if resolution.dropped_objects:
            logger.info(
                "Join path resolution dropped unconnected objects %s",
                resolution.dropped_objects,
            )

        relevant_schema = self.expander.expand(
            selected_object_names=resolution.resolved_objects,
            schema=schema,
        )
        await self._enrich_schema_descriptions(relevant_schema, database)
        return relevant_schema, resolution

    async def _plan_generate_validate(
        self,
        question: str,
        analysis: QuestionAnalysis,
        relevant_schema: RelevantSchema,
        db_type: DatabaseType,
        initial_plan: QueryPlan | None = None,
        planner: QueryPlanner | None = None,
        generator: Any = None,
        validator: QueryValidator | None = None,
    ) -> tuple[Any, GeneratedQuery | None, ValidationResult | None, int, str | None]:
        """Query Planner -> Generator -> Validator bounded targeted retry loop."""
        active_generator = generator or (
            self.postgres_gen if db_type == DatabaseType.POSTGRESQL else self.mongo_gen
        )
        active_planner = planner or self.planner
        active_validator = validator or self.validator

        query_plan = initial_plan
        if query_plan is None:
            query_plan = await active_planner.plan(
                question_analysis=analysis,
                relevant_schema=relevant_schema,
                database_type=db_type,
            )

        last_query: GeneratedQuery | None = None
        last_val: ValidationResult | None = None
        gen_feedback: str | None = None
        planner_feedback: str | None = None
        retries = 0

        while retries <= self.max_retries:
            if query_plan is None:
                query_plan = await active_planner.plan(
                    question_analysis=analysis,
                    relevant_schema=relevant_schema,
                    database_type=db_type,
                    feedback=planner_feedback,
                )
                planner_feedback = None

            try:
                last_query = await active_generator.generate_query(
                    question=question,
                    plan=query_plan,
                    schema=relevant_schema,
                    feedback=gen_feedback,
                )
                gen_feedback = None
            except Exception as gen_err:
                if retries >= self.max_retries:
                    return query_plan, last_query, last_val, retries, f"Query generation failed after maximum retries: {gen_err}"
                retries += 1
                gen_feedback = f"Generation error: {gen_err}"
                continue

            last_val = await active_validator.validate(
                question=question,
                plan=query_plan,
                generated_query=last_query,
                schema=relevant_schema,
            )

            if last_val.valid:
                return query_plan, last_query, last_val, retries, None

            if last_val.error_type == ValidationErrorType.UNSAFE:
                return query_plan, last_query, last_val, retries, "Generated query is unsafe. Execution rejected."

            if retries >= self.max_retries:
                return query_plan, last_query, last_val, retries, (
                    "Query failed validation after maximum retries. Manual review required."
                )

            retries += 1
            clean_issues = [
                clean_unreadable_characters(strip_ansi_escapes(str(i)))
                for i in last_val.issues
                if i
            ]
            feedback_msg = "; ".join(clean_issues)
            if last_val.suggestion:
                clean_sugg = clean_unreadable_characters(strip_ansi_escapes(str(last_val.suggestion)))
                feedback_msg += f". Suggestion: {clean_sugg}"
            feedback_msg = clean_unreadable_characters(strip_ansi_escapes(feedback_msg))

            if last_val.error_type == ValidationErrorType.PLANNER_ERROR:
                planner_feedback = feedback_msg
                query_plan = None
            else:
                gen_feedback = feedback_msg

        return query_plan, last_query, last_val, retries, "Query failed validation after maximum retries. Manual review required."

    async def execute_pipeline(
        self,
        question: str,
        database: str | DatabaseType | DatabaseTarget = "postgres",
        provider: str | ModelProvider | None = None,
        jargons: list[str] | None = None,
    ) -> PipelineResponse:
        """Run complete NL-to-query pipeline with clean stages and bounded retries."""
        provider_name = (
            provider.strip().lower()
            if isinstance(provider, str)
            else (getattr(provider, "model", "custom") if provider is not None else "koboldcpp")
        )
        active_provider = (
            resolve_model_provider(provider)
            if isinstance(provider, str)
            else (provider or self.provider)
        )
        (
            guardrail,
            semantic,
            selector,
            path_selector,
            planner,
            postgres_gen,
            mongo_gen,
            validator,
        ) = self._get_stages_for_provider(active_provider)

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

        try:
            schema = self.get_or_load_schema(target)
        except Exception as err:
            return self._build_response(
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                guardrail=GuardrailResult(decision=GuardrailDecision.REJECT, reason=str(err)),
                error=str(err),
            )

        # 2. Stage 1: Guardrail
        guardrail_result = await guardrail.classify(question)
        if guardrail_result.decision == GuardrailDecision.REJECT:
            return self._build_response(
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                guardrail=guardrail_result,
                error="Sorry, I can't help with this.",
            )

        if guardrail_result.decision == GuardrailDecision.BASIC:
            basic_ans = guardrail.handle_basic_question(
                question=question,
                database_type=target.key,
                database_name=schema.database_name,
            )
            return self._build_response(
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                guardrail=guardrail_result,
                basic_answer=basic_ans,
            )

        # 3. Stage 1.5: SymSpell Spelling Checker (immediately after Guardrail, before Semantic Analysis)
        spelling_res = self.spelling_checker.check(question=question, jargons=jargons)
        processed_question = spelling_res.corrected_question

        # 4. Stages 2 & 3: Concurrent Embedding, Semantic SLM & spaCy Linguistic Analysis
        try:
            query_vector, semantic_res, linguistic_res = await asyncio.gather(
                self.embedding_provider.embed(processed_question),
                semantic.analyze(processed_question),
                asyncio.to_thread(self.linguistic.analyze, processed_question),
            )
        except Exception as err:
            return self._build_response(
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                guardrail=guardrail_result,
                spelling_correction=spelling_res,
                error=f"Failed to analyze query: {err}",
            )

        analysis = QuestionAnalysis(
            question=processed_question,
            subjective=semantic_res.subjective,
            objective=semantic_res.objective,
            nouns=linguistic_res.nouns,
            verbs=linguistic_res.verbs,
            entities=linguistic_res.entities,
        )

        # 5. Stages 5 & 6: Candidate Table Retrieval & Selection
        selected_objs, candidates, sel_retries, sel_err = await self._select_tables_vector(
            question=processed_question,
            analysis=analysis,
            query_vector=query_vector,
            schema=schema,
            database=target,
            selector=selector,
        )

        if sel_err or not selected_objs:
            return self._build_response(
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=provider_name,
                guardrail=guardrail_result,
                spelling_correction=spelling_res,
                semantic=semantic_res,
                linguistic=linguistic_res.model_dump(),
                candidates=candidates,
                selected=selected_objs,
                selection_retries=sel_retries,
                error=sel_err or "I could not find any relevant tables or collections in the database schema matching your question.",
            )

        # 6. Stage 6.5 & 7: Join Path Resolution, Deterministic Schema Expansion
        #    and description enrichment via the Descriptions API.
        relevant_schema, join_resolution = await self._resolve_paths_and_expand(
            analysis=analysis,
            selected_objs=selected_objs,
            schema=schema,
            database=target,
            path_selector=path_selector,
        )

        # 7. Stage 8: Query Planning (with feedback loop to Table Selector if required tables are missing)
        planner_retries = 0
        query_plan = None
        total_sel_retries = sel_retries
        while planner_retries <= self.max_retries:
            query_plan = await planner.plan(
                question_analysis=analysis,
                relevant_schema=relevant_schema,
                database_type=db_type,
            )
            if not query_plan.missing_tables:
                break

            if planner_retries >= self.max_retries:
                logger.warning(
                    "Planner identified missing tables %s, but reached max retries.",
                    query_plan.missing_tables,
                )
                break

            planner_retries += 1
            total_sel_retries += 1
            missing_names = ", ".join(query_plan.missing_tables)
            selector_feedback = (
                f"The query plan identified that table(s) [{missing_names}] are required to answer "
                f"the question '{processed_question}', but their schema is not in the retrieved schema. "
                f"Please select ALL required tables (including {missing_names}) from the candidate tables."
            )
            re_selection = await selector.select(
                question_analysis=analysis,
                candidates=candidates,
                schema=schema,
                feedback=selector_feedback,
            )
            if re_selection.selected_objects:
                new_selected = list(dict.fromkeys(selected_objs + re_selection.selected_objects))
                if set(new_selected) == set(selected_objs):
                    break
                selected_objs = new_selected
                relevant_schema, join_resolution = await self._resolve_paths_and_expand(
                    analysis=analysis,
                    selected_objs=selected_objs,
                    schema=schema,
                    database=target,
                    path_selector=path_selector,
                )
            else:
                break

        # 8. Stages 9 & 10: Query Generation & Validation Loop
        query_plan, last_query, last_val, val_retries, val_err = await self._plan_generate_validate(
            question=processed_question,
            analysis=analysis,
            relevant_schema=relevant_schema,
            db_type=db_type,
            initial_plan=query_plan,
            planner=planner,
            generator=active_generator,
            validator=validator,
        )

        attempts = val_retries + 1
        ctx: dict[str, Any] = {
            "db_type": db_type,
            "target_name": target.key,
            "question": question,
            "provider": provider_name,
            "guardrail": guardrail_result,
            "spelling_correction": spelling_res,
            "semantic": semantic_res,
            "linguistic": linguistic_res.model_dump(),
            "candidates": candidates,
            "selected": selected_objs,
            "selection_retries": total_sel_retries,
            "join_path_resolution": join_resolution,
            "schema": relevant_schema,
            "plan": query_plan,
            "query": last_query,
            "attempts": attempts,
            "validation_retries": val_retries,
            "validation": last_val,
        }

        if val_err or not last_query or not last_val or not last_val.valid:
            return self._build_response(**ctx, error=val_err)

        # 9. Stage 11: Deterministic Safety Policy Check
        policy_result = self.policy.check(last_query, schema=relevant_schema)
        if not policy_result.allowed:
            return self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )

        # 10. Stages 12 & 13: Execution and Result Processing
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

