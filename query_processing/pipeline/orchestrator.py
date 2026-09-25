"""End-to-end pipeline orchestrator for NL2AnyQuery."""

import asyncio
import logging
from typing import Any

from ingestion.databases.detector import detect_database_type
from ingestion.schema.manager import get_default_schema_path
from ingestion.schema.toml_store import load_schema_file
from query_processing.core.config import settings
from query_processing.models.pipeline import (
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
    ValidationErrorType,
    ValidationResult,
)
from query_processing.models.schema import DatabaseSchema, DatabaseType
from query_processing.nlp.linguistic import LinguisticAnalyzer
from query_processing.pipeline.executor import QueryExecutor
from query_processing.pipeline.expansion import SchemaExpander
from query_processing.pipeline.generators.mongo import MongoQueryGenerator
from query_processing.pipeline.generators.postgres import PostgresQueryGenerator
from query_processing.pipeline.guardrail import GuardrailClassifier
from query_processing.pipeline.planner import QueryPlanner
from query_processing.pipeline.policy import SafetyPolicyValidator
from query_processing.pipeline.results import ResultProcessor
from query_processing.pipeline.selector import TableSelector
from query_processing.pipeline.semantic import SemanticAnalyzer
from query_processing.pipeline.validator import QueryValidator
from query_processing.providers.embedding.base import EmbeddingProvider
from query_processing.providers.embedding.koboldcpp import KoboldCppEmbeddingProvider
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider
from query_processing.retrieval.store import EmbeddingStore
from query_processing.retrieval.vector import VectorRetriever

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
        semantic: SemanticAnalyzer | None = None,
        linguistic: LinguisticAnalyzer | None = None,
        selector: TableSelector | None = None,
        expander: SchemaExpander | None = None,
        planner: QueryPlanner | None = None,
        postgres_gen: PostgresQueryGenerator | None = None,
        mongo_gen: MongoQueryGenerator | None = None,
        validator: QueryValidator | None = None,
        policy: SafetyPolicyValidator | None = None,
        executor: QueryExecutor | None = None,
        results: ResultProcessor | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.embedding_provider = embedding_provider or KoboldCppEmbeddingProvider()
        self.embedding_store = embedding_store
        self.vector_retriever = vector_retriever
        self.guardrail = guardrail or GuardrailClassifier(provider=self.provider)
        self.semantic = semantic or SemanticAnalyzer(provider=self.provider)
        self.linguistic = linguistic or LinguisticAnalyzer()
        self.selector = selector or TableSelector(provider=self.provider)
        self.expander = expander or SchemaExpander()
        self.planner = planner or QueryPlanner(provider=self.provider)
        self.postgres_gen = postgres_gen or PostgresQueryGenerator(provider=self.provider)
        self.mongo_gen = mongo_gen or MongoQueryGenerator(provider=self.provider)
        self.validator = validator or QueryValidator(provider=self.provider)
        self.policy = policy or SafetyPolicyValidator()
        self.executor = executor or QueryExecutor()
        self.results = results or ResultProcessor()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

        self._schemas: dict[DatabaseType, DatabaseSchema] = {}
        self._vector_retrievers: dict[DatabaseType, VectorRetriever] = {}

    def get_or_load_schema(self, db_type: DatabaseType) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file."""
        if db_type in self._schemas:
            return self._schemas[db_type]
        schema_path = get_default_schema_path(db_type)
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Canonical schema not found at {schema_path}. Run 'init-schema' first."
            )
        schema = load_schema_file(schema_path)
        self._schemas[db_type] = schema
        return schema

    def _get_or_load_vector_retriever(self, db_type: DatabaseType) -> VectorRetriever:
        """Get or initialize the vector retriever for this database's embeddings.

        An explicitly injected vector_retriever/embedding_store overrides for all
        database types; otherwise each type lazily loads its own embeddings file.
        """
        if self.vector_retriever is not None:
            return self.vector_retriever

        if db_type not in self._vector_retrievers:
            store = self.embedding_store or EmbeddingStore(
                settings.postgres_embeddings_path
                if db_type == DatabaseType.POSTGRESQL
                else settings.mongo_embeddings_path
            )
            self._vector_retrievers[db_type] = VectorRetriever(store)
        return self._vector_retrievers[db_type]

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
    ) -> PipelineResponse:
        cand_list = candidates or []
        query_repr = (
            query.formatted_query
            if query
            else ("No query generated" if (attempts > 1 and not basic_answer and not error) else None)
        )
        return PipelineResponse(
            database=db_type.value,
            question=question,
            guardrail=guardrail,
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

    async def _select_tables_vector(
        self,
        question: str,
        analysis: QuestionAnalysis,
        query_vector: list[float],
        schema: DatabaseSchema,
        db_type: DatabaseType,
    ) -> tuple[list[str], list[CandidateTable], int, str | None]:
        """Embedding-based table candidate retrieval + Table Selector SLM bounded retry."""
        try:
            retriever = self._get_or_load_vector_retriever(db_type)
            initial_candidates = retriever.retrieve(query_vector)
        except Exception as err:
            logger.error("Vector retrieval failed: %s", err)
            return [], [], 0, f"Vector retrieval failed: {err}"

        if not initial_candidates:
            return [], [], 0, "I could not find any relevant tables in the database schema matching your question."

        current_candidates = list(initial_candidates)
        attempt = 0
        feedback: str | None = None

        while True:
            selection = await self.selector.select(
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

            sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: settings.max_candidate_tables]
            current_candidates = [
                CandidateTable(table_name=tbl, similarity=sim, rank=r)
                for r, (tbl, sim) in enumerate(sorted_merged, start=1)
            ]
            missing_info = f" Missing: {selection.missing_objects}." if selection.missing_objects else ""
            feedback = (
                f"Previous attempt selected {selection.selected_objects} but was marked insufficient. "
                f"Reason: {selection.reason}.{missing_info}"
            )

    async def _plan_generate_validate(
        self,
        question: str,
        analysis: QuestionAnalysis,
        relevant_schema: RelevantSchema,
        db_type: DatabaseType,
    ) -> tuple[Any, GeneratedQuery | None, ValidationResult | None, int, str | None]:
        """Query Planner -> Generator -> Validator bounded targeted retry loop."""
        generator = self.postgres_gen if db_type == DatabaseType.POSTGRESQL else self.mongo_gen
        query_plan = await self.planner.plan(
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
                query_plan = await self.planner.plan(
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
                if retries >= self.max_retries:
                    return query_plan, last_query, last_val, retries, f"Query generation failed after maximum retries: {gen_err}"
                retries += 1
                gen_feedback = f"Generation error: {gen_err}"
                continue

            last_val = await self.validator.validate(
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
            feedback_msg = "; ".join(last_val.issues)
            if last_val.suggestion:
                feedback_msg += f". Suggestion: {last_val.suggestion}"

            if last_val.error_type == ValidationErrorType.PLANNER_ERROR:
                planner_feedback = feedback_msg
                query_plan = None
            else:
                gen_feedback = feedback_msg

        return query_plan, last_query, last_val, retries, "Query failed validation after maximum retries. Manual review required."

    async def execute_pipeline(
        self,
        question: str,
        database: str | DatabaseType = "postgres",
    ) -> PipelineResponse:
        """Run complete NL-to-query pipeline with clean stages and bounded retries."""
        # 1. Resolve database type & load schema
        if isinstance(database, str):
            db_clean = database.strip().lower()
            db_type = (
                DatabaseType.POSTGRESQL
                if db_clean in ("postgres", "postgresql")
                else (DatabaseType.MONGODB if db_clean in ("mongo", "mongodb") else detect_database_type(database))
            )
        else:
            db_type = database

        try:
            schema = self.get_or_load_schema(db_type)
        except Exception as err:
            return self._build_response(
                db_type=db_type,
                question=question,
                guardrail=GuardrailResult(decision=GuardrailDecision.REJECT, reason=str(err)),
                error=str(err),
            )

        # 2. Stage 1: Guardrail
        guardrail_result = await self.guardrail.classify(question)
        if guardrail_result.decision == GuardrailDecision.REJECT:
            return self._build_response(
                db_type=db_type,
                question=question,
                guardrail=guardrail_result,
                error="Sorry, I can't help with this.",
            )

        if guardrail_result.decision == GuardrailDecision.BASIC:
            basic_ans = self.guardrail.handle_basic_question(
                question=question,
                database_type=db_type.value,
                database_name=schema.database_name,
            )
            return self._build_response(
                db_type=db_type,
                question=question,
                guardrail=guardrail_result,
                basic_answer=basic_ans,
            )

        # 3. Stages 2 & 3: Concurrent Embedding, Semantic SLM & spaCy Linguistic Analysis
        try:
            query_vector, semantic_res, linguistic_res = await asyncio.gather(
                self.embedding_provider.embed(question),
                self.semantic.analyze(question),
                asyncio.to_thread(self.linguistic.analyze, question),
            )
        except Exception as err:
            return self._build_response(
                db_type=db_type,
                question=question,
                guardrail=guardrail_result,
                error=f"Failed to analyze query: {err}",
            )

        analysis = QuestionAnalysis(
            question=question,
            subjective=semantic_res.subjective,
            objective=semantic_res.objective,
            nouns=linguistic_res.nouns,
            verbs=linguistic_res.verbs,
            entities=linguistic_res.entities,
        )

        # 4. Stages 5 & 6: Candidate Table Retrieval & Selection
        # Both PostgreSQL and MongoDB retrieve candidates from their own
        # precomputed description embeddings, produced by the ingestion pipeline.
        selected_objs, candidates, sel_retries, sel_err = await self._select_tables_vector(
            question=question,
            analysis=analysis,
            query_vector=query_vector,
            schema=schema,
            db_type=db_type,
        )

        if sel_err or not selected_objs:
            return self._build_response(
                db_type=db_type,
                question=question,
                guardrail=guardrail_result,
                semantic=semantic_res,
                linguistic=linguistic_res.model_dump(),
                candidates=candidates,
                selected=selected_objs,
                selection_retries=sel_retries,
                error=sel_err or "I could not find any relevant tables or collections in the database schema matching your question.",
            )

        # 5. Stage 7: Deterministic Schema Expansion
        relevant_schema = self.expander.expand(selected_object_names=selected_objs, schema=schema)

        # 6. Stages 8, 9, 10: Query Planning, Generation & Validation Loop
        query_plan, last_query, last_val, val_retries, val_err = await self._plan_generate_validate(
            question=question,
            analysis=analysis,
            relevant_schema=relevant_schema,
            db_type=db_type,
        )

        attempts = val_retries + 1
        ctx: dict[str, Any] = {
            "db_type": db_type,
            "question": question,
            "guardrail": guardrail_result,
            "semantic": semantic_res,
            "linguistic": linguistic_res.model_dump(),
            "candidates": candidates,
            "selected": selected_objs,
            "selection_retries": sel_retries,
            "schema": relevant_schema,
            "plan": query_plan,
            "query": last_query,
            "attempts": attempts,
            "validation_retries": val_retries,
            "validation": last_val,
        }

        if val_err or not last_query or not last_val or not last_val.valid:
            return self._build_response(**ctx, error=val_err)

        # 7. Stage 11: Deterministic Safety Policy Check
        policy_result = self.policy.check(last_query, schema=relevant_schema)
        if not policy_result.allowed:
            return self._build_response(
                **ctx,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )

        # 8. Stages 12 & 13: Execution and Result Processing
        try:
            columns, raw_rows = self.executor.execute(last_query)
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
