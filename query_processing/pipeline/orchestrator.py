"""End-to-end pipeline orchestrator for NL2AnyQuery."""

import logging
from pathlib import Path
from typing import Any

from ingestion.databases.detector import detect_database_type
from query_processing.models.pipeline import (
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    PipelineResponse,
    PolicyResult,
    QuestionAnalysis,
    RelevantSchema,
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
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider
from query_processing.retrieval.bm25 import BM25Retriever
from ingestion.schema.manager import get_default_schema_path
from ingestion.schema.toml_store import load_schema_file

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3


class NL2AnyQueryOrchestrator:
    """Coordinates the deterministic pipeline from natural language to database results."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
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
    ) -> None:
        self.provider = provider or KoboldCppProvider()
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

        # In-memory schema cache and BM25 index cache
        self._schemas: dict[DatabaseType, DatabaseSchema] = {}
        self._bm25_indices: dict[DatabaseType, BM25Retriever] = {}

    def get_or_load_schema(self, db_type: DatabaseType) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file."""
        if db_type in self._schemas:
            return self._schemas[db_type]

        schema_path = get_default_schema_path(db_type)
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Canonical schema not found at {schema_path}. "
                f"Please run 'uv run init-schema --database {db_type.value}' first."
            )

        schema = load_schema_file(schema_path)
        self._schemas[db_type] = schema
        self._bm25_indices[db_type] = BM25Retriever(schema)
        return schema

    async def execute_pipeline(
        self,
        question: str,
        database: str | DatabaseType = "postgres",
    ) -> PipelineResponse:
        """Run complete NL-to-query pipeline."""
        # 1. Resolve database type
        if isinstance(database, str):
            db_clean = database.strip().lower()
            if db_clean in ("postgres", "postgresql"):
                db_type = DatabaseType.POSTGRESQL
            elif db_clean in ("mongo", "mongodb"):
                db_type = DatabaseType.MONGODB
            else:
                db_type = detect_database_type(database)
        else:
            db_type = database

        # 2. Load canonical schema & BM25 retriever
        try:
            schema = self.get_or_load_schema(db_type)
        except Exception as err:
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=GuardrailResult(
                    decision=GuardrailDecision.REJECT,
                    reason=f"Failed to load schema: {err}",
                ),
                error=str(err),
            )

        retriever = self._bm25_indices[db_type]

        # 3. Stage 1: Guardrail
        guardrail_result = await self.guardrail.classify(question)

        # Handle REJECT
        if guardrail_result.decision == GuardrailDecision.REJECT:
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=guardrail_result,
                error="Sorry, I can't help with this.",
            )

        # Handle BASIC
        if guardrail_result.decision == GuardrailDecision.BASIC:
            basic_ans = self.guardrail.handle_basic_question(
                question=question,
                database_type=db_type.value,
                database_name=schema.database_name,
            )
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=guardrail_result,
                basic_answer=basic_ans,
            )

        # 4. Stage 2: Semantic Analysis
        semantic_result = await self.semantic.analyze(question)

        # 5. Stage 3: spaCy Linguistic Analysis
        linguistic_result = self.linguistic.analyze(question)

        # 6. Stage 4: Unified Question Analysis
        analysis = QuestionAnalysis(
            question=question,
            subjective=semantic_result.subjective,
            objective=semantic_result.objective,
            nouns=linguistic_result.nouns,
            verbs=linguistic_result.verbs,
            entities=linguistic_result.entities,
        )

        # 7. Stage 5: BM25 Candidate Retrieval
        bm25_query = analysis.to_bm25_query()
        candidates = retriever.retrieve(bm25_query)
        candidate_names = [c.object_name for c in candidates]

        # Refinement 4: Graceful handling for zero BM25 candidates
        if not candidates or all(c.score <= 0.0 for c in candidates):
            # Check if query had any overlap
            if not candidates or candidates[0].score == 0.0:
                return PipelineResponse(
                    database=db_type.value,
                    question=question,
                    guardrail=guardrail_result,
                    semantic_analysis=semantic_result,
                    linguistic_analysis=linguistic_result.model_dump(),
                    candidate_objects=candidate_names,
                    error="I could not find any relevant tables or collections in the database schema matching your question.",
                )

        # 8. Stage 6: Table / Collection Selection SLM
        selection_result = await self.selector.select(
            question_analysis=analysis,
            candidates=candidates,
            schema=schema,
        )

        # Refinement 4: Graceful handling for zero selected objects
        if not selection_result.selected_objects:
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=guardrail_result,
                semantic_analysis=semantic_result,
                linguistic_analysis=linguistic_result.model_dump(),
                candidate_objects=candidate_names,
                selected_objects=[],
                error="I could not find any relevant tables or collections in the database schema matching your question.",
            )

        # 9. Stage 7: Deterministic Relationship & Schema Expansion
        relevant_schema = self.expander.expand(
            selected_object_names=selection_result.selected_objects,
            schema=schema,
        )

        # 10. Stage 8: Query Planning
        query_plan = await self.planner.plan(
            question_analysis=analysis,
            relevant_schema=relevant_schema,
            database_type=db_type,
        )

        # 11. Stages 9 & 10: Generation, Validation & Retry Loop (Max 3 attempts)
        generator = (
            self.postgres_gen if db_type == DatabaseType.POSTGRESQL else self.mongo_gen
        )

        last_query: GeneratedQuery | None = None
        last_validation: ValidationResult | None = None
        feedback: str | None = None
        attempt = 0

        while attempt < MAX_GENERATION_ATTEMPTS:
            attempt += 1
            try:
                last_query = await generator.generate_query(
                    question=question,
                    plan=query_plan,
                    schema=relevant_schema,
                    feedback=feedback,
                )
            except Exception as gen_err:
                logger.warning("Query generation attempt %d failed: %s", attempt, gen_err)
                feedback = f"Generation error: {gen_err}"
                continue

            last_validation = await self.validator.validate(
                question=question,
                plan=query_plan,
                generated_query=last_query,
                schema=relevant_schema,
            )

            if last_validation.valid:
                break
            else:
                issue_str = "; ".join(last_validation.issues)
                if last_validation.suggestion:
                    issue_str += f". Suggestion: {last_validation.suggestion}"
                feedback = issue_str
                logger.info(
                    "Query generation attempt %d failed validation. Retrying with feedback: %s",
                    attempt,
                    feedback,
                )

        if not last_query or not last_validation or not last_validation.valid:
            query_repr = (
                last_query.formatted_query
                if last_query
                else "No query generated"
            )
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=guardrail_result,
                semantic_analysis=semantic_result,
                linguistic_analysis=linguistic_result.model_dump(),
                candidate_objects=candidate_names,
                selected_objects=selection_result.selected_objects,
                relevant_schema=relevant_schema.model_dump(),
                query_plan=query_plan,
                generated_query=query_repr,
                attempts=attempt,
                validation=last_validation,
                error="Query failed validation after 3 attempts. Manual review required.",
            )

        # 12. Stage 11: Deterministic Safety Policy Check
        policy_result = self.policy.check(last_query)
        if not policy_result.allowed:
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=guardrail_result,
                semantic_analysis=semantic_result,
                linguistic_analysis=linguistic_result.model_dump(),
                candidate_objects=candidate_names,
                selected_objects=selection_result.selected_objects,
                relevant_schema=relevant_schema.model_dump(),
                query_plan=query_plan,
                generated_query=last_query.formatted_query,
                attempts=attempt,
                validation=last_validation,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )

        # 13. Stage 12: Database Execution
        try:
            columns, raw_rows = self.executor.execute(last_query)
        except Exception as exec_err:
            logger.error("Execution error: %s", exec_err)
            return PipelineResponse(
                database=db_type.value,
                question=question,
                guardrail=guardrail_result,
                semantic_analysis=semantic_result,
                linguistic_analysis=linguistic_result.model_dump(),
                candidate_objects=candidate_names,
                selected_objects=selection_result.selected_objects,
                relevant_schema=relevant_schema.model_dump(),
                query_plan=query_plan,
                generated_query=last_query.formatted_query,
                attempts=attempt,
                validation=last_validation,
                policy=policy_result,
                error=f"Database execution error: {exec_err}",
            )

        # 14. Stage 13: Result Processing
        execution_results = self.results.process(columns, raw_rows)

        # Return full trace response
        return PipelineResponse(
            database=db_type.value,
            question=question,
            guardrail=guardrail_result,
            semantic_analysis=semantic_result,
            linguistic_analysis=linguistic_result.model_dump(),
            candidate_objects=candidate_names,
            selected_objects=selection_result.selected_objects,
            relevant_schema=relevant_schema.model_dump(),
            query_plan=query_plan,
            generated_query=last_query.formatted_query,
            attempts=attempt,
            validation=last_validation,
            policy=policy_result,
            results=execution_results,
        )
