"""End-to-end pipeline orchestrator for NL2AnyQuery.

This module is the sequence and nothing else: which stage runs when, where a
run can stop, and what it answers with. The stages themselves live in their own
modules, as does everything the orchestrator used to do inline --
`resources` for the cached schema and embeddings, `stages` for provider
binding, `understanding` for the stages that only read the question,
`selection` for retrieval through table choice, then `materialization`,
`planning` and `generation` for the loops, and `response` for the one shape
every exit returns.
"""

import logging
from typing import Any

from backend.src.config import settings
from backend.src.control.providers.embedding.base import EmbeddingProvider
from backend.src.control.providers.embedding.koboldcpp import KoboldCppEmbeddingProvider
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.core.query_processing.nlp.linguistic import LinguisticAnalyzer
from backend.src.core.query_processing.pipeline.approval import (
    PendingRun,
    PendingRunStore,
    normalize_approved_objects,
)
from backend.src.core.query_processing.pipeline.descriptions_client import DescriptionsClient
from backend.src.core.query_processing.pipeline.executor import QueryExecutor
from backend.src.core.query_processing.pipeline.expansion import SchemaExpander
from backend.src.core.query_processing.pipeline.generation import plan_generate_validate
from backend.src.core.query_processing.pipeline.generators.mongo import MongoQueryGenerator
from backend.src.core.query_processing.pipeline.generators.postgres import PostgresQueryGenerator
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier
from backend.src.core.query_processing.pipeline.materialization import materialize_schema
from backend.src.core.query_processing.pipeline.planner import QueryPlanner
from backend.src.core.query_processing.pipeline.planning import plan_with_table_feedback
from backend.src.core.query_processing.pipeline.policy import SafetyPolicyValidator
from backend.src.core.query_processing.pipeline.resources import (
    SchemaResources,
    resolve_database_target,
)
from backend.src.core.query_processing.pipeline.response import build_response
from backend.src.core.query_processing.pipeline.results import ResultProcessor
from backend.src.core.query_processing.pipeline.selection import (
    SelectionOutcome,
    resolve_selection,
    retrieve_and_select,
)
from backend.src.core.query_processing.pipeline.selector import TableSelector
from backend.src.core.query_processing.pipeline.semantic import SemanticAnalyzer
from backend.src.core.query_processing.pipeline.spelling import SpellingChecker
from backend.src.core.query_processing.pipeline.stages import (
    StageBinder,
    StageSet,
    plan_providers,
    resolve_provider,
)
from backend.src.core.query_processing.pipeline.understanding import understand_question
from backend.src.core.query_processing.pipeline.validator import QueryValidator
from backend.src.core.query_processing.retrieval.store import EmbeddingStore
from backend.src.core.query_processing.retrieval.vector import VectorRetriever
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.models.targets import DatabaseTarget
from backend.src.schemas.pipeline import (
    GuardrailDecision,
    GuardrailResult,
    PipelineResponse,
    QuestionAnalysis,
    SemanticAnalysisResult,
    SpellingCorrectionResult,
    TableApprovalRecord,
    TableApprovalRequest,
)

logger = logging.getLogger(__name__)

# Kept importable from here: this is where the type lived before the selection
# stage moved into its own module.
_SelectionOutcome = SelectionOutcome

__all__ = ["NL2AnyQueryOrchestrator", "SelectionOutcome"]


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
        self.descriptions_client = descriptions_client or DescriptionsClient()
        self.planner = planner or QueryPlanner(provider=self.provider)
        self.postgres_gen = postgres_gen or PostgresQueryGenerator(provider=self.provider)
        self.mongo_gen = mongo_gen or MongoQueryGenerator(provider=self.provider)
        self.validator = validator or QueryValidator(provider=self.provider)
        self.policy = policy or SafetyPolicyValidator()
        self.executor = executor or QueryExecutor()
        self.results = results or ResultProcessor()
        self.max_retries = max_retries if max_retries is not None else settings.max_retries
        # Runs parked at the table-approval gate, waiting on a human. Held here
        # because the API keeps one orchestrator per process, so the resume call
        # finds the run the first call parked.
        self.pending_runs = PendingRunStore()

        self._resources = SchemaResources(
            embedding_store=embedding_store,
            vector_retriever=vector_retriever,
        )
        self._stage_binder = StageBinder(
            provider=self.provider,
            defaults=StageSet(
                guardrail=self.guardrail,
                semantic=self.semantic,
                selector=self.selector,
                planner=self.planner,
                postgres_gen=self.postgres_gen,
                mongo_gen=self.mongo_gen,
                validator=self.validator,
            ),
            injected={
                "guardrail": guardrail is not None,
                "semantic": semantic is not None,
                "selector": selector is not None,
                "planner": planner is not None,
                "postgres_gen": postgres_gen is not None,
                "mongo_gen": mongo_gen is not None,
                "validator": validator is not None,
            },
        )

    # -- cached per-database artifacts -------------------------------------

    @property
    def _schemas(self) -> dict[Any, DatabaseSchema]:
        return self._resources.schemas

    @property
    def _vector_retrievers(self) -> dict[Any, VectorRetriever]:
        return self._resources.vector_retrievers

    @property
    def _join_graphs(self) -> dict[int, tuple[DatabaseSchema, Any]]:
        return self._resources.join_graphs

    def invalidate_caches(self) -> None:
        """Drop every cached schema, embedding set and join graph."""
        self._resources.invalidate()

    def get_or_load_schema(self, database: DatabaseTarget | DatabaseType | str) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file."""
        return self._resources.schema(database)

    # -- provider and stage binding ----------------------------------------

    def _get_stages_for_provider(
        self,
        active_provider: ModelProvider,
        selector_provider: ModelProvider | None = None,
        generator_provider: ModelProvider | None = None,
    ) -> StageSet:
        """Stage instances, each bound to the provider chosen for its stage."""
        return self._stage_binder.bind(
            active_provider,
            selector_provider=selector_provider,
            generator_provider=generator_provider,
        )

    def _resolve_provider(
        self,
        provider: "str | ModelProvider | None",
        default_name: str = "koboldcpp",
    ) -> tuple[str, ModelProvider] | tuple[None, None]:
        """Turn a request's provider argument into (name, instance)."""
        return resolve_provider(provider)

    # -- the run -----------------------------------------------------------

    async def execute_pipeline(
        self,
        question: str,
        database: str | DatabaseType | DatabaseTarget = "postgres",
        provider: str | ModelProvider | None = None,
        jargons: list[str] | None = None,
        allow_table_approval: bool = False,
        selector_provider: str | ModelProvider | None = None,
        generator_provider: str | ModelProvider | None = None,
    ) -> PipelineResponse:
        """Run complete NL-to-query pipeline with clean stages and bounded retries.

        `provider` is the default for every stage. `selector_provider` and
        `generator_provider` override it for table selection and for the
        planning/generation/validation loop, so the two halves of the pipeline
        can run on different models -- a local SLM for one and a hosted model
        for the other, say. Left unset, they follow `provider` and the run
        behaves exactly as a single-provider run always did.

        With `allow_table_approval`, a run whose table selection carries
        uncertainty stops at the gate instead of proceeding: the response comes
        back with `status="awaiting_table_approval"` and a run id, and
        `resume_pipeline` finishes it once a human has settled the tables. The
        caller has to ask for that, because a caller that cannot answer the
        question is better served by today's behaviour -- carry on, and say so
        on `selection_notice`.
        """
        providers = plan_providers(
            self.provider,
            provider=provider,
            selector_provider=selector_provider,
            generator_provider=generator_provider,
        )
        stages = self._get_stages_for_provider(
            providers.default,
            selector_provider=providers.selector,
            generator_provider=providers.generator,
        )

        # 1. Resolve database target & type
        target, db_type = resolve_database_target(database)

        def respond(**kwargs) -> PipelineResponse:
            return build_response(
                db_type=db_type,
                target_name=target.key,
                question=question,
                provider=providers.name,
                providers=providers.provider_map,
                **kwargs,
            )

        try:
            schema = self.get_or_load_schema(target)
        except Exception as err:
            return respond(
                guardrail=GuardrailResult(decision=GuardrailDecision.REJECT, reason=str(err)),
                error=str(err),
            )

        # 2. Stages 1 to 3: guardrail, spelling, then the concurrent embedding,
        #    semantic and linguistic reads of the question. Either of the first
        #    two can end the run on its own.
        understood = await understand_question(
            question=question,
            target_key=target.key,
            database_name=schema.database_name,
            guardrail=stages.guardrail,
            semantic=stages.semantic,
            linguistic=self.linguistic,
            spelling_checker=self.spelling_checker,
            embedding_provider=self.embedding_provider,
            jargons=jargons,
        )
        if understood.is_terminal:
            return respond(**understood.terminal_fields())
        analysis = understood.analysis
        processed_question = understood.processed_question

        # 3. Stages 5, 6 & 7: candidate retrieval -> BFS neighborhood -> table selection
        outcome = await retrieve_and_select(
            question=processed_question,
            analysis=analysis,
            query_vector=understood.query_vector,
            schema=schema,
            graph=self._resources.join_graph(schema),
            selector=stages.selector,
            retriever=self._resources.vector_retriever(target),
            embedding_provider=self.embedding_provider,
            max_retries=self.max_retries,
        )
        if outcome.notice:
            logger.warning("Proceeding despite selector doubt: %s", outcome.notice)

        selection_fields = {
            **understood.response_fields(),
            "candidates": outcome.candidates,
            "selected": outcome.selected_objects,
            "selected_relationships": outcome.relationships,
            "connector_objects": outcome.connector_objects,
            "unjoinable_objects": outcome.unjoinable_objects,
            "selection_notice": outcome.notice,
            "table_neighborhood": outcome.neighborhood,
            "selection_retries": outcome.retries,
        }

        if outcome.error or not outcome.selected_objects:
            return respond(
                **selection_fields,
                error=outcome.error
                or "I could not find any relevant tables or collections in the database schema matching your question.",
            )

        # 4. The table gate. Everything after this point is scoped by the tables
        #    chosen above, so a wrong choice does not fail, it answers a
        #    different question. When the selection carries uncertainty and the
        #    caller can ask a human, the run stops here instead.
        if (
            allow_table_approval
            and settings.table_approval_enabled
            and outcome.uncertainty is not None
        ):
            pending = PendingRun(
                target=target,
                db_type=db_type,
                provider_name=providers.name,
                provider_arg=provider,
                provider_map=dict(providers.provider_map),
                selector_provider_arg=selector_provider,
                generator_provider_arg=generator_provider,
                question=question,
                processed_question=processed_question,
                schema=schema,
                guardrail_result=understood.guardrail,
                spelling_res=understood.spelling,
                semantic_res=understood.semantic,
                linguistic_res=understood.linguistic,
                analysis=analysis,
                outcome=outcome,
                uncertainty=outcome.uncertainty,
            )
            run_id = self.pending_runs.put(pending)
            logger.info(
                "Pausing for table approval (run %s): %s", run_id, outcome.uncertainty.reasons
            )
            return respond(
                **selection_fields,
                status="awaiting_table_approval",
                approval=TableApprovalRequest(
                    run_id=run_id,
                    proposed_objects=outcome.selected_objects,
                    selectable_objects=(
                        outcome.neighborhood.table_names() if outcome.neighborhood else []
                    ),
                    uncertainty=outcome.uncertainty,
                    expires_at=pending.expires_at(),
                ),
            )

        return await self._finish_pipeline(
            target=target,
            db_type=db_type,
            provider_name=providers.name,
            provider_map=providers.provider_map,
            question=question,
            processed_question=processed_question,
            schema=schema,
            analysis=analysis,
            outcome=outcome,
            guardrail_result=understood.guardrail,
            spelling_res=understood.spelling,
            semantic_res=understood.semantic,
            linguistic_res=understood.linguistic,
            selector=stages.selector,
            planner=stages.planner,
            generator=stages.generator_for(db_type),
            validator=stages.validator,
        )

    async def resume_pipeline(
        self,
        run_id: str,
        approved_objects: list[str],
    ) -> PipelineResponse:
        """Finish a run a human stopped at the table gate.

        The approved names are checked against the neighborhood and then put
        through the same deterministic repair the model's answer goes through,
        so a hand-picked set gets its bridge tables and loses its unreachable
        ones exactly as the automatic path would. A human chooses the query's
        subject; whether two tables join is still a fact about the schema.
        """
        run = self.pending_runs.get(run_id)
        outcome = run.outcome
        chosen = normalize_approved_objects(approved_objects, outcome.neighborhood)

        resolved, relationships, added, unjoinable = resolve_selection(
            chosen,
            outcome.neighborhood,
            run.schema,
            self._resources.join_graph(run.schema),
        )
        if not resolved:
            raise ValueError("The approved tables could not be resolved into a joinable set.")

        # Answered, so the run is spent: it may not be resumed twice.
        self.pending_runs.pop(run_id)
        proposed = list(outcome.selected_objects)
        outcome.selected_objects = resolved
        outcome.relationships = relationships
        outcome.connector_objects = added
        outcome.unjoinable_objects = unjoinable
        logger.info(
            "Resuming run %s with human-approved tables %s (proposed %s).",
            run_id,
            resolved,
            proposed,
        )

        # Re-resolved from the arguments the first call was given, so a run
        # cannot finish against a provider the request never asked for.
        providers = plan_providers(
            self.provider,
            provider=run.provider_arg,
            selector_provider=run.selector_provider_arg,
            generator_provider=run.generator_provider_arg,
        )
        stages = self._get_stages_for_provider(
            providers.default,
            selector_provider=providers.selector,
            generator_provider=providers.generator,
        )

        return await self._finish_pipeline(
            target=run.target,
            db_type=run.db_type,
            provider_name=run.provider_name,
            provider_map=run.provider_map or None,
            question=run.question,
            processed_question=run.processed_question,
            schema=run.schema,
            analysis=run.analysis,
            outcome=outcome,
            guardrail_result=run.guardrail_result,
            spelling_res=run.spelling_res,
            semantic_res=run.semantic_res,
            linguistic_res=run.linguistic_res,
            selector=stages.selector,
            planner=stages.planner,
            generator=stages.generator_for(run.db_type),
            validator=stages.validator,
            approval_record=TableApprovalRecord(
                proposed_objects=proposed,
                approved_objects=resolved,
                changed=set(proposed) != set(resolved),
                uncertainty=run.uncertainty,
            ),
        )

    async def _finish_pipeline(
        self,
        *,
        target: DatabaseTarget,
        db_type: DatabaseType,
        provider_name: str,
        question: str,
        processed_question: str,
        schema: DatabaseSchema,
        analysis: QuestionAnalysis,
        outcome: SelectionOutcome,
        guardrail_result: GuardrailResult,
        spelling_res: SpellingCorrectionResult | None,
        semantic_res: SemanticAnalysisResult | None,
        linguistic_res: Any,
        selector: TableSelector,
        planner: QueryPlanner,
        generator: Any,
        validator: QueryValidator,
        provider_map: dict[str, str] | None = None,
        approval_record: TableApprovalRecord | None = None,
    ) -> PipelineResponse:
        """Everything downstream of a settled table selection.

        Split out so the run can be interrupted at the gate and picked up again
        from exactly here, whether the tables were settled by the selector or by
        a human.
        """
        graph = self._resources.join_graph(schema)

        async def materialize(selected: list[str], relationships: list[Any]):
            return await materialize_schema(
                selected_objects=selected,
                relationships=relationships,
                schema=schema,
                database=target,
                expander=self.expander,
                descriptions_client=self.descriptions_client,
            )

        # 1. Deterministic materialization of the selected tables and their
        #    relationships, plus description enrichment via the Descriptions API.
        relevant_schema = await materialize(outcome.selected_objects, outcome.relationships)

        # 2. Stage 8: query planning, with a feedback loop back into table
        #    selection when the plan needs a table the selector left out.
        planned = await plan_with_table_feedback(
            question=processed_question,
            analysis=analysis,
            relevant_schema=relevant_schema,
            schema=schema,
            neighborhood=outcome.neighborhood,
            graph=graph,
            db_type=db_type,
            planner=planner,
            selector=selector,
            materialize=materialize,
            max_retries=self.max_retries,
            selected_objects=outcome.selected_objects,
            relationships=outcome.relationships,
            connector_objects=outcome.connector_objects,
            unjoinable_objects=outcome.unjoinable_objects,
            tables_are_human_approved=approval_record is not None,
        )

        # 3. Stages 9 & 10: query generation and validation loop
        generated = await plan_generate_validate(
            question=processed_question,
            analysis=analysis,
            relevant_schema=planned.relevant_schema,
            db_type=db_type,
            planner=planner,
            generator=generator,
            validator=validator,
            max_retries=self.max_retries,
            initial_plan=planned.plan,
        )

        ctx: dict[str, Any] = {
            "db_type": db_type,
            "target_name": target.key,
            "question": question,
            "provider": provider_name,
            "providers": provider_map,
            "guardrail": guardrail_result,
            "spelling_correction": spelling_res,
            "semantic": semantic_res,
            "linguistic": linguistic_res.model_dump(),
            "candidates": outcome.candidates,
            "selected": planned.selected_objects,
            "selected_relationships": planned.relationships,
            "connector_objects": planned.connector_objects,
            "unjoinable_objects": planned.unjoinable_objects,
            "selection_notice": outcome.notice,
            "table_neighborhood": outcome.neighborhood,
            "table_approval": approval_record,
            "selection_retries": outcome.retries + planned.extra_selection_retries,
            "schema": planned.relevant_schema,
            "plan": generated.plan,
            "query": generated.query,
            "attempts": generated.retries + 1,
            "validation_retries": generated.retries,
            "validation": generated.validation,
        }

        if generated.error or not generated.query or not generated.validation or not generated.validation.valid:
            return build_response(**ctx, error=generated.error)

        # 4. Stage 11: deterministic safety policy check
        policy_result = self.policy.check(generated.query, schema=planned.relevant_schema)
        if not policy_result.allowed:
            return build_response(
                **ctx,
                policy=policy_result,
                error=f"Policy rejected query: {policy_result.reason}",
            )

        # 5. Stages 12 & 13: execution and result processing
        try:
            columns, raw_rows = self.executor.execute(
                generated.query,
                connection=target.connection,
                database=target.mongo_database,
            )
        except Exception as exec_err:
            logger.error("Execution error: %s", exec_err)
            return build_response(
                **ctx,
                policy=policy_result,
                error=f"Database execution error: {exec_err}",
            )

        return build_response(
            **ctx,
            policy=policy_result,
            results=self.results.process(columns, raw_rows),
        )
