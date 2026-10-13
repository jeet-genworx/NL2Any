"""Tests for the human-in-the-loop gate on table selection."""

import time

import pytest
from fastapi.testclient import TestClient

from backend.src.api.rest.app import app
from backend.src.api.rest.dependencies import get_orchestrator
from backend.src.core.query_processing.pipeline.approval import (
    HALLUCINATED_TABLES,
    INSUFFICIENT_AFTER_RETRIES,
    SELECTOR_ERROR_FALLBACK,
    UNJOINABLE_DROPPED,
    PendingRun,
    PendingRunNotFound,
    PendingRunStore,
    assess_selection,
)
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.schemas.pipeline import (
    CandidateTable,
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    PolicyResult,
    QueryPlan,
    SemanticAnalysisResult,
    TableSelectionResult,
    ValidationErrorType,
    ValidationResult,
)


# --------------------------------------------------------------------------
# Stage doubles
# --------------------------------------------------------------------------


class StubEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 384


class StubRetriever:
    def __init__(self, candidates: list[CandidateTable]) -> None:
        self.candidates = candidates

    def retrieve(self, query_vector: list[float]) -> list[CandidateTable]:
        return list(self.candidates)


class StubGuardrail:
    async def classify(self, question: str) -> GuardrailResult:
        return GuardrailResult(decision=GuardrailDecision.READ_QUERY, reason="ok")

    def handle_basic_question(self, question: str, database_type: str, database_name: str) -> str:
        return "I am NL2AnyQuery."


class StubSemantic:
    async def analyze(self, question: str) -> SemanticAnalysisResult:
        return SemanticAnalysisResult(subjective=["customers"], objective=["all"])


class StubSelector:
    def __init__(self, results: list[TableSelectionResult]) -> None:
        self.results = results
        self.call_count = 0

    async def select(self, *args, **kwargs) -> TableSelectionResult:
        res = self.results[min(self.call_count, len(self.results) - 1)]
        self.call_count += 1
        return res


class RecordingPlanner:
    def __init__(self) -> None:
        self.seen_sources: list[list[str]] = []

    async def plan(self, *args, **kwargs) -> QueryPlan:
        schema = kwargs["relevant_schema"]
        self.seen_sources.append([obj.name for obj in schema.objects])
        return QueryPlan(sources=[obj.name for obj in schema.objects], projections=["id"])


class StubGenerator:
    async def generate_query(self, *args, **kwargs) -> GeneratedQuery:
        return GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query="SELECT id FROM customers;",
            formatted_query="SELECT id FROM customers;",
        )


class StubValidator:
    async def validate(self, *args, **kwargs) -> ValidationResult:
        return ValidationResult(valid=True, error_type=ValidationErrorType.VALID)


class StubPolicy:
    def check(self, *args, **kwargs) -> PolicyResult:
        return PolicyResult(allowed=True, reason="safe")


class StubExecutor:
    def execute(self, *args, **kwargs):
        return ["id"], [{"id": 1}]


class StubDescriptions:
    async def get_table_description(self, *args, **kwargs):
        return None


@pytest.fixture
def shop_schema() -> DatabaseSchema:
    """Two joinable tables plus one that nothing links to."""
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="test_shop",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="integer"), Field(name="name", type="varchar")],
            ),
            SchemaObject(
                name="orders",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="integer"), Field(name="customer_id", type="integer")],
            ),
            SchemaObject(
                name="audit_log",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="integer")],
            ),
        ],
        relationships=[
            Relationship(
                from_object="orders",
                from_field="customer_id",
                to_object="customers",
                to_field="id",
            )
        ],
    )


def build_orchestrator(schema: DatabaseSchema, selector: StubSelector, planner=None):
    orchestrator = NL2AnyQueryOrchestrator(
        embedding_provider=StubEmbedding(),
        vector_retriever=StubRetriever([
            CandidateTable(table_name="customers", similarity=0.9, rank=1)
        ]),
        guardrail=StubGuardrail(),
        semantic=StubSemantic(),
        selector=selector,
        planner=planner or RecordingPlanner(),
        postgres_gen=StubGenerator(),
        validator=StubValidator(),
        policy=StubPolicy(),
        executor=StubExecutor(),
        descriptions_client=StubDescriptions(),
        max_retries=1,
    )
    orchestrator._schemas[DatabaseType.POSTGRESQL] = schema
    orchestrator._schemas["postgres"] = schema
    return orchestrator


# --------------------------------------------------------------------------
# assess_selection: what counts as uncertain
# --------------------------------------------------------------------------


def test_a_clean_selection_raises_nothing():
    """The ordinary case has to stay silent, or the gate is just a nuisance."""
    selection = TableSelectionResult(selected_objects=["customers"], sufficient=True)

    assert assess_selection(selection, unjoinable=[], exhausted_retries=False) is None


def test_insufficiency_surviving_the_retries_is_uncertainty():
    selection = TableSelectionResult(
        selected_objects=["customers"],
        sufficient=False,
        missing_objects=["invoices"],
        reason="No table holds invoice totals.",
    )

    uncertainty = assess_selection(selection, unjoinable=[], exhausted_retries=True)

    assert uncertainty is not None
    assert uncertainty.reasons == [INSUFFICIENT_AFTER_RETRIES]
    assert "No table holds invoice totals." in uncertainty.detail
    assert "invoices" in uncertainty.detail
    assert uncertainty.missing_objects == ["invoices"]


def test_a_fallback_selection_is_uncertainty_even_though_it_claims_sufficiency():
    """The stage reports success on its error path; the flag is what gives it away."""
    selection = TableSelectionResult(
        selected_objects=["customers"],
        sufficient=True,
        reason="Fallback selection due to error: boom",
        fallback=True,
    )

    uncertainty = assess_selection(selection, unjoinable=[], exhausted_retries=False)

    assert uncertainty is not None
    assert uncertainty.reasons == [SELECTOR_ERROR_FALLBACK]
    assert uncertainty.fallback is True


def test_discarded_and_unjoinable_tables_are_both_reported():
    selection = TableSelectionResult(
        selected_objects=["customers"],
        sufficient=True,
        discarded_objects=["invoices"],
    )

    uncertainty = assess_selection(selection, unjoinable=["audit_log"], exhausted_retries=False)

    assert uncertainty is not None
    assert uncertainty.reasons == [HALLUCINATED_TABLES, UNJOINABLE_DROPPED]
    assert "invoices" in uncertainty.detail
    assert "audit_log" in uncertainty.detail


# --------------------------------------------------------------------------
# The selector now reports what it used to repair in silence
# --------------------------------------------------------------------------


async def test_selector_reports_hallucinated_names(monkeypatch):
    from backend.src.core.query_processing.pipeline import selector as selector_module
    from backend.src.schemas.pipeline import (
        NeighborhoodTable,
        QuestionAnalysis,
        TableNeighborhood,
    )

    class Provider:
        async def generate(self, *args, **kwargs) -> str:
            return '{"selected_objects": ["customers", "invoices"], "sufficient": true}'

    sel = selector_module.TableSelector(provider=Provider())
    result = await sel.select(
        question_analysis=QuestionAnalysis(question="show customers"),
        neighborhood=TableNeighborhood(tables=[NeighborhoodTable(name="customers")]),
    )

    assert result.selected_objects == ["customers"]
    assert result.discarded_objects == ["invoices"]
    assert result.fallback is False


async def test_selector_reports_its_fallback():
    from backend.src.core.query_processing.pipeline import selector as selector_module
    from backend.src.schemas.pipeline import (
        NeighborhoodTable,
        QuestionAnalysis,
        TableNeighborhood,
    )

    class BrokenProvider:
        async def generate(self, *args, **kwargs) -> str:
            raise RuntimeError("model is down")

    sel = selector_module.TableSelector(provider=BrokenProvider())
    result = await sel.select(
        question_analysis=QuestionAnalysis(question="show customers"),
        neighborhood=TableNeighborhood(tables=[NeighborhoodTable(name="customers")]),
    )

    assert result.fallback is True
    assert result.selected_objects == ["customers"]


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def _pending(**overrides) -> PendingRun:
    base = dict(
        target=None,
        db_type=DatabaseType.POSTGRESQL,
        provider_name="koboldcpp",
        provider_arg=None,
        question="q",
        processed_question="q",
        schema=None,
        guardrail_result=GuardrailResult(decision=GuardrailDecision.READ_QUERY),
        spelling_res=None,
        semantic_res=None,
        linguistic_res=None,
        analysis=None,
        outcome=None,
        uncertainty=None,
    )
    base.update(overrides)
    return PendingRun(**base)


def test_a_parked_run_is_resumable_exactly_once():
    store = PendingRunStore()
    run_id = store.put(_pending())

    assert store.pop(run_id) is not None
    with pytest.raises(PendingRunNotFound):
        store.pop(run_id)


def test_an_unanswered_run_expires():
    """A human who closes the tab never comes back, so the run must not wait forever."""
    store = PendingRunStore(ttl_seconds=60)
    run_id = store.put(_pending(created_at=time.time() - 120))

    with pytest.raises(PendingRunNotFound):
        store.pop(run_id)
    assert len(store) == 0


def test_unanswered_runs_cannot_pile_up_without_bound():
    store = PendingRunStore(max_entries=2)
    first = store.put(_pending(created_at=time.time() - 3))
    store.put(_pending(created_at=time.time() - 2))
    store.put(_pending(created_at=time.time() - 1))

    assert len(store) == 2
    with pytest.raises(PendingRunNotFound):
        store.pop(first)


# --------------------------------------------------------------------------
# The gate, end to end through the orchestrator
# --------------------------------------------------------------------------


async def test_a_confident_selection_never_stops(shop_schema):
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)]),
    )

    resp = await orchestrator.execute_pipeline(
        "Show customers", database="postgres", allow_table_approval=True
    )

    assert resp.status == "complete"
    assert resp.approval is None
    assert resp.results is not None
    assert len(orchestrator.pending_runs) == 0


async def test_an_uncertain_selection_proceeds_when_the_caller_cannot_ask(shop_schema):
    """Today's behaviour, which callers that cannot answer still depend on."""
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"],
                sufficient=False,
                reason="Not sure.",
            )
        ]),
    )

    resp = await orchestrator.execute_pipeline("Show customers", database="postgres")

    assert resp.status == "complete"
    assert resp.results is not None
    assert resp.selection_notice is not None
    assert len(orchestrator.pending_runs) == 0


async def test_the_kill_switch_turns_the_gate_off_for_every_caller(shop_schema, monkeypatch):
    from backend.src.core.query_processing.pipeline import orchestrator as orch_module

    monkeypatch.setattr(orch_module.settings, "table_approval_enabled", False)
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"], sufficient=False, reason="Not sure."
            )
        ]),
    )

    resp = await orchestrator.execute_pipeline(
        "Show customers", database="postgres", allow_table_approval=True
    )

    assert resp.status == "complete"
    assert resp.results is not None


async def test_an_uncertain_selection_stops_for_a_human(shop_schema):
    planner = RecordingPlanner()
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"],
                sufficient=False,
                reason="Not sure.",
            )
        ]),
        planner=planner,
    )

    resp = await orchestrator.execute_pipeline(
        "Show customers", database="postgres", allow_table_approval=True
    )

    assert resp.status == "awaiting_table_approval"
    assert resp.approval is not None
    assert resp.approval.proposed_objects == ["customers"]
    assert "customers" in resp.approval.selectable_objects
    assert INSUFFICIENT_AFTER_RETRIES in resp.approval.uncertainty.reasons
    # Nothing downstream ran: no plan, no query, no execution.
    assert resp.results is None
    assert resp.query_plan is None
    assert planner.seen_sources == []
    assert len(orchestrator.pending_runs) == 1


async def test_a_human_can_change_the_tables_and_the_run_finishes(shop_schema):
    planner = RecordingPlanner()
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"],
                sufficient=False,
                reason="Not sure.",
            )
        ]),
        planner=planner,
    )
    paused = await orchestrator.execute_pipeline(
        "Show customer orders", database="postgres", allow_table_approval=True
    )

    resp = await orchestrator.resume_pipeline(
        run_id=paused.approval.run_id, approved_objects=["customers", "orders"]
    )

    assert resp.status == "complete"
    assert resp.error is None
    assert resp.results is not None
    assert resp.table_approval is not None
    assert resp.table_approval.changed is True
    assert resp.table_approval.proposed_objects == ["customers"]
    assert set(resp.selected_objects) == {"customers", "orders"}
    # The human's choice is what the planner is given, not the selector's.
    assert planner.seen_sources and set(planner.seen_sources[0]) == {"customers", "orders"}
    # The join was derived from the schema, not from the human's say-so.
    assert [
        (r.from_object, r.to_object) for r in resp.selected_relationships
    ] == [("orders", "customers")]


async def test_approving_the_proposal_unchanged_is_recorded_as_unchanged(shop_schema):
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"], sufficient=False, reason="Not sure."
            )
        ]),
    )
    paused = await orchestrator.execute_pipeline(
        "Show customers", database="postgres", allow_table_approval=True
    )

    resp = await orchestrator.resume_pipeline(
        run_id=paused.approval.run_id, approved_objects=["customers"]
    )

    assert resp.table_approval.changed is False
    assert resp.selected_objects == ["customers"]


async def test_a_human_may_not_add_a_table_the_selector_was_never_shown(shop_schema):
    """A table outside the candidate set has no guarantee of joining to anything."""
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"], sufficient=False, reason="Not sure."
            )
        ]),
    )
    paused = await orchestrator.execute_pipeline(
        "Show customers", database="postgres", allow_table_approval=True
    )
    run_id = paused.approval.run_id

    with pytest.raises(ValueError, match="not in this run's candidate set"):
        await orchestrator.resume_pipeline(run_id=run_id, approved_objects=["spaceships"])

    # The rejection costs a typo, not the whole run.
    resumed = await orchestrator.resume_pipeline(run_id=run_id, approved_objects=["customers"])
    assert resumed.status == "complete"


async def test_the_planner_cannot_widen_a_human_approved_selection(shop_schema):
    """Re-selecting behind the human's back would defeat the point of asking."""

    class PlannerWantingMore:
        def __init__(self) -> None:
            self.seen_sources: list[list[str]] = []

        async def plan(self, *args, **kwargs) -> QueryPlan:
            schema = kwargs["relevant_schema"]
            self.seen_sources.append([obj.name for obj in schema.objects])
            return QueryPlan(
                sources=[obj.name for obj in schema.objects],
                projections=["id"],
                missing_tables=["orders"],
            )

    planner = PlannerWantingMore()
    selector = StubSelector([
        TableSelectionResult(selected_objects=["customers"], sufficient=False, reason="Not sure.")
    ])
    orchestrator = build_orchestrator(shop_schema, selector, planner=planner)
    paused = await orchestrator.execute_pipeline(
        "Show customers", database="postgres", allow_table_approval=True
    )
    calls_before = selector.call_count

    resp = await orchestrator.resume_pipeline(
        run_id=paused.approval.run_id, approved_objects=["customers"]
    )

    assert resp.selected_objects == ["customers"]
    assert selector.call_count == calls_before
    assert planner.seen_sources == [["customers"]]


async def test_resuming_an_unknown_run_is_an_error(shop_schema):
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([TableSelectionResult(selected_objects=["customers"], sufficient=True)]),
    )

    with pytest.raises(PendingRunNotFound):
        await orchestrator.resume_pipeline(run_id="nope", approved_objects=["customers"])


# --------------------------------------------------------------------------
# The two-call API contract
# --------------------------------------------------------------------------


@pytest.fixture
def api_client(shop_schema):
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"], sufficient=False, reason="Not sure."
            )
        ]),
    )
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    yield TestClient(app), orchestrator
    app.dependency_overrides.pop(get_orchestrator, None)


def test_query_endpoint_defaults_to_running_straight_through(api_client):
    client, _ = api_client

    data = client.post(
        "/query", json={"database": "postgres", "question": "Show customers"}
    ).json()

    assert data["status"] == "complete"
    assert data["approval"] is None
    assert data["selection_notice"] is not None


def test_query_endpoint_pauses_and_resumes(api_client):
    client, _ = api_client

    paused = client.post(
        "/query",
        json={
            "database": "postgres",
            "question": "Show customer orders",
            "allow_table_approval": True,
        },
    ).json()
    assert paused["status"] == "awaiting_table_approval"
    run_id = paused["approval"]["run_id"]

    resumed = client.post(
        f"/query/{run_id}/tables", json={"approved_objects": ["customers", "orders"]}
    )

    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "complete"
    assert body["results"]["row_count"] == 1
    assert set(body["selected_objects"]) == {"customers", "orders"}
    assert body["table_approval"]["changed"] is True

    # The run is spent; answering it twice is not a way to re-run a query.
    assert client.post(
        f"/query/{run_id}/tables", json={"approved_objects": ["customers"]}
    ).status_code == 404


def test_resuming_an_expired_or_unknown_run_is_a_404(api_client):
    client, _ = api_client

    response = client.post("/query/deadbeef/tables", json={"approved_objects": ["customers"]})

    assert response.status_code == 404
    assert "deadbeef" in response.json()["detail"]


def test_approving_a_table_outside_the_candidate_set_is_a_400(api_client):
    client, _ = api_client
    paused = client.post(
        "/query",
        json={
            "database": "postgres",
            "question": "Show customers",
            "allow_table_approval": True,
        },
    ).json()

    response = client.post(
        f"/query/{paused['approval']['run_id']}/tables",
        json={"approved_objects": ["spaceships"]},
    )

    assert response.status_code == 400
    assert "spaceships" in response.json()["detail"]


def test_approving_nothing_is_rejected(api_client):
    client, _ = api_client
    paused = client.post(
        "/query",
        json={
            "database": "postgres",
            "question": "Show customers",
            "allow_table_approval": True,
        },
    ).json()

    response = client.post(
        f"/query/{paused['approval']['run_id']}/tables",
        json={"approved_objects": []},
    )

    assert response.status_code == 422


async def test_a_paused_run_resumes_on_the_providers_it_started_with(shop_schema):
    """The gate must not quietly move a run onto a different model.

    Everything after the gate is the generation loop, so a resume that fell
    back to the default provider would answer on a model the request never
    asked for -- and the response would still claim the split it was started
    with.
    """
    orchestrator = build_orchestrator(
        shop_schema,
        StubSelector([
            TableSelectionResult(
                selected_objects=["customers"], sufficient=False, reason="Not sure."
            )
        ]),
    )
    paused = await orchestrator.execute_pipeline(
        "Show customers",
        database="postgres",
        provider="koboldcpp",
        selector_provider="slm",
        allow_table_approval=True,
    )
    assert paused.status == "awaiting_table_approval"

    run = orchestrator.pending_runs.get(paused.approval.run_id)
    assert run.selector_provider_arg == "slm"
    assert run.generator_provider_arg is None
    assert run.provider_map == {
        "default": "koboldcpp",
        "selector": "slm",
        "generator": "koboldcpp",
    }

    resp = await orchestrator.resume_pipeline(
        run_id=paused.approval.run_id, approved_objects=["customers"]
    )
    assert resp.status == "complete"
    assert resp.providers == run.provider_map
