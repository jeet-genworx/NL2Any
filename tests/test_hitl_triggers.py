"""Every way the table-approval gate can fire, and the ways it must not.

The gate needs three things at once: the caller willing to be asked
(`allow_table_approval`), the deployment-wide switch on, and the selection
carrying uncertainty. The four reason codes below are the only sources of that
uncertainty, and each gets a case here.
"""

import pytest
from fastapi.testclient import TestClient

from backend.src.api.rest.app import app
from backend.src.api.rest.dependencies import get_orchestrator
from backend.src.core.query_processing.pipeline.approval import (
    HALLUCINATED_TABLES,
    INSUFFICIENT_AFTER_RETRIES,
    SELECTOR_ERROR_FALLBACK,
    UNJOINABLE_DROPPED,
)
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import CandidateTable, TableSelectionResult
from tests.test_table_approval import StubRetriever, build_orchestrator, shop_schema  # noqa: F401


def _client(schema, selection, *, candidates=("customers",)):
    """An API client whose selector returns `selection` for this one question."""
    from tests.test_table_approval import StubSelector

    orch = build_orchestrator(schema, StubSelector([selection]))
    orch.vector_retriever = StubRetriever(
        [CandidateTable(table_name=n, similarity=0.9 - i / 10, rank=i + 1)
         for i, n in enumerate(candidates)]
    )
    orch._vector_retrievers.clear()
    app.dependency_overrides[get_orchestrator] = lambda: orch
    return TestClient(app), orch


def _ask(client, **extra):
    body = {"database": "postgres", "question": "Show customers", **extra}
    return client.post("/query", json=body).json()


@pytest.fixture(autouse=True)
def _clean_overrides():
    yield
    app.dependency_overrides.pop(get_orchestrator, None)


# ---------------------------------------------------------------- triggers


def test_trigger_selector_never_became_confident(shop_schema):
    """1. The selector reports `sufficient=False` on every attempt.

    The commonest trigger: the question needs something the schema does not
    obviously cover, so the model keeps saying it is not sure.
    """
    client, _ = _client(
        shop_schema,
        TableSelectionResult(
            selected_objects=["customers"], sufficient=False, reason="No revenue column anywhere.",
            missing_objects=["payments"],
        ),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "awaiting_table_approval"
    assert data["approval"]["uncertainty"]["reasons"] == [INSUFFICIENT_AFTER_RETRIES]
    assert data["approval"]["uncertainty"]["missing_objects"] == ["payments"]


def test_trigger_selector_call_failed_and_fell_back(shop_schema):
    """2. The selector stage failed and the top embedding match stood in.

    `fallback=True` means nothing actually read the question against the
    schema -- the SLM call or its JSON broke -- so the tables are a guess even
    though the result claims sufficiency.
    """
    client, _ = _client(
        shop_schema,
        TableSelectionResult(selected_objects=["customers"], sufficient=True, fallback=True),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "awaiting_table_approval"
    assert data["approval"]["uncertainty"]["reasons"] == [SELECTOR_ERROR_FALLBACK]
    assert data["approval"]["uncertainty"]["fallback"] is True


def test_trigger_selector_invented_a_table(shop_schema):
    """3. The model named tables that are not in the candidate set.

    A model that hallucinates one name has misread the schema it was shown, so
    the names it got right are not trustworthy either.
    """
    # Mirrors what the real selector returns: invented names are filtered out of
    # `selected_objects` and recorded on `discarded_objects` (selector.py:296-307).
    client, _ = _client(
        shop_schema,
        TableSelectionResult(
            selected_objects=["customers"],
            discarded_objects=["invoices", "subscriptions"],
            sufficient=True,
        ),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "awaiting_table_approval"
    assert data["approval"]["uncertainty"]["reasons"] == [HALLUCINATED_TABLES]
    assert sorted(data["approval"]["uncertainty"]["discarded_objects"]) == [
        "invoices", "subscriptions"
    ]
    # The invented names never reach the query.
    assert data["approval"]["proposed_objects"] == ["customers"]


def test_trigger_chosen_table_cannot_be_joined(shop_schema):
    """4. A chosen table has no foreign-key path to the rest of the query.

    `audit_log` is a real table and a real candidate here, but nothing links it
    to `customers`, so any join would have to be invented. It is dropped, and
    that drop silently changes the question being answered.
    """
    client, _ = _client(
        shop_schema,
        TableSelectionResult(selected_objects=["customers", "audit_log"], sufficient=True),
        candidates=("customers", "audit_log"),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "awaiting_table_approval"
    assert data["approval"]["uncertainty"]["reasons"] == [UNJOINABLE_DROPPED]
    assert data["approval"]["uncertainty"]["unjoinable_objects"] == ["audit_log"]


def test_trigger_several_signals_at_once(shop_schema):
    """Reasons accumulate; the gate reports every signal it saw."""
    client, _ = _client(
        shop_schema,
        TableSelectionResult(
            selected_objects=["customers", "audit_log"],
            discarded_objects=["invoices"],
            sufficient=False, reason="Unsure.", fallback=True,
        ),
        candidates=("customers", "audit_log"),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "awaiting_table_approval"
    assert set(data["approval"]["uncertainty"]["reasons"]) == {
        INSUFFICIENT_AFTER_RETRIES, SELECTOR_ERROR_FALLBACK,
        HALLUCINATED_TABLES, UNJOINABLE_DROPPED,
    }


# ------------------------------------------------------------ no trigger


def test_a_confident_joinable_selection_does_not_stop(shop_schema):
    """The ordinary case has to stay off the gate, or it stops being useful."""
    client, _ = _client(
        shop_schema, TableSelectionResult(selected_objects=["customers"], sufficient=True)
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "complete"
    assert data["approval"] is None
    assert data["selection_notice"] is None


def test_a_caller_that_cannot_be_asked_proceeds_with_a_notice(shop_schema):
    """Same uncertainty, no `allow_table_approval`: carry on and say so."""
    client, _ = _client(
        shop_schema,
        TableSelectionResult(selected_objects=["customers"], sufficient=False, reason="Unsure."),
    )
    data = _ask(client)

    assert data["status"] == "complete"
    assert data["approval"] is None
    assert "never became confident" in data["selection_notice"]


def test_the_deployment_switch_overrides_the_request(shop_schema, monkeypatch):
    """TABLE_APPROVAL_ENABLED=false turns the gate off for every caller."""
    from backend.src.config import settings

    monkeypatch.setattr(settings, "table_approval_enabled", False)
    client, _ = _client(
        shop_schema,
        TableSelectionResult(selected_objects=["customers"], sufficient=False, reason="Unsure."),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "complete"
    assert data["selection_notice"] is not None


def test_nothing_selectable_is_an_error_not_a_gate(shop_schema):
    """With no tables left there is nothing for a human to approve.

    The selector discarded every name it produced, so the run ends rather than
    asking someone to choose from an empty list.
    """
    client, _ = _client(
        shop_schema,
        TableSelectionResult(
            selected_objects=[], discarded_objects=["nonexistent"],
            sufficient=False, reason="Lost.",
        ),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "complete"
    assert data["approval"] is None
    assert "could not" in (data["error"] or "").lower()


def test_a_single_invented_name_is_not_caught_downstream(shop_schema):
    """Documents a defence-in-depth gap rather than a reachable bug.

    `connect_selection` short-circuits for a selection of fewer than two tables
    (join_graph.py:354-355), so it never checks that one lone name exists. The
    real selector filters invented names before this point, so nothing reaches
    it today -- but a second producer of a selection would not be checked here.
    """
    client, _ = _client(
        shop_schema,
        TableSelectionResult(selected_objects=["nonexistent"], sufficient=False, reason="Lost."),
    )
    data = _ask(client, allow_table_approval=True)

    assert data["status"] == "awaiting_table_approval"
    assert data["approval"]["proposed_objects"] == ["nonexistent"]
