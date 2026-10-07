"""Tests for the JoinPathSelector stage."""

import json
import pytest

from backend.src.schemas.pipeline import QuestionAnalysis
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.core.query_processing.pipeline.path_selector import JoinPathSelector


class MockPathProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""
        self.calls = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        raise RuntimeError("KoboldCpp unreachable")


def _table(name: str, *columns: str, description: str = "") -> SchemaObject:
    return SchemaObject(
        name=name,
        kind=SchemaObjectKind.TABLE,
        description=description,
        fields=[Field(name=c, type="int") for c in columns],
    )


def _analysis(question: str) -> QuestionAnalysis:
    return QuestionAnalysis(question=question, subjective=["customers"], objective=["total"])


@pytest.fixture
def transitive_schema() -> DatabaseSchema:
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[
            _table("customers", "id", "city"),
            _table("orders", "id", "customer_id", description="Customer purchase orders"),
            _table("order_items", "id", "order_id", "product_id", description="Line items per order"),
            _table("products", "id", "name"),
        ],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="order_items", from_field="order_id", to_object="orders", to_field="id"),
            Relationship(from_object="order_items", from_field="product_id", to_object="products", to_field="id"),
        ],
    )


@pytest.fixture
def ambiguous_schema() -> DatabaseSchema:
    """customers reaches payments via orders OR via subscriptions."""
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="billing",
        objects=[
            _table("customers", "id"),
            _table("orders", "id", "customer_id", description="One-off purchase orders"),
            _table("subscriptions", "id", "customer_id", description="Recurring billing subscriptions"),
            _table("payments", "id", "order_id", "subscription_id"),
        ],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="subscriptions", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="payments", from_field="order_id", to_object="orders", to_field="id"),
            Relationship(from_object="payments", from_field="subscription_id", to_object="subscriptions", to_field="id"),
        ],
    )


@pytest.mark.asyncio
async def test_single_selected_object_skips_the_model(transitive_schema):
    provider = MockPathProvider("")
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("How many customers?"),
        selected_objects=["customers"],
        schema=transitive_schema,
    )

    assert provider.calls == 0
    assert resolution.slm_invoked is False
    assert resolution.resolved_objects == ["customers"]


@pytest.mark.asyncio
async def test_transitive_path_adds_connector_tables(transitive_schema):
    provider = MockPathProvider(json.dumps({"chosen_paths": ["p1"], "reason": "only route"}))
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Which products do customers in Chennai buy?"),
        selected_objects=["customers", "products"],
        schema=transitive_schema,
    )

    assert resolution.slm_invoked is True
    assert resolution.connector_objects == ["order_items", "orders"]
    assert set(resolution.resolved_objects) == {"customers", "products", "orders", "order_items"}
    # Selected objects keep their original order ahead of the connectors.
    assert resolution.resolved_objects[:2] == ["customers", "products"]
    assert resolution.dropped_objects == []
    assert resolution.unjoinable_objects == []


@pytest.mark.asyncio
async def test_model_picks_between_competing_paths(ambiguous_schema):
    candidates_seen = {}

    provider = MockPathProvider(json.dumps({"chosen_paths": ["p2"], "reason": "recurring billing"}))
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Total recurring subscription payments per customer"),
        selected_objects=["customers", "payments"],
        schema=ambiguous_schema,
    )

    # Both routes were offered, with the descriptions that distinguish them.
    assert "orders" in provider.last_prompt
    assert "subscriptions" in provider.last_prompt
    assert "Recurring billing subscriptions" in provider.last_prompt
    assert resolution.chosen_path_ids == ["p2"]
    assert "subscriptions" in resolution.resolved_objects
    assert "orders" not in resolution.resolved_objects


@pytest.mark.asyncio
async def test_hallucinated_path_ids_are_discarded(transitive_schema):
    provider = MockPathProvider(
        json.dumps({"chosen_paths": ["p1", "p99", "customers->products"], "reason": "x"})
    )
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Which products do customers buy?"),
        selected_objects=["customers", "products"],
        schema=transitive_schema,
    )

    assert resolution.chosen_path_ids == ["p1"]
    assert resolution.fallback_used is False


@pytest.mark.asyncio
async def test_all_invalid_ids_fall_back_to_shortest_paths(ambiguous_schema):
    provider = MockPathProvider(json.dumps({"chosen_paths": ["nope"], "reason": "x"}))
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("payments per customer"),
        selected_objects=["customers", "payments"],
        schema=ambiguous_schema,
    )

    assert resolution.fallback_used is True
    # Shortest path per endpoint pair: one of the two 2-hop routes, not both.
    assert len(resolution.chosen_path_ids) == 1
    assert "customers" in resolution.resolved_objects
    assert "payments" in resolution.resolved_objects


@pytest.mark.asyncio
async def test_provider_failure_falls_back_without_raising(transitive_schema):
    provider = FailingProvider()
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Which products do customers buy?"),
        selected_objects=["customers", "products"],
        schema=transitive_schema,
    )

    assert provider.calls == 1
    assert resolution.fallback_used is True
    assert set(resolution.resolved_objects) == {"customers", "products", "orders", "order_items"}


@pytest.mark.asyncio
async def test_standalone_object_is_offered_and_can_be_dropped():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[
            _table("customers", "id"),
            _table("orders", "id", "customer_id"),
            _table("audit_log", "id", description="Administrative action log"),
        ],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")
        ],
    )
    # p1 = customers->orders, p2 = audit_log standalone. The model keeps only p1.
    provider = MockPathProvider(json.dumps({"chosen_paths": ["p1"], "reason": "audit log irrelevant"}))
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("How many orders per customer?"),
        selected_objects=["customers", "orders", "audit_log"],
        schema=schema,
    )

    assert "STANDALONE" in provider.last_prompt
    assert resolution.dropped_objects == ["audit_log"]
    assert set(resolution.resolved_objects) == {"customers", "orders"}


@pytest.mark.asyncio
async def test_standalone_object_can_be_kept_and_is_flagged_unjoinable():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[
            _table("customers", "id"),
            _table("orders", "id", "customer_id"),
            _table("exchange_rates", "id", description="Daily currency rates"),
        ],
        relationships=[
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")
        ],
    )
    provider = MockPathProvider(
        json.dumps({"chosen_paths": ["p1", "p2"], "reason": "rates needed as a lookup"})
    )
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Order totals converted to USD"),
        selected_objects=["customers", "orders", "exchange_rates"],
        schema=schema,
    )

    assert resolution.dropped_objects == []
    assert resolution.unjoinable_objects == ["exchange_rates"]


@pytest.mark.asyncio
async def test_edgeless_schema_keeps_everything_without_calling_model():
    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop",
        objects=[
            SchemaObject(name="users", kind=SchemaObjectKind.COLLECTION, fields=[Field(name="_id", type="objectId")]),
            SchemaObject(name="carts", kind=SchemaObjectKind.COLLECTION, fields=[Field(name="_id", type="objectId")]),
        ],
        relationships=[],
    )
    provider = MockPathProvider("")
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("users and carts"),
        selected_objects=["users", "carts"],
        schema=schema,
    )

    assert provider.calls == 0
    assert resolution.slm_invoked is False
    assert resolution.resolved_objects == ["users", "carts"]
    assert resolution.dropped_objects == []


@pytest.mark.asyncio
async def test_parallel_foreign_keys_surface_both_join_columns():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="travel",
        objects=[
            _table("flights", "id", "origin_airport_id", "destination_airport_id"),
            _table("airports", "id", "city"),
        ],
        relationships=[
            Relationship(from_object="flights", from_field="origin_airport_id", to_object="airports", to_field="id"),
            Relationship(from_object="flights", from_field="destination_airport_id", to_object="airports", to_field="id"),
        ],
    )
    provider = MockPathProvider(json.dumps({"chosen_paths": ["p2"], "reason": "departures"}))
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Flights departing from Chennai"),
        selected_objects=["flights", "airports"],
        schema=schema,
    )

    assert "origin_airport_id" in provider.last_prompt
    assert "destination_airport_id" in provider.last_prompt
    assert len(resolution.candidates) == 2
    assert resolution.chosen_path_ids == ["p2"]


@pytest.mark.asyncio
async def test_curated_non_fk_relationship_is_walkable():
    """The finops case: a documented join that is not a declared foreign key."""
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="finopsiq_be",
        objects=[
            _table("discrepancies_details", "id", "match_id"),
            _table("genbooks_requests", "match_id", "invoice_id"),
            _table("invoices", "invoice_id"),
        ],
        relationships=[
            Relationship(
                from_object="discrepancies_details",
                from_field="match_id",
                to_object="genbooks_requests",
                to_field="match_id",
            ),
            Relationship(
                from_object="genbooks_requests",
                from_field="invoice_id",
                to_object="invoices",
                to_field="invoice_id",
            ),
        ],
    )
    provider = MockPathProvider(json.dumps({"chosen_paths": ["p1"], "reason": "via match_id"}))
    selector = JoinPathSelector(provider=provider)

    resolution = await selector.resolve(
        question_analysis=_analysis("Discrepancies traced to their invoices"),
        selected_objects=["discrepancies_details", "invoices"],
        schema=schema,
    )

    # The path exists only because the curated relationship is in the schema.
    assert resolution.connector_objects == ["genbooks_requests"]
    assert "match_id" in provider.last_prompt
