"""Tests for TableSelector stage."""

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
from backend.src.core.query_processing.pipeline.join_graph import build_table_neighborhood
from backend.src.core.query_processing.pipeline.selector import TableSelector


class MockSelectorProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""
        self.last_kwargs = {}

    async def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self.response


@pytest.fixture
def sample_schema() -> DatabaseSchema:
    cust_obj = SchemaObject(
        name="customers",
        kind=SchemaObjectKind.TABLE,
        description="People who buy things",
        fields=[Field(name="id", type="int"), Field(name="city", type="varchar")],
    )
    orders_obj = SchemaObject(
        name="orders",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="int"), Field(name="customer_id", type="int")],
    )
    items_obj = SchemaObject(
        name="order_items",
        kind=SchemaObjectKind.TABLE,
        description="One row per product on an order",
        fields=[
            Field(name="id", type="int"),
            Field(name="order_id", type="int"),
            Field(name="product_id", type="int"),
        ],
    )
    products_obj = SchemaObject(
        name="products",
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name="id", type="int"), Field(name="name", type="varchar")],
    )
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[cust_obj, orders_obj, items_obj, products_obj],
        relationships=[
            Relationship(
                from_object="orders",
                from_field="customer_id",
                to_object="customers",
                to_field="id",
                relationship_type="many_to_one",
            ),
            Relationship(
                from_object="order_items",
                from_field="order_id",
                to_object="orders",
                to_field="id",
                relationship_type="many_to_one",
            ),
            Relationship(
                from_object="order_items",
                from_field="product_id",
                to_object="products",
                to_field="id",
                relationship_type="many_to_one",
            ),
        ],
    )


def _neighborhood(schema, *seeds, levels=3, cap=40):
    return build_table_neighborhood(
        seed_tables=[(name, 0.92 - 0.02 * i, i + 1) for i, name in enumerate(seeds)],
        schema=schema,
        max_levels=levels,
        max_tables=cap,
    )


def _response(**overrides) -> str:
    payload = {
        "selected_objects": ["customers", "orders"],
        "sufficient": True,
        "missing_objects": [],
        "reason": "Both tables are needed to show customer orders.",
        "retrieval_hint": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_selects_tables_and_attaches_their_relationships(sample_schema):
    provider = MockSelectorProvider(_response())
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show customer orders", subjective=["customers", "orders"])

    res = await selector.select(qa, _neighborhood(sample_schema, "customers"), sample_schema)

    assert res.selected_objects == ["customers", "orders"]
    assert res.sufficient is True
    # Relationships are derived from the schema, not read from the model.
    assert len(res.selected_relationships) == 1
    rel = res.selected_relationships[0]
    assert (rel.from_object, rel.from_field, rel.to_object) == ("orders", "customer_id", "customers")


@pytest.mark.asyncio
async def test_relationships_exclude_edges_to_unselected_tables(sample_schema):
    """order_items -> products must not ride along when products was not selected."""
    provider = MockSelectorProvider(
        _response(selected_objects=["orders", "order_items"])
    )
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show order line items", subjective=["line items"])

    res = await selector.select(qa, _neighborhood(sample_schema, "orders"), sample_schema)

    assert res.selected_objects == ["orders", "order_items"]
    assert [(r.from_object, r.to_object) for r in res.selected_relationships] == [
        ("order_items", "orders")
    ]


@pytest.mark.asyncio
async def test_bfs_discovered_bridge_table_can_be_selected(sample_schema):
    """customers and products never match the bridge; the walk supplies it."""
    provider = MockSelectorProvider(
        _response(selected_objects=["customers", "orders", "order_items", "products"])
    )
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Which products do customers buy?")

    neighborhood = _neighborhood(sample_schema, "customers", "products")
    res = await selector.select(qa, neighborhood, sample_schema)

    assert "order_items" in res.selected_objects
    assert len(res.selected_relationships) == 3
    # The prompt said why a table nobody searched for is on the list.
    assert "1 hop from customers" in provider.last_prompt
    assert "query match" in provider.last_prompt


@pytest.mark.asyncio
async def test_prompt_carries_descriptions_columns_and_relationships(sample_schema):
    from backend.src.config import settings

    provider = MockSelectorProvider(_response())
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show customer orders", subjective=["customers", "orders"])

    await selector.select(qa, _neighborhood(sample_schema, "customers"), sample_schema)

    assert "People who buy things" in provider.last_prompt
    assert "Columns: id (int), city (varchar)" in provider.last_prompt
    assert "Relationships (the ONLY valid joins between these tables):" in provider.last_prompt
    assert "orders.customer_id -> customers.id (many_to_one)" in provider.last_prompt
    assert provider.last_kwargs.get("max_tokens") == settings.table_selector_max_tokens


@pytest.mark.asyncio
async def test_filters_hallucinated_tables(sample_schema):
    provider = MockSelectorProvider(
        _response(selected_objects=["customers", "fake_table"])
    )
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show customers", subjective=["customers"])

    res = await selector.select(qa, _neighborhood(sample_schema, "customers"), sample_schema)

    assert res.selected_objects == ["customers"]
    assert "fake_table" not in res.selected_objects


@pytest.mark.asyncio
async def test_filters_tables_outside_the_neighborhood(sample_schema):
    """A real table the walk never reached is still not a valid selection."""
    provider = MockSelectorProvider(
        _response(selected_objects=["customers", "products"])
    )
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show customers", subjective=["customers"])

    # One hop from customers reaches orders only.
    res = await selector.select(
        qa, _neighborhood(sample_schema, "customers", levels=1), sample_schema
    )

    assert res.selected_objects == ["customers"]


@pytest.mark.asyncio
async def test_reports_insufficiency_with_a_retrieval_hint(sample_schema):
    provider = MockSelectorProvider(
        _response(
            selected_objects=["customers"],
            sufficient=False,
            missing_objects=["shipments"],
            reason="No shipment data available.",
            retrieval_hint="tables containing shipment tracking records",
        )
    )
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="When did each customer's order ship?")

    res = await selector.select(
        qa, _neighborhood(sample_schema, "customers", levels=1), sample_schema
    )

    assert res.sufficient is False
    assert "shipments" in res.missing_objects
    assert res.retrieval_hint == "tables containing shipment tracking records"


@pytest.mark.asyncio
async def test_passes_retry_feedback_into_the_prompt(sample_schema):
    provider = MockSelectorProvider(_response())
    selector = TableSelector(provider=provider)
    qa = QuestionAnalysis(question="Show orders", subjective=["orders"])

    feedback_msg = "Previous attempt was missing orders."
    await selector.select(
        qa, _neighborhood(sample_schema, "customers"), sample_schema, feedback=feedback_msg
    )

    assert "PREVIOUS SELECTION FEEDBACK (RETRY)" in provider.last_prompt
    assert feedback_msg in provider.last_prompt


@pytest.mark.asyncio
async def test_empty_neighborhood_is_insufficient(sample_schema):
    selector = TableSelector(provider=None)
    qa = QuestionAnalysis(question="Show astronauts", subjective=["astronauts"])

    res = await selector.select(qa, _neighborhood(sample_schema), sample_schema)

    assert res.selected_objects == []
    assert res.selected_relationships == []
    assert res.sufficient is False


@pytest.mark.asyncio
async def test_model_failure_falls_back_to_the_nearest_table(sample_schema):
    class FailingProvider:
        async def generate(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("model offline")

    selector = TableSelector(provider=FailingProvider())
    qa = QuestionAnalysis(question="Show customers")

    res = await selector.select(qa, _neighborhood(sample_schema, "customers"), sample_schema)

    assert res.selected_objects == ["customers"]
    assert res.selected_relationships == []
    assert res.sufficient is True


@pytest.mark.asyncio
async def test_relationship_block_is_explicit_when_there_are_no_foreign_keys():
    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop",
        objects=[
            SchemaObject(
                name="users",
                kind=SchemaObjectKind.COLLECTION,
                fields=[Field(name="_id", type="objectId")],
            )
        ],
        relationships=[],
    )
    provider = MockSelectorProvider(_response(selected_objects=["users"]))
    selector = TableSelector(provider=provider)

    res = await selector.select(
        QuestionAnalysis(question="Show users"), _neighborhood(schema, "users"), schema
    )

    assert res.selected_objects == ["users"]
    assert "Do not invent one." in provider.last_prompt


@pytest.mark.asyncio
async def test_oversized_neighborhood_sheds_columns_rather_than_overflowing(sample_schema):
    """An overflowing prompt loses its instructions, so detail is shed instead."""
    provider = MockSelectorProvider(_response())
    selector = TableSelector(provider=provider)
    template_size = len(selector._load_prompt_template()) + 512
    # Room for the bare table lines and the relationship block, but not columns.
    selector._prompt_char_budget = lambda: template_size + 400

    await selector.select(
        QuestionAnalysis(question="Show customer orders"),
        _neighborhood(sample_schema, "customers"),
        sample_schema,
    )

    assert "Columns:" not in provider.last_prompt
    # Names, descriptions and relationships -- this stage's actual output -- survive.
    assert "• customers (query match" in provider.last_prompt
    assert "People who buy things" in provider.last_prompt
    assert "orders.customer_id -> customers.id" in provider.last_prompt
    assert len(provider.last_prompt) <= template_size + 400


@pytest.mark.asyncio
async def test_unfittable_prompt_keeps_at_least_the_nearest_table(sample_schema):
    provider = MockSelectorProvider(_response(selected_objects=["customers"]))
    selector = TableSelector(provider=provider)
    selector._prompt_char_budget = lambda: 1

    res = await selector.select(
        QuestionAnalysis(question="Show customers"),
        _neighborhood(sample_schema, "customers"),
        sample_schema,
    )

    assert "• customers" in provider.last_prompt
    assert res.selected_objects == ["customers"]
