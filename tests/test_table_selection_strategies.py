"""Tests for the table-selection strategies and their wiring into Stage 1.

Covers the three things the toggle has to get right: BFS narrows the view and
turns the prefix cache off, whole-schema leaves both exactly as they were, and
a BFS that cannot run degrades to whole-schema instead of failing the run.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.src.core.query_processing.pipeline.fsm.adapters import TableSelectionAdapter
from backend.src.core.query_processing.pipeline.fsm.context import PipelineContext
from backend.src.core.query_processing.pipeline.fsm.events import TableSelectionCompleted
from backend.src.core.query_processing.pipeline.join_graph import JoinGraph
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.core.query_processing.pipeline.strategies import (
    BfsRetrievalUnavailable,
    BfsStrategy,
    WholeSchemaStrategy,
)
from backend.src.core.query_processing.pipeline.strategies.bfs import narrow_schema
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
)
from backend.src.data.models.targets import DatabaseTarget
from backend.src.schemas.pipeline import CandidateTable, TableSelectionResult


def _obj(name: str) -> SchemaObject:
    return SchemaObject(
        name=name,
        description=f"{name} table",
        fields=[Field(name="id", type="integer"), Field(name="label", type="varchar")],
    )


@pytest.fixture
def chain_schema() -> DatabaseSchema:
    """customers -> orders -> order_items, plus an unrelated audit_log.

    `order_items` is the bridge case the walk exists for: a question about
    customer revenue never names it, so embedding retrieval never returns it.
    """
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[_obj("customers"), _obj("orders"), _obj("order_items"), _obj("audit_log")],
        relationships=[
            Relationship(
                from_object="orders",
                from_field="customer_id",
                to_object="customers",
                to_field="id",
            ),
            Relationship(
                from_object="order_items",
                from_field="order_id",
                to_object="orders",
                to_field="id",
            ),
        ],
    )


def _context(schema: DatabaseSchema, **kwargs) -> PipelineContext:
    return PipelineContext(
        run_id="a" * 32,
        schema=schema,
        target_name="postgres",
        question="revenue per customer",
        processed_question="revenue per customer",
        **kwargs,
    )


# -- narrow_schema ------------------------------------------------------


def test_narrow_schema_keeps_only_named_tables(chain_schema):
    narrowed = narrow_schema(chain_schema, ["customers", "orders"])
    assert [o.name for o in narrowed.objects] == ["customers", "orders"]


def test_narrow_schema_drops_edges_whose_endpoints_are_gone(chain_schema):
    """An edge must never outlive the tables it joins."""
    narrowed = narrow_schema(chain_schema, ["customers", "orders"])
    assert [(r.from_object, r.to_object) for r in narrowed.relationships] == [
        ("orders", "customers")
    ]


def test_narrow_schema_preserves_database_identity(chain_schema):
    narrowed = narrow_schema(chain_schema, ["customers"])
    assert narrowed.database_name == chain_schema.database_name
    assert narrowed.database_type == chain_schema.database_type


# -- WholeSchemaStrategy ------------------------------------------------


@pytest.mark.asyncio
async def test_whole_schema_passes_the_full_schema_and_a_cache_key(chain_schema):
    scope = await WholeSchemaStrategy().scope(_context(chain_schema))
    assert scope.schema_view is chain_schema
    assert scope.cache_key == "postgres"


# -- BfsStrategy --------------------------------------------------------


def _retriever(*names: str) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        CandidateTable(table_name=n, similarity=0.9 - i / 100, rank=i + 1)
        for i, n in enumerate(names)
    ]
    return retriever


def _embedder() -> MagicMock:
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return provider


@pytest.mark.asyncio
async def test_bfs_reaches_the_bridge_table_retrieval_missed(chain_schema):
    """The point of the walk: order_items is never retrieved, but must be offered."""
    strategy = BfsStrategy(
        retriever=_retriever("customers"),
        embedding_provider=_embedder(),
        graph=JoinGraph(chain_schema),
    )
    scope = await strategy.scope(_context(chain_schema))

    offered = {o.name for o in scope.schema_view.objects}
    assert "customers" in offered
    assert "order_items" in offered, "the FK walk should supply the unretrieved bridge"


@pytest.mark.asyncio
async def test_bfs_excludes_tables_with_no_path_from_a_match(chain_schema):
    strategy = BfsStrategy(
        retriever=_retriever("customers"),
        embedding_provider=_embedder(),
        graph=JoinGraph(chain_schema),
    )
    scope = await strategy.scope(_context(chain_schema))
    assert "audit_log" not in {o.name for o in scope.schema_view.objects}


@pytest.mark.asyncio
async def test_bfs_never_offers_a_cache_key(chain_schema):
    """The prompt is question-specific, so a KV slot must never be reused."""
    strategy = BfsStrategy(
        retriever=_retriever("customers"),
        embedding_provider=_embedder(),
        graph=JoinGraph(chain_schema),
    )
    scope = await strategy.scope(_context(chain_schema))
    assert scope.cache_key is None


@pytest.mark.asyncio
async def test_bfs_raises_when_nothing_matches(chain_schema):
    strategy = BfsStrategy(
        retriever=_retriever(),
        embedding_provider=_embedder(),
        graph=JoinGraph(chain_schema),
    )
    with pytest.raises(BfsRetrievalUnavailable):
        await strategy.scope(_context(chain_schema))


@pytest.mark.asyncio
async def test_bfs_raises_when_the_embedding_call_fails(chain_schema):
    provider = MagicMock()
    provider.embed = AsyncMock(side_effect=RuntimeError("embedding server down"))
    strategy = BfsStrategy(
        retriever=_retriever("customers"),
        embedding_provider=provider,
        graph=JoinGraph(chain_schema),
    )
    with pytest.raises(BfsRetrievalUnavailable):
        await strategy.scope(_context(chain_schema))


@pytest.mark.asyncio
async def test_bfs_retry_widens_rather_than_replaces(chain_schema):
    """A retry must offer a superset: the attempt that failed is never narrowed."""
    retriever = _retriever("customers")
    strategy = BfsStrategy(
        retriever=retriever,
        embedding_provider=_embedder(),
        graph=JoinGraph(chain_schema),
    )
    first = await strategy.scope(_context(chain_schema))

    retriever.retrieve.return_value = [
        CandidateTable(table_name="audit_log", similarity=0.7, rank=1)
    ]
    second = await strategy.scope(
        _context(chain_schema, selector_retries=1, s1_system_retry="need the audit log")
    )

    first_names = {o.name for o in first.schema_view.objects}
    second_names = {o.name for o in second.schema_view.objects}
    assert first_names <= second_names
    assert "audit_log" in second_names


# -- adapter wiring -----------------------------------------------------


def _selector(**kwargs) -> MagicMock:
    selector = MagicMock()
    selector.select = AsyncMock(
        return_value=TableSelectionResult(
            selected_objects=["customers"],
            sufficient=True,
            missing_objects=[],
            reason="ok",
            retrieval_hint=None,
        )
    )
    selector.last_prompt_tokens = 11
    return selector


@pytest.mark.asyncio
async def test_adapter_default_call_is_unchanged_by_strategies(chain_schema):
    """Whole-schema must not pass use_prefix_cache at all.

    A selector written before the flag existed -- and every existing test
    double -- would raise TypeError on an unexpected keyword.
    """
    selector = _selector()
    adapter = TableSelectionAdapter(selector)
    event, tokens = await adapter.execute(_context(chain_schema))

    assert isinstance(event, TableSelectionCompleted)
    assert tokens == 11
    kwargs = selector.select.await_args.kwargs
    assert "use_prefix_cache" not in kwargs
    assert kwargs["schema"] is chain_schema


@pytest.mark.asyncio
async def test_adapter_disables_prefix_cache_for_bfs(chain_schema):
    selector = _selector()
    adapter = TableSelectionAdapter(
        selector,
        BfsStrategy(
            retriever=_retriever("customers"),
            embedding_provider=_embedder(),
            graph=JoinGraph(chain_schema),
        ),
    )
    await adapter.execute(_context(chain_schema))

    kwargs = selector.select.await_args.kwargs
    assert kwargs["use_prefix_cache"] is False
    assert kwargs["schema"] is not chain_schema


@pytest.mark.asyncio
async def test_adapter_falls_back_to_whole_schema_when_bfs_cannot_run(chain_schema):
    """Losing the narrowing should cost tokens, not the answer."""
    selector = _selector()
    context = _context(chain_schema)
    adapter = TableSelectionAdapter(
        selector,
        BfsStrategy(
            retriever=_retriever(),  # nothing above threshold
            embedding_provider=_embedder(),
            graph=JoinGraph(chain_schema),
        ),
    )
    event, _ = await adapter.execute(context)

    assert isinstance(event, TableSelectionCompleted)
    assert selector.select.await_args.kwargs["schema"] is chain_schema
    assert context.selection_strategy_used == "whole_schema"
    assert context.selection_notice is not None


@pytest.mark.asyncio
async def test_adapter_records_the_strategy_that_actually_ran(chain_schema):
    context = _context(chain_schema)
    adapter = TableSelectionAdapter(
        _selector(),
        BfsStrategy(
            retriever=_retriever("customers"),
            embedding_provider=_embedder(),
            graph=JoinGraph(chain_schema),
        ),
    )
    await adapter.execute(context)

    assert context.selection_strategy_used == "bfs"
    assert context.table_neighborhood is not None
    assert [c.table_name for c in context.candidates] == ["customers"]


# -- orchestrator resolution --------------------------------------------


def test_bfs_is_refused_for_mongo_without_failing_the_run():
    """MongoDB has no foreign keys, so the walk cannot add anything."""
    orch = NL2AnyQueryOrchestrator()
    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB, database_name="m", objects=[_obj("c")]
    )
    target = DatabaseTarget(
        key="mongo", label="mongo", db_type=DatabaseType.MONGODB, file_stem="mongo"
    )

    strategy, name = orch._resolve_selection_strategy("bfs", target, DatabaseType.MONGODB, schema)
    assert name == "whole_schema"
    assert isinstance(strategy, WholeSchemaStrategy)


def test_unknown_strategy_name_falls_back_to_whole_schema(chain_schema):
    orch = NL2AnyQueryOrchestrator()
    target = DatabaseTarget(
        key="postgres",
        label="postgres",
        db_type=DatabaseType.POSTGRESQL,
        file_stem="postgres",
    )
    _, name = orch._resolve_selection_strategy(
        "nonsense", target, DatabaseType.POSTGRESQL, chain_schema
    )
    assert name == "whole_schema"


def test_join_graph_is_rebuilt_for_a_reloaded_schema(chain_schema):
    """Caching by schema identity, not name: a reload must not reuse the graph."""
    orch = NL2AnyQueryOrchestrator()
    reloaded = chain_schema.model_copy(deep=True)

    first = orch._get_join_graph(chain_schema)
    assert orch._get_join_graph(chain_schema) is first
    assert orch._get_join_graph(reloaded) is not first
