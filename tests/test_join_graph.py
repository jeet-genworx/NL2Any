"""Tests for the deterministic join-path graph."""

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.core.query_processing.pipeline.join_graph import (
    JoinGraph,
    build_join_path_candidates,
)


def _table(name: str, *columns: str) -> SchemaObject:
    return SchemaObject(
        name=name,
        kind=SchemaObjectKind.TABLE,
        fields=[Field(name=c, type="int") for c in columns],
    )


def _schema(objects: list[SchemaObject], relationships: list[Relationship]) -> DatabaseSchema:
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=objects,
        relationships=relationships,
    )


def _transitive_schema() -> DatabaseSchema:
    """customers -> orders -> order_items -> products: endpoints share no column."""
    return _schema(
        [
            _table("customers", "id", "city"),
            _table("orders", "id", "customer_id"),
            _table("order_items", "id", "order_id", "product_id"),
            _table("products", "id", "name"),
        ],
        [
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="order_items", from_field="order_id", to_object="orders", to_field="id"),
            Relationship(from_object="order_items", from_field="product_id", to_object="products", to_field="id"),
        ],
    )


def test_finds_transitive_path_between_unconnected_endpoints():
    graph = JoinGraph(_transitive_schema())
    paths = graph.find_paths("customers", "products", max_hops=3, max_paths=8)

    assert len(paths) == 1
    tables, edges = paths[0]
    assert tables == ["customers", "orders", "order_items", "products"]
    assert len(edges) == 3


def test_paths_are_undirected():
    graph = JoinGraph(_transitive_schema())
    forward = graph.find_paths("customers", "orders", max_hops=3, max_paths=8)
    backward = graph.find_paths("orders", "customers", max_hops=3, max_paths=8)

    assert len(forward) == 1 and len(backward) == 1
    assert forward[0][0] == ["customers", "orders"]
    assert backward[0][0] == ["orders", "customers"]


def test_max_hops_excludes_paths_that_are_too_long():
    graph = JoinGraph(_transitive_schema())
    assert graph.find_paths("customers", "products", max_hops=2, max_paths=8) == []
    assert len(graph.find_paths("customers", "products", max_hops=3, max_paths=8)) == 1


def test_paths_are_returned_shortest_first():
    # customers reaches payments directly, and also via orders.
    schema = _schema(
        [
            _table("customers", "id"),
            _table("orders", "id", "customer_id"),
            _table("payments", "id", "customer_id", "order_id"),
        ],
        [
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="payments", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="payments", from_field="order_id", to_object="orders", to_field="id"),
        ],
    )
    paths = JoinGraph(schema).find_paths("customers", "payments", max_hops=3, max_paths=8)

    hops = [len(edges) for _, edges in paths]
    assert hops == sorted(hops)
    assert hops[0] == 1


def test_parallel_foreign_keys_are_distinct_paths():
    """origin and destination both point at airports: two different joins."""
    schema = _schema(
        [
            _table("flights", "id", "origin_airport_id", "destination_airport_id"),
            _table("airports", "id", "city"),
        ],
        [
            Relationship(from_object="flights", from_field="origin_airport_id", to_object="airports", to_field="id"),
            Relationship(from_object="flights", from_field="destination_airport_id", to_object="airports", to_field="id"),
        ],
    )
    paths = JoinGraph(schema).find_paths("flights", "airports", max_hops=3, max_paths=8)

    assert len(paths) == 2
    joined_on = {edges[0].from_field for _, edges in paths}
    assert joined_on == {"origin_airport_id", "destination_airport_id"}


def test_components_separate_unrelated_tables():
    schema = _schema(
        [_table("customers", "id"), _table("orders", "id", "customer_id"), _table("audit_log", "id")],
        [Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")],
    )
    graph = JoinGraph(schema)

    assert graph.component_of("customers") == graph.component_of("orders")
    assert graph.component_of("audit_log") != graph.component_of("customers")
    assert graph.find_paths("customers", "audit_log", max_hops=3, max_paths=8) == []


def test_self_referencing_key_does_not_break_traversal():
    schema = _schema(
        [_table("employees", "id", "manager_id", "dept_id"), _table("departments", "id")],
        [
            Relationship(from_object="employees", from_field="manager_id", to_object="employees", to_field="id"),
            Relationship(from_object="employees", from_field="dept_id", to_object="departments", to_field="id"),
        ],
    )
    paths = JoinGraph(schema).find_paths("employees", "departments", max_hops=3, max_paths=8)

    assert len(paths) == 1
    assert paths[0][0] == ["employees", "departments"]


def test_relationship_naming_a_missing_object_is_skipped():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop",
        objects=[_table("customers", "id")],
        relationships=[],
    )
    # Injected after construction so model validation does not reject it.
    schema.relationships.append(
        Relationship(from_object="ghost", from_field="id", to_object="customers", to_field="id")
    )
    graph = JoinGraph(schema)

    assert graph.find_paths("customers", "ghost", max_hops=3, max_paths=8) == []
    assert not graph.has_edges()


def test_candidates_include_transitive_path_and_no_standalone():
    candidates = build_join_path_candidates(
        selected_object_names=["customers", "products"],
        schema=_transitive_schema(),
        max_hops=3,
        max_paths_per_pair=8,
        max_candidates=40,
    )

    assert len(candidates) == 1
    assert candidates[0].path_id == "p1"
    assert candidates[0].connected is True
    assert candidates[0].tables == ["customers", "orders", "order_items", "products"]
    assert candidates[0].hops == 3


def test_candidates_offer_unreachable_object_as_standalone():
    schema = _schema(
        [_table("customers", "id"), _table("orders", "id", "customer_id"), _table("audit_log", "id")],
        [Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")],
    )
    candidates = build_join_path_candidates(
        selected_object_names=["customers", "orders", "audit_log"],
        schema=schema,
        max_hops=3,
        max_paths_per_pair=8,
        max_candidates=40,
    )

    connected = [c for c in candidates if c.connected]
    standalone = [c for c in candidates if not c.connected]
    assert [c.tables for c in connected] == [["customers", "orders"]]
    assert [c.tables for c in standalone] == [["audit_log"]]
    assert "separate part of the schema" in standalone[0].note


def test_standalone_note_distinguishes_too_far_from_disconnected():
    # a -- b -- c -- d chain; a and d are 3 hops apart, excluded at max_hops=1.
    schema = _schema(
        [_table("a", "id"), _table("b", "id", "a_id"), _table("c", "id", "b_id"), _table("d", "id", "c_id")],
        [
            Relationship(from_object="b", from_field="a_id", to_object="a", to_field="id"),
            Relationship(from_object="c", from_field="b_id", to_object="b", to_field="id"),
            Relationship(from_object="d", from_field="c_id", to_object="c", to_field="id"),
        ],
    )
    candidates = build_join_path_candidates(
        selected_object_names=["a", "d"],
        schema=schema,
        max_hops=1,
        max_paths_per_pair=8,
        max_candidates=40,
    )

    assert all(not c.connected for c in candidates)
    assert all("within 1 hops" in c.note for c in candidates)


def test_candidate_ids_stay_contiguous_after_truncation():
    schema = _schema(
        [
            _table("customers", "id"),
            _table("orders", "id", "customer_id"),
            _table("payments", "id", "customer_id", "order_id"),
        ],
        [
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="payments", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="payments", from_field="order_id", to_object="orders", to_field="id"),
        ],
    )
    candidates = build_join_path_candidates(
        selected_object_names=["customers", "orders", "payments"],
        schema=schema,
        max_hops=3,
        max_paths_per_pair=8,
        max_candidates=2,
    )

    assert [c.path_id for c in candidates] == ["p1", "p2"]
    assert [c.hops for c in candidates] == [1, 1]


def test_mongodb_schema_without_relationships_yields_only_standalone():
    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop",
        objects=[
            SchemaObject(name="users", kind=SchemaObjectKind.COLLECTION, fields=[Field(name="_id", type="objectId")]),
            SchemaObject(name="carts", kind=SchemaObjectKind.COLLECTION, fields=[Field(name="_id", type="objectId")]),
        ],
        relationships=[],
    )
    candidates = build_join_path_candidates(
        selected_object_names=["users", "carts"],
        schema=schema,
        max_hops=3,
        max_paths_per_pair=8,
        max_candidates=40,
    )

    assert len(candidates) == 2
    assert all(not c.connected for c in candidates)
