"""Tests for the deterministic foreign-key graph and the BFS neighborhood walk."""

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
    build_table_neighborhood,
    connect_selection,
    relationships_among,
)


def _table(name: str, *columns: str, description: str = "") -> SchemaObject:
    return SchemaObject(
        name=name,
        kind=SchemaObjectKind.TABLE,
        description=description,
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
            _table("order_items", "id", "order_id", "product_id", description="line items"),
            _table("products", "id", "name"),
        ],
        [
            Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id"),
            Relationship(from_object="order_items", from_field="order_id", to_object="orders", to_field="id"),
            Relationship(from_object="order_items", from_field="product_id", to_object="products", to_field="id"),
        ],
    )


def _seeds(*names: str) -> list[tuple[str, float | None, int | None]]:
    """Seed triples as embedding retrieval produces them, descending similarity."""
    return [(name, 0.9 - 0.01 * i, i + 1) for i, name in enumerate(names)]


def test_bfs_levels_records_shortest_hop_count():
    graph = JoinGraph(_transitive_schema())
    levels = graph.bfs_levels("customers", max_levels=3)

    assert levels == {"customers": 0, "orders": 1, "order_items": 2, "products": 3}


def test_bfs_levels_respects_the_limit():
    graph = JoinGraph(_transitive_schema())

    assert graph.bfs_levels("customers", max_levels=1) == {"customers": 0, "orders": 1}
    assert "products" not in graph.bfs_levels("customers", max_levels=2)


def test_traversal_is_undirected():
    graph = JoinGraph(_transitive_schema())

    # orders points at customers; the walk reaches customers from orders anyway.
    assert graph.bfs_levels("orders", max_levels=1) == {
        "orders": 0,
        "customers": 1,
        "order_items": 1,
    }


def test_parallel_foreign_keys_yield_one_neighbor():
    """origin and destination both point at airports: one table, two joins."""
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
    graph = JoinGraph(schema)

    assert graph.neighbors("flights") == ["airports"]
    # Both keys survive as relationships, so the generator can still pick one.
    neighborhood = build_table_neighborhood(_seeds("flights"), schema, max_levels=1, max_tables=10)
    assert {r.from_field for r in neighborhood.relationships} == {
        "origin_airport_id",
        "destination_airport_id",
    }


def test_components_separate_unrelated_tables():
    schema = _schema(
        [_table("customers", "id"), _table("orders", "id", "customer_id"), _table("audit_log", "id")],
        [Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")],
    )
    graph = JoinGraph(schema)

    assert graph.component_of("customers") == graph.component_of("orders")
    assert graph.component_of("audit_log") != graph.component_of("customers")
    assert "audit_log" not in graph.bfs_levels("customers", max_levels=3)


def test_self_referencing_key_does_not_break_traversal():
    schema = _schema(
        [_table("employees", "id", "manager_id", "dept_id"), _table("departments", "id")],
        [
            Relationship(from_object="employees", from_field="manager_id", to_object="employees", to_field="id"),
            Relationship(from_object="employees", from_field="dept_id", to_object="departments", to_field="id"),
        ],
    )

    assert JoinGraph(schema).bfs_levels("employees", max_levels=2) == {
        "employees": 0,
        "departments": 1,
    }


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

    assert graph.bfs_levels("customers", max_levels=3) == {"customers": 0}
    assert not graph.has_edges()


def test_neighborhood_reaches_the_bridge_table_nobody_matched():
    """The case the stage exists for: customers and products match, the bridge does not."""
    neighborhood = build_table_neighborhood(
        _seeds("customers", "products"),
        _transitive_schema(),
        max_levels=3,
        max_tables=40,
    )

    assert neighborhood.seed_objects == ["customers", "products"]
    assert set(neighborhood.table_names()) == {"customers", "products", "orders", "order_items"}
    levels = {t.name: t.level for t in neighborhood.tables}
    assert levels == {"customers": 0, "products": 0, "order_items": 1, "orders": 1}
    # Every foreign key internal to the set comes along.
    assert len(neighborhood.relationships) == 3
    assert neighborhood.truncated is False


def test_neighborhood_records_which_seed_reached_each_table():
    neighborhood = build_table_neighborhood(
        _seeds("customers", "products"),
        _transitive_schema(),
        max_levels=1,
        max_tables=40,
    )
    reached = {t.name: t.reached_from for t in neighborhood.tables}

    assert reached["orders"] == ["customers"]
    assert reached["order_items"] == ["products"]
    assert reached["customers"] == ["customers"]


def test_seeds_come_first_and_carry_their_retrieval_scores():
    neighborhood = build_table_neighborhood(
        [("products", 0.81, 2), ("customers", 0.93, 1)],
        _transitive_schema(),
        max_levels=2,
        max_tables=40,
    )

    # Seed order is retrieval order, not alphabetical.
    assert neighborhood.table_names()[:2] == ["products", "customers"]
    assert neighborhood.tables[0].similarity == 0.81
    assert neighborhood.tables[0].rank == 2
    # Discovered tables carry no score, because nothing scored them.
    discovered = [t for t in neighborhood.tables if t.level > 0]
    assert discovered and all(t.similarity is None and t.rank is None for t in discovered)


def test_neighborhood_descriptions_come_from_the_schema():
    neighborhood = build_table_neighborhood(
        _seeds("customers"), _transitive_schema(), max_levels=2, max_tables=40
    )
    by_name = {t.name: t for t in neighborhood.tables}

    assert by_name["order_items"].description == "line items"


def test_cap_discards_the_far_end_of_the_walk_and_keeps_seeds():
    neighborhood = build_table_neighborhood(
        _seeds("customers"),
        _transitive_schema(),
        max_levels=3,
        max_tables=2,
    )

    assert neighborhood.truncated is True
    assert neighborhood.table_names() == ["customers", "orders"]
    assert neighborhood.discarded_objects == ["order_items", "products"]
    # Relationships never reference a discarded table.
    names = {n.lower() for n in neighborhood.table_names()}
    assert all(
        r.from_object.lower() in names and r.to_object.lower() in names
        for r in neighborhood.relationships
    )


def test_cap_never_drops_a_seed():
    neighborhood = build_table_neighborhood(
        _seeds("customers", "products", "orders"),
        _transitive_schema(),
        max_levels=3,
        max_tables=1,
    )

    assert set(neighborhood.table_names()) == {"customers", "products", "orders"}
    assert neighborhood.truncated is True


def test_unknown_seed_is_skipped_rather_than_walked():
    neighborhood = build_table_neighborhood(
        _seeds("customers", "not_a_table"),
        _transitive_schema(),
        max_levels=1,
        max_tables=40,
    )

    assert neighborhood.seed_objects == ["customers"]
    assert "not_a_table" not in neighborhood.table_names()


def test_duplicate_seeds_are_collapsed():
    neighborhood = build_table_neighborhood(
        [("customers", 0.9, 1), ("CUSTOMERS", 0.7, 2)],
        _transitive_schema(),
        max_levels=0,
        max_tables=40,
    )

    assert neighborhood.table_names() == ["customers"]


def test_mongodb_schema_without_relationships_yields_only_the_seeds():
    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop",
        objects=[
            SchemaObject(name="users", kind=SchemaObjectKind.COLLECTION, fields=[Field(name="_id", type="objectId")]),
            SchemaObject(name="carts", kind=SchemaObjectKind.COLLECTION, fields=[Field(name="_id", type="objectId")]),
        ],
        relationships=[],
    )
    neighborhood = build_table_neighborhood(
        _seeds("users"), schema, max_levels=3, max_tables=40
    )

    assert neighborhood.table_names() == ["users"]
    assert neighborhood.relationships == []
    assert neighborhood.tables[0].kind == "collection"


def test_relationships_among_is_restricted_to_the_given_set():
    schema = _transitive_schema()

    assert relationships_among(["customers", "orders"], schema) == [schema.relationships[0]]
    assert relationships_among(["customers", "products"], schema) == []


def test_connect_selection_adds_the_bridge_the_selector_left_out():
    """The observed failure: endpoints selected, connecting table omitted."""
    resolved, added, unjoinable = connect_selection(
        ["customers", "products"], _transitive_schema()
    )

    assert added == ["orders", "order_items"]
    assert set(resolved) == {"customers", "products", "orders", "order_items"}
    assert unjoinable == []


def test_connect_selection_leaves_a_joinable_selection_alone():
    resolved, added, unjoinable = connect_selection(
        ["customers", "orders"], _transitive_schema()
    )

    assert resolved == ["customers", "orders"]
    assert added == []
    assert unjoinable == []


def test_connect_selection_only_bridges_through_tables_the_selector_saw():
    """A bridge from outside the neighborhood is one nobody could have rejected."""
    schema = _transitive_schema()

    resolved, added, unjoinable = connect_selection(
        ["customers", "products"],
        schema,
        allowed=["customers", "products", "orders"],
    )

    # order_items is the only route onward and was not shown, so products stays
    # unjoinable rather than being connected through an unseen table.
    assert "order_items" not in resolved
    assert unjoinable == ["products"]
    assert added == []


def test_connect_selection_reports_what_no_path_can_reach():
    schema = _schema(
        [
            _table("customers", "id"),
            _table("orders", "id", "customer_id"),
            _table("audit_log", "id"),
        ],
        [Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")],
    )

    resolved, added, unjoinable = connect_selection(
        ["customers", "orders", "audit_log"], schema
    )

    assert set(resolved) == {"customers", "orders"}
    assert unjoinable == ["audit_log"]
    assert added == []


def test_connect_selection_takes_the_shortest_repair():
    """customers reaches payments directly; the one-hop edge beats going via orders."""
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

    resolved, added, _ = connect_selection(["orders", "payments"], schema)

    # Already joinable: payments.order_id -> orders.id needs no repair at all.
    assert added == []
    assert resolved == ["orders", "payments"]


def test_connect_selection_is_a_no_op_below_two_tables():
    assert connect_selection(["customers"], _transitive_schema()) == (["customers"], [], [])
    assert connect_selection([], _transitive_schema()) == ([], [], [])


def test_connect_selection_canonicalizes_and_de_duplicates():
    resolved, added, _ = connect_selection(
        ["CUSTOMERS", "customers", "Orders"], _transitive_schema()
    )

    assert resolved == ["customers", "orders"]
    assert added == []
