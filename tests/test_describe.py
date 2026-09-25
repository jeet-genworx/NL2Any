"""Tests for batched SLM table/column description generation from a graph/MST TOML."""

import json
from pathlib import Path

import pytest

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from ingestion.schema.describe import (
    TableDescription,
    apply_descriptions,
    describe_tables,
    save_descriptions,
    get_default_descriptions_path,
    table_descriptions,
)

# A chain of 7 tables: t1 -- t2 -- ... -- t7, connected in MST order.
# With batch_size=5, this should split into batches of [t1..t5] and [t6, t7].
CHAIN_MST_TOML = (
    "[graph]\n"
    'database_type = "postgresql"\n'
    'database_name = "shop_db"\n\n'
    + "\n".join(
        f'[[nodes]]\nid = "t{i}"\nkind = "table"\n'
        f'columns = [ {{ name = "id", type = "integer", nullable = false }} ]\n'
        for i in range(1, 8)
    )
    + "\n"
    + "\n".join(
        f'[[mst_edges]]\nfrom = "t{i+1}"\nfrom_column = "prev_id"\nto = "t{i}"\nto_column = "id"\ntype = "many_to_one"\n'
        for i in range(1, 7)
    )
)


def _parse_batch_tables(prompt: str) -> dict[str, list[str]]:
    """Read back the table -> column names the prompt listed for this batch."""
    tables_section = prompt.split("Tables in this batch:")[1].split("Relationships")[0]
    batch: dict[str, list[str]] = {}
    for line in tables_section.strip().splitlines():
        name, _, columns = line.strip("- ").partition(":")
        batch[name.strip()] = [
            col.split("(")[0].strip() for col in columns.split(",") if col.strip()
        ]
    return batch


class _RecordingProvider:
    """Mock provider that records each prompt and returns batched JSON descriptions."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        # Echo back a description per table, plus one per column, for the tables
        # listed in this batch's "Tables in this batch:" section.
        descriptions = {
            name: {
                "description": f"Description of {name}.",
                "columns": {col: f"Column {col} of {name}." for col in columns},
            }
            for name, columns in _parse_batch_tables(prompt).items()
        }
        return "<think>reasoning</think>\n```json\n" + json.dumps({"descriptions": descriptions}) + "\n```"


class _TruncatedThenValidProvider(_RecordingProvider):
    """Returns unusable output for the first `fail_times` calls, then valid JSON.

    Mimics a reasoning model whose <think> block consumes the whole token budget,
    leaving nothing parseable once think-tags are stripped.
    """

    def __init__(self, fail_times: int) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.calls = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            self.prompts.append(prompt)
            return "<think>reasoning that never finished"
        return await super().generate(prompt, **kwargs)


@pytest.mark.asyncio
async def test_describe_tables_retries_when_response_has_no_parseable_json(tmp_path):
    mst_path = tmp_path / "postgres_mst.toml"
    mst_path.write_text(CHAIN_MST_TOML)
    # Fail the very first call, then succeed: the batch must still be described.
    provider = _TruncatedThenValidProvider(fail_times=1)

    descriptions = await describe_tables(mst_path, provider=provider)

    assert provider.calls == 3  # 1 failed + 1 retry for batch 1, + 1 for batch 2
    assert descriptions["t1"].description == "Description of t1."
    assert list(descriptions.keys()) == [f"t{i}" for i in range(1, 8)]


@pytest.mark.asyncio
async def test_describe_tables_gives_up_after_max_attempts(tmp_path):
    mst_path = tmp_path / "postgres_mst.toml"
    mst_path.write_text(CHAIN_MST_TOML)
    provider = _TruncatedThenValidProvider(fail_times=99)

    with pytest.raises(ValueError, match="after 3 attempts"):
        await describe_tables(mst_path, provider=provider)

    assert provider.calls == 3


@pytest.mark.asyncio
async def test_describe_tables_batches_in_groups_of_five(tmp_path):
    mst_path = tmp_path / "postgres_mst.toml"
    mst_path.write_text(CHAIN_MST_TOML)
    provider = _RecordingProvider()

    descriptions = await describe_tables(mst_path, provider=provider)

    # 7 tables at batch_size=5 -> 2 calls: [t1..t5], [t6, t7].
    assert len(provider.prompts) == 2
    assert all(f"t{i}" in provider.prompts[0] for i in range(1, 6))
    assert "t6" in provider.prompts[1] and "t7" in provider.prompts[1]

    # All 7 tables described, in the MST's own sequential order.
    assert list(descriptions.keys()) == [f"t{i}" for i in range(1, 8)]
    assert descriptions["t3"].description == "Description of t3."
    assert descriptions["t3"].columns == {"id": "Column id of t3."}


@pytest.mark.asyncio
async def test_describe_tables_includes_within_batch_relationships(tmp_path):
    mst_path = tmp_path / "postgres_mst.toml"
    mst_path.write_text(CHAIN_MST_TOML)
    provider = _RecordingProvider()

    await describe_tables(mst_path, provider=provider, batch_size=5)

    # t2->t1 relationship should appear in the first batch's prompt (both endpoints in batch 1).
    assert "t2.prev_id -> t1.id" in provider.prompts[0]
    # t6->t5 crosses the batch boundary (t5 is in batch 1, t6 is in batch 2), so it must not
    # be claimed as a within-batch relationship in either prompt.
    assert "t6.prev_id -> t5.id" not in provider.prompts[0]
    assert "t6.prev_id -> t5.id" not in provider.prompts[1]


@pytest.mark.asyncio
async def test_describe_tables_missing_mst_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await describe_tables(tmp_path / "missing.toml", provider=_RecordingProvider())


@pytest.mark.asyncio
async def test_describe_tables_reads_plain_graph_file(tmp_path):
    # Plain graph.py output: key is "edges", not "mst_edges" -- and all edges
    # are present (no tree reduction), unlike an MST file.
    graph_toml = (
        "[graph]\n"
        'database_type = "postgresql"\n'
        'database_name = "shop_db"\n\n'
        '[[nodes]]\nid = "customers"\nkind = "table"\n'
        'columns = [ { name = "id", type = "integer", nullable = false } ]\n\n'
        '[[nodes]]\nid = "orders"\nkind = "table"\n'
        'columns = [ { name = "id", type = "integer", nullable = false } ]\n\n'
        '[[edges]]\nfrom = "orders"\nfrom_column = "customer_id"\nto = "customers"\nto_column = "id"\ntype = "many_to_one"\n'
    )
    graph_path = tmp_path / "postgres_graph.toml"
    graph_path.write_text(graph_toml)
    provider = _RecordingProvider()

    descriptions = await describe_tables(graph_path, provider=provider)

    assert list(descriptions.keys()) == ["customers", "orders"]
    assert "orders.customer_id -> customers.id" in provider.prompts[0]


@pytest.mark.asyncio
async def test_describe_tables_accepts_bare_string_entry(tmp_path):
    """A model that collapses the object to a plain string still yields the
    table description -- it just contributes no column descriptions."""

    class _FlatProvider:
        async def generate(self, prompt: str, **kwargs) -> str:
            names = _parse_batch_tables(prompt).keys()
            return json.dumps({"descriptions": {name: f"Flat {name}." for name in names}})

    graph_path = tmp_path / "postgres_graph.toml"
    graph_path.write_text(CHAIN_MST_TOML)

    descriptions = await describe_tables(graph_path, provider=_FlatProvider())

    assert descriptions["t1"].description == "Flat t1."
    assert descriptions["t1"].columns == {}


def test_table_descriptions_drops_columns():
    result = {
        "customers": TableDescription(
            description="Customer records.", columns={"id": "The id."}
        )
    }
    assert table_descriptions(result) == {"customers": "Customer records."}


def test_apply_descriptions_writes_table_and_column_descriptions():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="integer"), Field(name="city", type="text")],
            )
        ],
    )

    applied = apply_descriptions(
        schema,
        {
            "customers": TableDescription(
                description="Customer records.",
                columns={"id": "Primary key.", "city": "Where they live."},
            )
        },
    )

    assert applied == 2
    customers = schema.get_object("customers")
    assert customers.description == "Customer records."
    assert customers.get_field("id").description == "Primary key."
    assert customers.get_field("city").description == "Where they live."
    assert schema.description_generated_at is not None


def test_apply_descriptions_ignores_columns_not_in_schema():
    """The SLM must never be able to add a field to the canonical schema."""
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="shop_db",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                fields=[Field(name="id", type="integer")],
            )
        ],
    )

    applied = apply_descriptions(
        schema,
        {
            "customers": TableDescription(
                description="Customer records.",
                columns={"id": "Primary key.", "invented_column": "Not real."},
            )
        },
    )

    assert applied == 1
    customers = schema.get_object("customers")
    assert [field.name for field in customers.fields] == ["id"]


def test_apply_descriptions_resolves_nested_dotted_paths():
    """MongoDB nested fields are addressed by dotted path."""
    schema = DatabaseSchema(
        database_type=DatabaseType.MONGODB,
        database_name="shop_demo",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.COLLECTION,
                fields=[
                    Field(
                        name="address",
                        type="object",
                        nested=[Field(name="city", type="string")],
                    )
                ],
            )
        ],
    )

    applied = apply_descriptions(
        schema,
        {
            "customers": TableDescription(
                description="Customer documents.",
                columns={"address.city": "City within the address subdocument."},
            )
        },
    )

    assert applied == 1
    field = schema.get_object("customers").get_field("address.city")
    assert field.description == "City within the address subdocument."


def test_save_descriptions_writes_table_and_column_descriptions(tmp_path):
    out_path = tmp_path / "postgres_descriptions.json"
    save_descriptions(
        {
            "customers": TableDescription(
                description="Customer records.",
                columns={"id": "Primary key.", "city": "Where they live."},
            ),
            "orders": TableDescription(description="Order records.", columns={}),
        },
        out_path,
    )

    assert out_path.exists()
    assert json.loads(out_path.read_text()) == {
        "customers": {
            "description": "Customer records.",
            "columns": {"id": "Primary key.", "city": "Where they live."},
        },
        "orders": {"description": "Order records.", "columns": {}},
    }


def test_get_default_descriptions_path():
    assert get_default_descriptions_path(DatabaseType.POSTGRESQL) == Path(
        "ingestion/schemas/postgres_descriptions.json"
    )
