"""Tests for batched SLM table description generation from a graph/MST TOML."""

import json
from pathlib import Path

import pytest

from query_processing.models.schema import DatabaseType
from ingestion.schema.describe import (
    describe_tables,
    save_descriptions,
    get_default_descriptions_path,
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


class _RecordingProvider:
    """Mock provider that records each prompt and returns batched JSON descriptions."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        # Echo back a description per table mentioned in this batch's "Tables in this batch:" section.
        tables_section = prompt.split("Tables in this batch:")[1].split("Relationships")[0]
        table_names = [line.split(":")[0].strip("- ").strip() for line in tables_section.strip().splitlines()]
        descriptions = {name: f"Description of {name}." for name in table_names}
        return "<think>reasoning</think>\n```json\n" + json.dumps({"descriptions": descriptions}) + "\n```"


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
    assert descriptions["t3"] == "Description of t3."


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


def test_save_descriptions_writes_table_name_keyed_json(tmp_path):
    out_path = tmp_path / "postgres_descriptions.json"
    save_descriptions({"customers": "Customer records.", "orders": "Order records."}, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data == {"customers": "Customer records.", "orders": "Order records."}


def test_get_default_descriptions_path():
    assert get_default_descriptions_path(DatabaseType.POSTGRESQL) == Path(
        "ingestion/schemas/postgres_descriptions.json"
    )
