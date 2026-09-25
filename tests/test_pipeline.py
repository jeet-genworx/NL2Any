"""Tests for the full ingestion pipeline orchestrator."""

import json

import pytest

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from ingestion.pipeline import run_ingestion_pipeline


def _fake_schema(db_type: DatabaseType) -> DatabaseSchema:
    kind = SchemaObjectKind.TABLE if db_type == DatabaseType.POSTGRESQL else SchemaObjectKind.COLLECTION
    customers = SchemaObject(name="customers", kind=kind, fields=[Field(name="id", type="integer")])
    orders = SchemaObject(
        name="orders", kind=kind, fields=[Field(name="id", type="integer"), Field(name="customer_id", type="integer")]
    )
    relationships = (
        [Relationship(from_object="orders", from_field="customer_id", to_object="customers", to_field="id")]
        if db_type == DatabaseType.POSTGRESQL
        else []
    )
    return DatabaseSchema(
        database_type=db_type,
        database_name="shop_db",
        objects=[customers, orders],
        relationships=relationships,
    )


class _FakeAdapter:
    def __init__(self, schema: DatabaseSchema) -> None:
        self._schema = schema

    def get_metadata(self) -> DatabaseSchema:
        return self._schema

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _RecordingChatProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        tables_section = prompt.split("Tables in this batch:")[1].split("Relationships")[0]
        descriptions = {}
        for line in tables_section.strip().splitlines():
            name, _, columns = line.strip("- ").partition(":")
            descriptions[name.strip()] = {
                "description": f"Desc of {name.strip()}.",
                "columns": {
                    col.split("(")[0].strip(): f"Col {col.split('(')[0].strip()}."
                    for col in columns.split(",")
                    if col.strip()
                },
            }
        return "```json\n" + json.dumps({"descriptions": descriptions}) + "\n```"


class _RecordingEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), float(len(t))] for i, t in enumerate(texts)]


def _patch_default_paths(monkeypatch, tmp_path) -> None:
    """Redirect every get_default_*_path() the pipeline uses into tmp_path,
    without changing the process CWD (prompt loading is still CWD-relative
    to the real repo, same as in production)."""
    monkeypatch.setattr("ingestion.pipeline.get_default_schema_path", lambda db_type: tmp_path / f"{db_type.value}.toml")
    monkeypatch.setattr("ingestion.pipeline.get_default_graph_path", lambda db_type: tmp_path / f"{db_type.value}_graph.toml")
    monkeypatch.setattr("ingestion.pipeline.get_default_mst_path", lambda db_type: tmp_path / f"{db_type.value}_mst.toml")
    monkeypatch.setattr("ingestion.pipeline.get_default_descriptions_path", lambda db_type: tmp_path / f"{db_type.value}_descriptions.json")
    monkeypatch.setattr("ingestion.pipeline.get_default_embeddings_path", lambda db_type: tmp_path / f"{db_type.value}_embeddings.json")
    monkeypatch.setattr("ingestion.schema.describe.KoboldCppProvider", lambda: _RecordingChatProvider())
    monkeypatch.setattr("ingestion.schema.embed.KoboldCppEmbeddingProvider", lambda: _RecordingEmbeddingProvider())


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_postgres_uses_mst(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    monkeypatch.setattr("ingestion.pipeline.get_adapter", lambda db_type: _FakeAdapter(schema))
    _patch_default_paths(monkeypatch, tmp_path)

    summary = await run_ingestion_pipeline(DatabaseType.POSTGRESQL, use_mst=True)

    assert summary["used_mst"] is True
    assert summary["table_count"] == 2
    assert summary["relationship_count"] == 1
    assert summary["description_count"] == 2
    assert summary["embedding_count"] == 2
    assert (tmp_path / "postgresql.toml").exists()
    assert (tmp_path / "postgresql_graph.toml").exists()
    assert (tmp_path / "postgresql_mst.toml").exists()
    assert (tmp_path / "postgresql_descriptions.json").exists()
    assert (tmp_path / "postgresql_embeddings.json").exists()

    # The descriptions JSON is the full record: table descriptions AND column
    # descriptions. Only the embedding step narrows to table text.
    assert json.loads((tmp_path / "postgresql_descriptions.json").read_text()) == {
        "customers": {"description": "Desc of customers.", "columns": {"id": "Col id."}},
        "orders": {
            "description": "Desc of orders.",
            "columns": {"id": "Col id.", "customer_id": "Col customer_id."},
        },
    }

    # The embedded text is the table description alone -- the mock encodes each
    # input's length, so this pins down that no column text was appended.
    embeddings = json.loads((tmp_path / "postgresql_embeddings.json").read_text())
    assert embeddings["customers"][1] == float(len("Desc of customers."))

    # customers.id + orders.id + orders.customer_id
    assert summary["column_description_count"] == 3
    schema_toml = (tmp_path / "postgresql.toml").read_text()
    assert "Desc of customers." in schema_toml
    assert "Col customer_id." in schema_toml
    assert 'description_generated_at = ""' not in schema_toml


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_postgres_use_mst_false_reads_graph(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    monkeypatch.setattr("ingestion.pipeline.get_adapter", lambda db_type: _FakeAdapter(schema))
    _patch_default_paths(monkeypatch, tmp_path)

    summary = await run_ingestion_pipeline(DatabaseType.POSTGRESQL, use_mst=False)

    assert summary["used_mst"] is False
    assert summary["description_count"] == 2


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_mongo_always_uses_graph_no_mst(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.MONGODB)
    monkeypatch.setattr("ingestion.pipeline.get_adapter", lambda db_type: _FakeAdapter(schema))
    _patch_default_paths(monkeypatch, tmp_path)

    # use_mst=True requested, but Mongo has no MST, so it must fall back to the graph.
    summary = await run_ingestion_pipeline(DatabaseType.MONGODB, use_mst=True)

    assert summary["used_mst"] is False
    assert summary["mst_path"] is None
    assert not (tmp_path / "mongodb_mst.toml").exists()
    assert (tmp_path / "mongodb_graph.toml").exists()

    # Column descriptions land in the collections' schema TOML for Mongo too.
    assert summary["column_description_count"] == 3
    schema_toml = (tmp_path / "mongodb.toml").read_text()
    assert "Desc of customers." in schema_toml
    assert "Col customer_id." in schema_toml
