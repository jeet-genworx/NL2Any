"""Tests for the full ingestion pipeline orchestrator."""

import json
import tomllib

import pytest

from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    Relationship,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.core.ingestion.pipeline import existing_ingestion, run_ingestion_pipeline
from backend.src.data.models.targets import as_target
from backend.src.data.repositories import files


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


# Artifact path resolver -> the filename suffix it produces.
_ARTIFACT_PATHS = {
    "schema_path": ".toml",
    "graph_path": "_graph.toml",
    "mst_path": "_mst.toml",
    "descriptions_path": "_descriptions.toml",
    "embeddings_path": "_embeddings.json",
}


def _patch_default_paths(monkeypatch, tmp_path) -> None:
    """Redirect every canonical artifact path into tmp_path, without changing
    the process CWD (prompt loading is still CWD-relative to the real repo,
    same as in production)."""
    for resolver, suffix in _ARTIFACT_PATHS.items():
        monkeypatch.setattr(
            f"backend.src.data.repositories.paths.{resolver}",
            lambda database, suffix=suffix: tmp_path / f"{as_target(database).file_stem}{suffix}",
        )
    monkeypatch.setattr("backend.src.core.ingestion.schema.describe.KoboldCppProvider", lambda: _RecordingChatProvider())
    monkeypatch.setattr("backend.src.core.ingestion.schema.embed.KoboldCppEmbeddingProvider", lambda: _RecordingEmbeddingProvider())


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_postgres_uses_mst(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    monkeypatch.setattr("backend.src.core.ingestion.pipeline.get_adapter", lambda db_type: _FakeAdapter(schema))
    _patch_default_paths(monkeypatch, tmp_path)

    summary = await run_ingestion_pipeline(DatabaseType.POSTGRESQL, use_mst=True)

    assert summary["used_mst"] is True
    assert summary["table_count"] == 2
    assert summary["relationship_count"] == 1
    assert summary["description_count"] == 2
    assert summary["embedding_count"] == 2
    assert (tmp_path / "postgres.toml").exists()
    assert (tmp_path / "postgres_graph.toml").exists()
    assert (tmp_path / "postgres_mst.toml").exists()
    assert (tmp_path / "postgres_descriptions.toml").exists()
    assert (tmp_path / "postgres_embeddings.json").exists()

    # The documentation TOML is the full record: table descriptions AND column
    # descriptions. Only the embedding step narrows to table text.
    document = tomllib.loads((tmp_path / "postgres_descriptions.toml").read_text())
    assert document["database"]["type"] == "postgresql"  # engine, not target key
    assert document["tables"] == {
        "customers": {"description": "Desc of customers.", "columns": {"id": "Col id."}},
        "orders": {
            "description": "Desc of orders.",
            "columns": {"id": "Col id.", "customer_id": "Col customer_id."},
        },
    }

    # The embedded text is the table description alone -- the mock encodes each
    # input's length, so this pins down that no column text was appended.
    embeddings = json.loads((tmp_path / "postgres_embeddings.json").read_text())
    assert embeddings["customers"][1] == float(len("Desc of customers."))

    # customers.id + orders.id + orders.customer_id
    assert summary["column_description_count"] == 3
    schema_toml = (tmp_path / "postgres.toml").read_text()
    assert "Desc of customers." in schema_toml
    assert "Col customer_id." in schema_toml
    assert 'description_generated_at = ""' not in schema_toml


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_postgres_use_mst_false_reads_graph(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    monkeypatch.setattr("backend.src.core.ingestion.pipeline.get_adapter", lambda db_type: _FakeAdapter(schema))
    _patch_default_paths(monkeypatch, tmp_path)

    summary = await run_ingestion_pipeline(DatabaseType.POSTGRESQL, use_mst=False)

    assert summary["used_mst"] is False
    assert summary["description_count"] == 2


@pytest.mark.asyncio
async def test_run_ingestion_pipeline_mongo_always_uses_graph_no_mst(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.MONGODB)
    monkeypatch.setattr("backend.src.core.ingestion.pipeline.get_adapter", lambda db_type: _FakeAdapter(schema))
    _patch_default_paths(monkeypatch, tmp_path)

    # use_mst=True requested, but Mongo has no MST, so it must fall back to the graph.
    summary = await run_ingestion_pipeline(DatabaseType.MONGODB, use_mst=True)

    assert summary["used_mst"] is False
    assert summary["mst_path"] is None
    assert not (tmp_path / "mongo_mst.toml").exists()
    assert (tmp_path / "mongo_graph.toml").exists()

    # Column descriptions land in the collections' schema TOML for Mongo too.
    assert summary["column_description_count"] == 3
    schema_toml = (tmp_path / "mongo.toml").read_text()
    assert "Desc of customers." in schema_toml
    assert "Col customer_id." in schema_toml


def _counting_adapter(schema, calls):
    """Adapter factory that records every time the pipeline asks for a connection."""

    def factory(db_type):
        calls.append(db_type)
        return _FakeAdapter(schema)

    return factory


@pytest.mark.asyncio
async def test_second_run_is_skipped_when_embeddings_already_exist(tmp_path, monkeypatch):
    """Selecting a database in the frontend triggers ingestion every time; once
    the embeddings are on disk the whole flow must be skipped, so no database
    round-trip and no model calls happen."""
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    calls: list = []
    monkeypatch.setattr(
        "backend.src.core.ingestion.pipeline.get_adapter", _counting_adapter(schema, calls)
    )
    _patch_default_paths(monkeypatch, tmp_path)

    first = await run_ingestion_pipeline(DatabaseType.POSTGRESQL)
    second = await run_ingestion_pipeline(DatabaseType.POSTGRESQL)

    assert first["skipped"] is False
    assert second["skipped"] is True
    # only the first run touched the database
    assert [as_target(c).key for c in calls] == ["postgres"]
    # The reused summary is shaped exactly like a real one, so callers that index
    # into it (the frontend does) keep working.
    assert sorted(second) == sorted(first)
    assert second["embedding_count"] == first["embedding_count"]
    assert second["table_count"] == first["table_count"]


@pytest.mark.asyncio
async def test_force_reingests_even_when_embeddings_exist(tmp_path, monkeypatch):
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    calls: list = []
    monkeypatch.setattr(
        "backend.src.core.ingestion.pipeline.get_adapter", _counting_adapter(schema, calls)
    )
    _patch_default_paths(monkeypatch, tmp_path)

    await run_ingestion_pipeline(DatabaseType.POSTGRESQL)
    forced = await run_ingestion_pipeline(DatabaseType.POSTGRESQL, force=True)

    assert forced["skipped"] is False
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, reason",
    [("{}", "empty"), ("{not json", "corrupt")],
)
async def test_unusable_embeddings_file_does_not_count_as_ingested(
    tmp_path, monkeypatch, content, reason
):
    """An embeddings file that exists but holds nothing usable must not be
    mistaken for a finished run."""
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    calls: list = []
    monkeypatch.setattr(
        "backend.src.core.ingestion.pipeline.get_adapter", _counting_adapter(schema, calls)
    )
    _patch_default_paths(monkeypatch, tmp_path)

    (tmp_path / "postgres_embeddings.json").write_text(content)
    files.clear_cache()

    summary = await run_ingestion_pipeline(DatabaseType.POSTGRESQL)

    assert summary["skipped"] is False, f"{reason} embeddings file was treated as ingested"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_skip_requires_the_schema_too(tmp_path, monkeypatch):
    """Embeddings without a schema TOML is a half-written state; re-run rather
    than report a successful ingestion that cannot be served."""
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    calls: list = []
    monkeypatch.setattr(
        "backend.src.core.ingestion.pipeline.get_adapter", _counting_adapter(schema, calls)
    )
    _patch_default_paths(monkeypatch, tmp_path)

    await run_ingestion_pipeline(DatabaseType.POSTGRESQL)
    (tmp_path / "postgres.toml").unlink()

    assert (await run_ingestion_pipeline(DatabaseType.POSTGRESQL))["skipped"] is False
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_embeddings_path_that_is_a_directory_does_not_crash(tmp_path, monkeypatch):
    """A bind mount can leave a directory where the embeddings file should be.
    That must read as 'not ingested', not raise IsADirectoryError out of the
    /databases listing."""
    schema = _fake_schema(DatabaseType.POSTGRESQL)
    calls: list = []
    monkeypatch.setattr(
        "backend.src.core.ingestion.pipeline.get_adapter", _counting_adapter(schema, calls)
    )
    _patch_default_paths(monkeypatch, tmp_path)
    (tmp_path / "postgres_embeddings.json").mkdir()

    assert existing_ingestion(DatabaseType.POSTGRESQL) is None
