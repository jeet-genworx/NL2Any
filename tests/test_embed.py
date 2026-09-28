"""Tests for embedding table descriptions via the local MiniLM embedding model."""

import json
from pathlib import Path

import pytest

from backend.src.config import settings
from backend.src.data.models.schema import DatabaseType
from backend.src.core.ingestion.schema.describe import table_descriptions
from backend.src.core.ingestion.schema.embed import embed_descriptions
from backend.src.data.repositories import paths
from backend.src.data.repositories.description_repository import load_descriptions
from backend.src.data.repositories.embedding_repository import save_embeddings


class _RecordingEmbeddingProvider:
    """Mock embedding provider that records call order and returns deterministic vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), float(i)] for i, text in enumerate(texts)]


@pytest.mark.asyncio
async def test_embed_descriptions_preserves_table_order():
    descriptions = {
        "customers": "Stores customer records.",
        "orders": "Tracks customer orders.",
    }
    provider = _RecordingEmbeddingProvider()

    embeddings = await embed_descriptions(descriptions, provider=provider)

    assert list(embeddings.keys()) == ["customers", "orders"]
    # One batched call, texts passed in the same order as the input dict.
    assert len(provider.calls) == 1
    assert provider.calls[0] == ["Stores customer records.", "Tracks customer orders."]
    assert embeddings["customers"] == [len("Stores customer records."), 0.0]
    assert embeddings["orders"] == [len("Tracks customer orders."), 1.0]


@pytest.mark.asyncio
async def test_embed_descriptions_empty_input_returns_empty():
    provider = _RecordingEmbeddingProvider()
    embeddings = await embed_descriptions({}, provider=provider)

    assert embeddings == {}
    assert provider.calls == []


def test_load_descriptions_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_descriptions(tmp_path / "missing.toml")


def test_load_descriptions_roundtrip(tmp_path):
    path = tmp_path / "postgres_descriptions.toml"
    path.write_text(
        '[tables.customers]\n'
        'description = "desc"\n'
        '[tables.customers.columns]\n'
        'id = "Primary key."\n'
    )

    loaded = load_descriptions(path)

    assert loaded["customers"].description == "desc"
    assert loaded["customers"].columns == {"id": "Primary key."}


def test_load_descriptions_reads_collections_section(tmp_path):
    """MongoDB documentation is stored under [collections.*]; the caller does
    not have to know which section it was written to."""
    path = tmp_path / "mongo_descriptions.toml"
    path.write_text(
        '[collections.customers]\n'
        'description = "Customer documents."\n'
        '[collections.customers.columns]\n'
        '"address.city" = "City of residence."\n'
    )

    loaded = load_descriptions(path)

    assert loaded["customers"].description == "Customer documents."
    assert loaded["customers"].columns == {"address.city": "City of residence."}


def test_load_descriptions_skips_undescribed_columns(tmp_path):
    """Column slots still awaiting a description are dropped on load, so an
    empty string never reaches the schema TOML as a description."""
    path = tmp_path / "postgres_descriptions.toml"
    path.write_text(
        '[tables.customers]\n'
        'description = "desc"\n'
        '[tables.customers.columns]\n'
        'id = "Primary key."\n'
        'city = ""\n'
    )

    loaded = load_descriptions(path)

    assert loaded["customers"].columns == {"id": "Primary key."}


def test_load_descriptions_accepts_flat_string_shape(tmp_path):
    """A table mapped straight to a string still loads, carrying no columns."""
    path = tmp_path / "postgres_descriptions.toml"
    path.write_text('[tables]\ncustomers = "desc"\n')

    loaded = load_descriptions(path)

    assert loaded["customers"].description == "desc"
    assert loaded["customers"].columns == {}


def test_load_descriptions_then_embed_uses_table_text_only(tmp_path):
    """The embedding input is the table description; column text is excluded."""
    path = tmp_path / "postgres_descriptions.toml"
    path.write_text(
        '[tables.customers]\n'
        'description = "Customer records."\n'
        '[tables.customers.columns]\n'
        'id = "Primary key."\n'
        'city = "Where they live."\n'
    )
    provider = _RecordingEmbeddingProvider()

    import asyncio

    asyncio.run(
        embed_descriptions(table_descriptions(load_descriptions(path)), provider=provider)
    )

    assert provider.calls == [["Customer records."]]


def test_save_embeddings_writes_table_name_keyed_json(tmp_path):
    out_path = tmp_path / "postgres_embeddings.json"
    save_embeddings({"customers": [0.1, 0.2], "orders": [0.3, 0.4]}, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data == {"customers": [0.1, 0.2], "orders": [0.3, 0.4]}


def test_embeddings_path():
    # Both write to the exact files query_processing's live EmbeddingStore reads.
    assert paths.embeddings_path(DatabaseType.POSTGRESQL) == Path(
        settings.postgres_embeddings_path
    )
    assert paths.embeddings_path(DatabaseType.MONGODB) == Path(
        settings.mongo_embeddings_path
    )
