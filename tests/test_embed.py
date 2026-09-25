"""Tests for embedding table descriptions via the local MiniLM embedding model."""

import json
from pathlib import Path

import pytest

from query_processing.core.config import settings
from query_processing.models.schema import DatabaseType
from ingestion.schema.embed import (
    embed_descriptions,
    load_descriptions,
    save_embeddings,
    get_default_embeddings_path,
)


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
        load_descriptions(tmp_path / "missing.json")


def test_load_descriptions_roundtrip(tmp_path):
    path = tmp_path / "postgres_descriptions.json"
    path.write_text(json.dumps({"customers": "desc"}))

    assert load_descriptions(path) == {"customers": "desc"}


def test_save_embeddings_writes_table_name_keyed_json(tmp_path):
    out_path = tmp_path / "postgres_embeddings.json"
    save_embeddings({"customers": [0.1, 0.2], "orders": [0.3, 0.4]}, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data == {"customers": [0.1, 0.2], "orders": [0.3, 0.4]}


def test_get_default_embeddings_path():
    # Both write to the exact files query_processing's live EmbeddingStore reads.
    assert get_default_embeddings_path(DatabaseType.POSTGRESQL) == Path(
        settings.postgres_embeddings_path
    )
    assert get_default_embeddings_path(DatabaseType.MONGODB) == Path(
        settings.mongo_embeddings_path
    )
