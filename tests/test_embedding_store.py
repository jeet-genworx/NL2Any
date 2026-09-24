"""Tests for EmbeddingStore."""

import json
from pathlib import Path
import pytest

from query_processing.retrieval.store import EmbeddingStore


def test_embedding_store_valid_json(tmp_path: Path):
    file_path = tmp_path / "postgres_embeddings.json"
    data = {
        "customers": [0.1] * 384,
        "orders": [0.2] * 384,
    }
    file_path.write_text(json.dumps(data), encoding="utf-8")

    store = EmbeddingStore(file_path=file_path, expected_dim=384)
    embeddings = store.load()

    assert len(embeddings) == 2
    assert "customers" in embeddings
    assert "orders" in embeddings
    assert len(store.get("customers")) == 384
    assert store.get("non_existent") is None
    assert len(store.all_embeddings()) == 2


def test_embedding_store_missing_file(tmp_path: Path):
    store = EmbeddingStore(file_path=tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Embedding file not found"):
        store.load()


def test_embedding_store_malformed_json(tmp_path: Path):
    file_path = tmp_path / "malformed.json"
    file_path.write_text("{not valid json", encoding="utf-8")

    store = EmbeddingStore(file_path=file_path)
    with pytest.raises(ValueError, match="Malformed JSON"):
        store.load()


def test_embedding_store_non_dict_root(tmp_path: Path):
    file_path = tmp_path / "root_list.json"
    file_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    store = EmbeddingStore(file_path=file_path)
    with pytest.raises(ValueError, match="root must be a JSON object"):
        store.load()


def test_embedding_store_empty_file(tmp_path: Path):
    file_path = tmp_path / "empty.json"
    file_path.write_text(json.dumps({}), encoding="utf-8")

    store = EmbeddingStore(file_path=file_path)
    with pytest.raises(ValueError, match="empty or contains no tables"):
        store.load()


def test_embedding_store_wrong_dimension(tmp_path: Path):
    file_path = tmp_path / "wrong_dim.json"
    data = {
        "customers": [0.1] * 128,  # only 128 instead of 384
    }
    file_path.write_text(json.dumps(data), encoding="utf-8")

    store = EmbeddingStore(file_path=file_path, expected_dim=384)
    with pytest.raises(ValueError, match="has dimension 128, expected 384"):
        store.load()


def test_embedding_store_non_numeric_values(tmp_path: Path):
    file_path = tmp_path / "non_numeric.json"
    vec = [0.1] * 383 + ["string_value"]
    file_path.write_text(json.dumps({"customers": vec}), encoding="utf-8")

    store = EmbeddingStore(file_path=file_path)
    with pytest.raises(ValueError, match="non-numeric value"):
        store.load()


def test_embedding_store_invalid_table_name(tmp_path: Path):
    file_path = tmp_path / "invalid_key.json"
    # Empty string table name
    file_path.write_text(json.dumps({"": [0.1] * 384}), encoding="utf-8")

    store = EmbeddingStore(file_path=file_path)
    with pytest.raises(ValueError, match="Invalid table name"):
        store.load()
