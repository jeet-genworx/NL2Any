"""Tests for VectorRetriever and cosine similarity."""

import math
from unittest.mock import MagicMock
import pytest

from query_processing.retrieval.store import EmbeddingStore
from query_processing.retrieval.vector import VectorRetriever, cosine_similarity


def test_cosine_similarity_identical():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v2), 1.0)


def test_cosine_similarity_orthogonal():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v2), 0.0)


def test_cosine_similarity_dimension_mismatch():
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="different dimensions"):
        cosine_similarity(v1, v2)


def test_retrieval_threshold_strictly_greater():
    # Setup mock store with known similarity values
    # Let query vector be [1.0, 0.0, 0.0]
    query_vec = [1.0, 0.0, 0.0]

    # Target table vectors:
    # table_above: cos = 0.85 (> 0.80) -> INCLUDE
    # table_exact: cos = 0.80 (== 0.80) -> EXCLUDE (strictly > 0.80 required!)
    # table_below: cos = 0.75 (< 0.80) -> EXCLUDE
    theta_above = math.acos(0.85)
    v_above = [0.85, math.sin(theta_above), 0.0]

    theta_exact = math.acos(0.80)
    v_exact = [0.80, math.sin(theta_exact), 0.0]

    theta_below = math.acos(0.75)
    v_below = [0.75, math.sin(theta_below), 0.0]

    mock_store = MagicMock(spec=EmbeddingStore)
    mock_store.all_embeddings.return_value = {
        "table_above": v_above,
        "table_exact": v_exact,
        "table_below": v_below,
    }

    retriever = VectorRetriever(store=mock_store, similarity_threshold=0.80)
    results = retriever.retrieve(query_vec)

    assert len(results) == 1
    assert results[0].table_name == "table_above"
    assert results[0].similarity > 0.80
    assert results[0].rank == 1


def test_retrieval_ranking_and_top_15_enforcement():
    query_vec = [1.0, 0.0]
    # Create 20 candidate tables with varying similarities > 0.80
    table_embeddings = {}
    for i in range(1, 21):
        # similarities from 0.81 up to 0.99
        sim = 0.80 + (i * 0.009)
        theta = math.acos(sim)
        table_embeddings[f"table_{i:02d}"] = [sim, math.sin(theta)]

    mock_store = MagicMock(spec=EmbeddingStore)
    mock_store.all_embeddings.return_value = table_embeddings

    retriever = VectorRetriever(store=mock_store, similarity_threshold=0.80, max_candidates=15)
    results = retriever.retrieve(query_vec)

    # Must be capped at top 15
    assert len(results) == 15

    # Must be ordered descending by similarity
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)

    # Ranks must be 1 to 15
    assert [r.rank for r in results] == list(range(1, 16))
    assert results[0].table_name == "table_20"


def test_retrieval_zero_candidates():
    query_vec = [1.0, 0.0, 0.0]
    mock_store = MagicMock(spec=EmbeddingStore)
    mock_store.all_embeddings.return_value = {
        "customers": [0.0, 1.0, 0.0],  # sim = 0.0
        "orders": [0.0, 0.0, 1.0],     # sim = 0.0
    }

    retriever = VectorRetriever(store=mock_store, similarity_threshold=0.80)
    results = retriever.retrieve(query_vec)

    assert results == []
