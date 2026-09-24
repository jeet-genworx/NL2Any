"""Tests for in-memory semantic (embedding-based) schema object retriever."""

import pytest

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from query_processing.retrieval.semantic import SemanticRetriever


class _FakeEmbeddingProvider:
    """Returns a fixed, caller-supplied vector for any query -- lets tests
    control cosine similarity deterministically without a live model."""

    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.query_vector for _ in texts]


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="company",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                description="Client profiles and contact details",
                fields=[Field(name="id", type="integer")],
            ),
            SchemaObject(
                name="orders",
                kind=SchemaObjectKind.TABLE,
                description="Purchase transactions and status",
                fields=[Field(name="id", type="integer")],
            ),
            SchemaObject(
                name="support_tickets",
                kind=SchemaObjectKind.TABLE,
                description="Customer complaints and bug reports",
                fields=[Field(name="id", type="integer")],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_semantic_retriever_ranks_by_cosine_similarity():
    schema = _schema()
    embeddings = {
        "customers": [1.0, 0.0, 0.0],
        "orders": [0.0, 1.0, 0.0],
        "support_tickets": [0.0, 0.0, 1.0],
    }
    # Query vector points squarely at "orders".
    provider = _FakeEmbeddingProvider([0.0, 1.0, 0.0])
    retriever = SemanticRetriever(schema, embeddings, provider=provider)

    results = await retriever.retrieve("Find transactions by status", top_k=3)

    assert len(results) == 3
    assert results[0].object_name == "orders"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score
    assert provider.calls == [["Find transactions by status"]]


@pytest.mark.asyncio
async def test_semantic_retriever_respects_top_k():
    schema = _schema()
    embeddings = {
        "customers": [1.0, 0.0, 0.0],
        "orders": [0.9, 0.1, 0.0],
        "support_tickets": [0.0, 0.0, 1.0],
    }
    provider = _FakeEmbeddingProvider([1.0, 0.0, 0.0])
    retriever = SemanticRetriever(schema, embeddings, provider=provider)

    results = await retriever.retrieve("customer profile", top_k=2)

    assert len(results) == 2
    assert results[0].object_name == "customers"


@pytest.mark.asyncio
async def test_semantic_retriever_excludes_objects_without_embeddings():
    schema = _schema()
    # Only "customers" has a precomputed embedding.
    embeddings = {"customers": [1.0, 0.0, 0.0]}
    provider = _FakeEmbeddingProvider([1.0, 0.0, 0.0])
    retriever = SemanticRetriever(schema, embeddings, provider=provider)

    results = await retriever.retrieve("anything", top_k=10)

    assert [r.object_name for r in results] == ["customers"]


@pytest.mark.asyncio
async def test_semantic_retriever_empty_query_returns_default_objects_without_embedding_call():
    schema = _schema()
    embeddings = {"customers": [1.0, 0.0, 0.0]}
    provider = _FakeEmbeddingProvider([1.0, 0.0, 0.0])
    retriever = SemanticRetriever(schema, embeddings, provider=provider)

    results = await retriever.retrieve("   ", top_k=2)

    assert len(results) == 2
    assert all(r.score == 0.0 for r in results)
    assert provider.calls == []
