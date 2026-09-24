"""In-memory semantic (embedding-based) schema object retriever.

Ranks candidate tables/collections by cosine similarity between the
question's embedding and each table's description embedding -- both from
the local MiniLM model served by KoboldCpp -- instead of BM25 lexical
keyword overlap. The embeddings are produced ahead of time by the ingestion
pipeline (ingestion/schema/describe.py + embed.py) and loaded here.
"""

import math
from pydantic import BaseModel

from query_processing.core.config import settings
from query_processing.models.schema import DatabaseSchema, SchemaObject
from query_processing.providers.model.base import EmbeddingProvider
from query_processing.providers.model.embedding import KoboldCppEmbeddingProvider


class RetrievalResult(BaseModel):
    """Result of semantic retrieval for a schema object."""

    object_name: str
    score: float
    schema_object: SchemaObject


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticRetriever:
    """Ranks schema objects by embedding similarity to the question."""

    def __init__(
        self,
        schema: DatabaseSchema,
        embeddings: dict[str, list[float]],
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.schema = schema
        self.objects = schema.objects
        self.embeddings = embeddings
        self.provider = provider or KoboldCppEmbeddingProvider()
        self._objects_by_name = {obj.name: obj for obj in self.objects}

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve schema objects ranked by cosine similarity to the query.

        Objects with no precomputed embedding (e.g. added since the last
        ingestion run) are excluded from ranking, not just scored zero.
        """
        k = top_k if top_k is not None else settings.retrieval_top_k

        scorable = [
            (name, self._objects_by_name[name])
            for name in self.embeddings
            if name in self._objects_by_name
        ]
        if not scorable or not query.strip():
            return [
                RetrievalResult(object_name=obj.name, score=0.0, schema_object=obj)
                for obj in self.objects[:k]
            ]

        [query_vector] = await self.provider.embed([query])

        scored = [
            (obj, _cosine_similarity(query_vector, self.embeddings[name]))
            for name, obj in scorable
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RetrievalResult(object_name=obj.name, score=score, schema_object=obj)
            for obj, score in scored[:k]
        ]
