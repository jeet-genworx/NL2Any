"""Vector similarity retrieval for table candidates."""

import logging
import math
from query_processing.core.config import settings
from query_processing.models.pipeline import CandidateTable
from query_processing.retrieval.store import EmbeddingStore

logger = logging.getLogger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vectors have different dimensions: {len(v1)} != {len(v2)}")

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class VectorRetriever:
    """Retrieves candidate database tables using cosine similarity over table embeddings."""

    def __init__(
        self,
        store: EmbeddingStore,
        similarity_threshold: float | None = None,
        max_candidates: int | None = None,
    ) -> None:
        self.store = store
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        self.max_candidates = (
            max_candidates
            if max_candidates is not None
            else settings.max_candidate_tables
        )

    def retrieve(self, query_vector: list[float]) -> list[CandidateTable]:
        """Compute cosine similarity against all tables, filter > threshold, rank descending, and limit."""
        table_embeddings = self.store.all_embeddings()
        scored: list[tuple[str, float]] = []

        for table_name, table_vec in table_embeddings.items():
            sim = cosine_similarity(query_vector, table_vec)
            # CRITICAL RULE: similarity > threshold (strictly greater than, NOT >=)
            if sim > self.similarity_threshold:
                scored.append((table_name, sim))

        # Rank surviving candidates by similarity descending
        scored.sort(key=lambda item: item[1], reverse=True)

        # Enforce max candidates limit (top 15)
        if len(scored) > self.max_candidates:
            scored = scored[: self.max_candidates]

        results: list[CandidateTable] = []
        for rank, (table_name, score) in enumerate(scored, start=1):
            results.append(
                CandidateTable(
                    table_name=table_name,
                    similarity=round(score, 4),
                    rank=rank,
                )
            )

        logger.info(
            "Vector retrieval found %d candidate tables above threshold %.2f",
            len(results),
            self.similarity_threshold,
        )
        return results
